# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm.bindings import DataType, LayerType
from tensorrt_llm.bindings import ModelConfig as ModelConfigCpp
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping

_BATCH = 4
_TOKENS_PER_BLOCK = 16
_TOKENS_PER_REQUEST = 5 * _TOKENS_PER_BLOCK
_SLEEP_CYCLES = 2_000_000_000
_STAGING_SOURCES = ("copy_idx", "page_table")


def _build_manager() -> KVCacheManagerV2:
    model_config = ModelConfigCpp(
        vocab_size=32000,
        num_layers=2,
        num_attention_layers=2,
        num_rnn_layers=0,
        num_heads=2,
        hidden_size=32,
        data_type=DataType.HALF,
    )
    model_config.layer_types = [LayerType.ATTENTION] * 2
    model_config.set_num_kv_heads(2)

    return KVCacheManagerV2(
        kv_cache_config=KvCacheConfig(
            max_tokens=4096,
            free_gpu_memory_fraction=0.1,
            enable_block_reuse=False,
            enable_swa_scratch_reuse=True,
            tokens_per_block=_TOKENS_PER_BLOCK,
        ),
        kv_cache_type=CacheType.SELF,
        num_layers=2,
        num_kv_heads=2,
        head_dim=16,
        tokens_per_block=_TOKENS_PER_BLOCK,
        max_seq_len=128,
        max_batch_size=2 * _BATCH,
        mapping=Mapping(world_size=1, tp_size=1, pp_size=1),
        dtype=DataType.HALF,
        model_config=model_config,
        max_beam_width=1,
    )


def _empty_offsets(manager: KVCacheManagerV2) -> torch.Tensor:
    return torch.full(
        (
            manager.num_attention_op_pools,
            _BATCH,
            2,
            manager.max_blocks_per_seq,
        ),
        -12345,
        dtype=torch.int32,
        device="cuda",
    )


def _copy_offsets(
    manager: KVCacheManagerV2,
    dst_tensor: torch.Tensor,
    request_ids: list[int],
    copy_idx: torch.Tensor,
) -> None:
    manager._copy_batch_block_offsets_per_layer(
        dst_tensor,
        request_ids,
        copy_idx,
        len(request_ids),
        len(request_ids),
    )


def _overwrite_source(manager: KVCacheManagerV2, source: str, copy_idx: torch.Tensor) -> None:
    if source == "copy_idx":
        copy_idx.fill_(manager.max_batch_size - 1)
    elif source == "scratch_metadata":
        manager._host_scratch_begs_staging.fill_(777777)
        manager._host_scratch_ends_staging.fill_(777777)
        manager._host_scratch_slots_staging.fill_(777777)
    elif source == "page_table":
        manager.host_kv_cache_block_offsets.fill_(777777)
    else:
        raise AssertionError(f"Unknown staging source: {source}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
@pytest.mark.parametrize("source", _STAGING_SOURCES)
def test_swa_staging_snapshot_survives_host_overwrite(source: str) -> None:
    manager = _build_manager()
    request_ids = list(range(1, 1 + _BATCH))

    try:
        manager.add_dummy_requests(
            request_ids=request_ids,
            token_nums=[_TOKENS_PER_REQUEST] * _BATCH,
        )
        copy_idx = manager.index_mapper.get_copy_index(request_ids, _BATCH, 1)

        baseline = _empty_offsets(manager)
        with torch.cuda.stream(manager._stream):
            _copy_offsets(manager, baseline, request_ids, copy_idx)
        manager._stream.synchronize()
        baseline = baseline.clone()

        raced = _empty_offsets(manager)
        with torch.cuda.stream(manager._stream):
            torch.cuda._sleep(_SLEEP_CYCLES)
            _copy_offsets(manager, raced, request_ids, copy_idx)

        _overwrite_source(manager, source, copy_idx)
        manager._stream.synchronize()

        torch.testing.assert_close(raced, baseline)
    finally:
        manager.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_swa_scratch_metadata_uses_fresh_host_snapshots(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _build_manager()
    request_ids = list(range(1, 1 + _BATCH))
    observed_sources: dict[str, torch.Tensor] = {}
    original_copy = torch.Tensor.copy_

    def record_scratch_copy(
        destination: torch.Tensor,
        source: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        staging_targets = {
            manager._device_scratch_begs_staging.data_ptr(): "begs",
            manager._device_scratch_ends_staging.data_ptr(): "ends",
            manager._device_scratch_slots_staging.data_ptr(): "slots",
        }
        source_name = staging_targets.get(destination.data_ptr())
        if source_name is not None:
            observed_sources[source_name] = source
        return original_copy(destination, source, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", record_scratch_copy)
    try:
        manager.add_dummy_requests(
            request_ids=request_ids,
            token_nums=[_TOKENS_PER_REQUEST] * _BATCH,
        )
        with torch.cuda.stream(manager._stream):
            manager._copy_scratch_metadata_to_device(request_ids, _BATCH)
        manager._stream.synchronize()

        assert set(observed_sources) == {"begs", "ends", "slots"}
        assert observed_sources["begs"].data_ptr() != manager._host_scratch_begs_staging.data_ptr()
        assert observed_sources["ends"].data_ptr() != manager._host_scratch_ends_staging.data_ptr()
        assert (
            observed_sources["slots"].data_ptr() != manager._host_scratch_slots_staging.data_ptr()
        )
    finally:
        manager.shutdown()
