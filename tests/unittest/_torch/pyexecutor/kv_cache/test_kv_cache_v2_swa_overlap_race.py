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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_swa_page_table_snapshot_survives_host_overwrite() -> None:
    manager = _build_manager()
    request_ids = list(range(1, 1 + _BATCH))

    try:
        manager.add_dummy_requests(
            request_ids=request_ids,
            token_nums=[_TOKENS_PER_REQUEST] * _BATCH,
        )

        baseline = _empty_offsets(manager)
        with torch.cuda.stream(manager._stream):
            manager.copy_batch_block_offsets(
                baseline,
                request_ids,
                1,
                _BATCH,
                _BATCH,
            )
        manager._stream.synchronize()
        baseline = baseline.clone()

        raced = _empty_offsets(manager)
        with torch.cuda.stream(manager._stream):
            torch.cuda._sleep(_SLEEP_CYCLES)
            manager.copy_batch_block_offsets(
                raced,
                request_ids,
                1,
                _BATCH,
                _BATCH,
            )

        manager.host_kv_cache_block_offsets.fill_(777777)
        manager._stream.synchronize()

        torch.testing.assert_close(raced, baseline)
    finally:
        manager.shutdown()
