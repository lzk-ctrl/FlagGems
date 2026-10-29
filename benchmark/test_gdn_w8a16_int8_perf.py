# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

import flag_gems
from benchmark.base import Benchmark
from tests.test_gdn_w8a16_int8 import (
    build_int8_gdn_decode_inputs,
    build_int8_gdn_prefill_inputs,
    torch_int8_state_gdn_decode,
    torch_int8_state_gdn_prefill,
)


class Int8StateGdnBenchmark(Benchmark):
    DEFAULT_DTYPES = [torch.bfloat16]
    DEFAULT_SHAPES = [
        (1, 8192, 32, 128),
        (3, 8192, 32, 128),
        ("update", 1, 3, 16, 128, 32, 128, 3),
        (1, 17, 16, 128, 32, 128, 1),
        (1, 1826, 16, 128, 32, 128, 2),
    ]
    DEFAULT_SHAPE_DESC = (
        "decode: T,packed,HV,V; prefill: B,T,H,K,HV,V,N; update: update,B,T,H,K,HV,V,N"
    )

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            torch.manual_seed(42)
            if shape[0] == "update":
                _, batch, tokens, query_heads, reduction, heads, columns, sequences = (
                    shape
                )
                if batch != 1 or sequences != tokens:
                    raise ValueError(
                        "update shapes require B=1 and one token per sequence"
                    )
                packed = 2 * query_heads * reduction + heads * columns
                _, reference, args = build_int8_gdn_decode_inputs(
                    (tokens, packed, heads, columns), dtype, reduction
                )
                q, k, v = torch.split(
                    args[0],
                    (query_heads * reduction, query_heads * reduction, heads * columns),
                    dim=-1,
                )
                qkv = (
                    q.reshape(1, tokens, query_heads, reduction).contiguous(),
                    k.reshape(1, tokens, query_heads, reduction).contiguous(),
                    v.reshape(1, tokens, heads, columns).contiguous(),
                )
                yield ("update", (qkv, *reference[1:])), (qkv, *args[1:])
            elif len(shape) == 7:
                _, reference, args = build_int8_gdn_prefill_inputs(shape, dtype)
                yield ("prefill", reference), args
            else:
                _, reference, args = build_int8_gdn_decode_inputs(shape, dtype)
                yield ("decode", reference), args


def torch_int8_state_gdn_baseline(reference, args):
    if reference[0] == "prefill":
        return torch_int8_state_gdn_prefill(**reference[1])
    else:
        return torch_int8_state_gdn_decode(*reference[1])


def run_int8_state_gdn(reference, args):
    if reference[0] == "prefill":
        return flag_gems.chunk_gated_delta_rule_w8a16_int8(*args)
    elif reference[0] == "update":
        tokens = args[0][0].shape[1]
        mixed = flag_gems.cat(
            tuple(tensor.view(tokens, -1) for tensor in args[0]), dim=1
        )
        return flag_gems.gdn_w8a16_int8(mixed, *args[1:])
    else:
        return flag_gems.gdn_w8a16_int8(*args)


@pytest.mark.gdn_w8a16_int8
@pytest.mark.skipif(
    "gdn_w8a16_int8" not in vars(flag_gems),
    reason="backend does not export INT8-state GDN",
)
def test_perf_gdn_w8a16_int8():
    bench = Int8StateGdnBenchmark(
        op_name="gdn_w8a16_int8", torch_op=torch_int8_state_gdn_baseline
    )
    bench.set_gems(run_int8_state_gdn)
    bench.run()
