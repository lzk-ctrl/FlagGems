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

import logging
import math
from collections.abc import Sequence

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)
MAX_NORMALIZED_SIZE = 32768
MAX_TILE_ELEMENTS = 8192
MAX_GRID_SIZE = 65535


@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_w8a16_int8_kernel(
    output_ptr,
    input_ptr,
    weight_ptr,
    scale_ptr,
    num_rows,
    num_columns,
    eps,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_ITERATIONS: tl.constexpr,
):
    columns = tl.arange(0, BLOCK_N)
    column_mask = columns < num_columns
    weight = tl.load(weight_ptr + columns, column_mask, other=0).to(tl.float32)
    scale = tl.load(scale_ptr + columns // GROUP_SIZE, column_mask, other=0).to(
        tl.float32
    )
    # Reuse the dequantized weights across rows and grid-stride iterations.
    weight_tile = tl.broadcast_to((weight * scale)[None, :], (BLOCK_M, BLOCK_N))
    # A compile-time trip count removes the loop for a single tile per program.
    for iteration in range(NUM_ITERATIONS):
        tile = tl.program_id(0) + iteration * tl.num_programs(0)
        rows = tile * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets = rows[:, None].to(tl.int64) * num_columns + columns[None, :]
        mask = (rows[:, None] < num_rows) & column_mask[None, :]
        activation = tl.load(input_ptr + offsets, mask, other=0).to(tl.float32)
        variance = tl.sum(activation * activation, axis=1) / num_columns
        inverse_rms = tl.rsqrt(variance + eps)
        output = activation * inverse_rms[:, None] * weight_tile
        tl.store(output_ptr + offsets, output, mask)


def rms_norm_w8a16_int8(
    activation: torch.Tensor,
    normalized_shape: Sequence[int],
    weight_int8: torch.Tensor,
    weight_scale: torch.Tensor,
    eps: float = 1e-5,
    group_size: int = 128,
) -> torch.Tensor:
    """Normalize trailing dimensions with symmetric group-wise INT8 weights."""
    if activation.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("activation must be float16 or bfloat16")
    if weight_int8.dtype != torch.int8:
        raise TypeError("weight_int8 must have dtype int8")
    if weight_scale.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("weight_scale must be float16, bfloat16 or float32")
    if not normalized_shape or tuple(
        activation.shape[-len(normalized_shape) :]
    ) != tuple(normalized_shape):
        raise ValueError("normalized_shape must match the trailing input dimensions")
    num_columns = math.prod(normalized_shape)
    if not 0 < num_columns <= MAX_NORMALIZED_SIZE or group_size <= 0 or eps < 0:
        raise ValueError(
            "require 0 < normalized size <= 32768, group_size > 0 and eps >= 0"
        )
    if weight_int8.ndim != 1 or weight_int8.numel() != num_columns:
        raise ValueError(
            "weight_int8 must be a vector with one entry per normalized element"
        )
    if (
        weight_scale.ndim != 1
        or weight_scale.numel() != (num_columns + group_size - 1) // group_size
    ):
        raise ValueError("weight_scale must contain one entry per quantization group")
    if (
        not activation.is_cuda
        or activation.device != weight_int8.device
        or activation.device != weight_scale.device
    ):
        raise ValueError("input, weights and scales must be on the same HCU device")
    logger.debug("GEMS_HYGON RMS_NORM_W8A16_INT8")
    num_rows = activation.numel() // num_columns
    output = torch.empty(
        activation.shape, dtype=activation.dtype, device=activation.device
    )
    if num_rows == 0:
        return output
    activation = activation.contiguous()
    weight_int8 = weight_int8.contiguous()
    weight_scale = weight_scale.contiguous()
    block_n = 1 << (num_columns - 1).bit_length()
    if num_columns == 2048 or num_rows <= 16:
        block_m = 1
    elif num_columns == 256 and num_rows > 4096:
        block_m = 8
    else:
        block_m = min(4, max(1, MAX_TILE_ELEMENTS // block_n))
    num_tiles = (num_rows + block_m - 1) // block_m
    grid = (min(num_tiles, MAX_GRID_SIZE),)
    num_iterations = (num_tiles + grid[0] - 1) // grid[0]
    with torch_device_fn.device(activation.device):
        rms_norm_w8a16_int8_kernel[grid](
            output,
            activation,
            weight_int8,
            weight_scale,
            num_rows,
            num_columns,
            eps,
            GROUP_SIZE=group_size,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            NUM_ITERATIONS=num_iterations,
            num_warps=4,
        )
    return output
