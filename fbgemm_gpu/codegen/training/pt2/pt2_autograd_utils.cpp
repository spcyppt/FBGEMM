/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/TypeDefault.h>

using Tensor = at::Tensor;

namespace fbgemm_gpu {

////////////////////////////////////////////////////////////////////////////////
// Helper Functions
////////////////////////////////////////////////////////////////////////////////

Tensor reshape_vbe_output(
    const Tensor& grad_output,
    const c10::SymInt max_B,
    const Tensor& B_offsets_rank_per_feature,
    const Tensor& D_offsets) {
  /* FOR CPU VBE to use the same backend */
  const auto T = D_offsets.numel() - 1;
  int32_t total_D = 0;
  // find total_D to create output [max_B, total_D]
  for (int32_t t = 0; t < T; t++) {
    total_D += D_offsets[t + 1].item<int32_t>() - D_offsets[t].item<int32_t>();
  }
  auto grad_output_ = at::empty({max_B, total_D}, grad_output.options());
  // for each feature
  auto offset = 0;

  const int32_t R = B_offsets_rank_per_feature.size(1) - 1;
  for (int32_t r = 0; r < R; r++) {
    auto D_offset = 0;
    for (int32_t t = 0; t < T; t++) {
      const int32_t b_begin = B_offsets_rank_per_feature[t][r].item<int32_t>();
      const int32_t b_end =
          B_offsets_rank_per_feature[t][r + 1].item<int32_t>();
      const int32_t D =
          D_offsets[t + 1].item<int32_t>() - D_offsets[t].item<int32_t>();
      const int32_t b = b_end - b_begin;
      const int32_t num_elm = b * D;
      auto values = grad_output.slice(0, offset, offset + num_elm);
      values = values.reshape({b, D});
      grad_output_.index_put_(
          {at::indexing::Slice(b_begin, b_end),
           at::indexing::Slice(D_offset, D_offset + D)},
          values);
      D_offset += D;
      offset += num_elm;
    }
  }
  return grad_output_;
}
Tensor reshape_offsets(
    const Tensor& offsets,
    const Tensor& B_offsets_rank_per_feature,
    const c10::SymInt max_B,
    const int32_t T) {
    auto offsets_ = at::empty({T * max_B + 1}, offsets.options());
    auto begin = 0;
    for (int32_t t = 0; t < T; t++) {
        const auto end = B_offsets_rank_per_feature[t][B_offsets_rank_per_feature[t].numel()-1] + begin;
        const auto values = offsets.slice(0, begin, end);
        offsets_.index_put_({t * max_B, t * max_B + end - begin}, values);
        offsets_[t * max_B + end - begin : (t + 1) * max_B] = offsets_[end];
        begin = end;
    }
    offsets_[offsets.numel()-1] = offsets[offsets.numel()-1];
    return offsets_;
}


} // namespace fbgemm_gpu
