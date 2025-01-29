/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <torch/library.h>
#include "fbgemm_gpu/split_embeddings_utils.h"
#include "fbgemm_gpu/utils/ops_utils.h"

using Tensor = at::Tensor;

DLL_PUBLIC std::tuple<int32_t, uint32_t> adjust_info_B_num_bits(
    int32_t B,
    int32_t T) {
  int32_t info_B_num_bits = DEFAULT_INFO_B_NUM_BITS;
  uint32_t info_B_mask = DEFAULT_INFO_B_MASK;
  uint32_t max_T = MAX_T;
  uint32_t max_B = MAX_B;
  bool invalid_T = T > max_T;
  bool invalid_B = B > max_B;

  TORCH_CHECK(
      !(invalid_T && invalid_B),
      "Not enough infos bits to accommodate T and B. Default num bits = ",
      DEFAULT_INFO_NUM_BITS);

  if (invalid_T) {
    // Reduce info_B_num_bits
    while (invalid_T && !invalid_B && info_B_num_bits > 0) {
      info_B_num_bits--;
      max_T = ((max_T + 1) << 1) - 1;
      max_B = ((max_B + 1) >> 1) - 1;
      invalid_T = T > max_T;
      invalid_B = B > max_B;
    }
  } else if (invalid_B) {
    // Increase info_B_num_bits
    while (!invalid_T && invalid_B && info_B_num_bits < DEFAULT_INFO_NUM_BITS) {
      info_B_num_bits++;
      max_T = ((max_T + 1) >> 1) - 1;
      max_B = ((max_B + 1) << 1) - 1;
      invalid_T = T > max_T;
      invalid_B = B > max_B;
    }
  }

  TORCH_CHECK(
      !invalid_T && !invalid_B,
      "Not enough infos bits to accommodate T and B. Default num bits = ",
      DEFAULT_INFO_NUM_BITS);

  // Recompute info_B_mask using new info_B_num_bits
  info_B_mask = (1u << info_B_num_bits) - 1;

  return {info_B_num_bits, info_B_mask};
}

namespace {

std::tuple<Tensor /*row_output_offsets*/, Tensor /*b_t_map*/>
generate_vbe_metadata_cpu(
    const Tensor& B_offsets,
    const Tensor& B_offsets_rank_per_feature,
    const Tensor& output_offsets_feature_rank,
    const Tensor& D_offsets,
    const int64_t D,
    const bool nobag,
    const c10::SymInt max_B_feature_rank,
    const int64_t info_B_num_bits,
    const c10::SymInt total_B) {

  TENSOR_ON_CPU(B_offsets);
  TENSORS_ON_SAME_DEVICE(B_offsets, B_offsets_rank_per_feature);
  TENSORS_ON_SAME_DEVICE(B_offsets, output_offsets_feature_rank);
  TENSORS_ON_SAME_DEVICE(B_offsets, D_offsets);

  TENSOR_NDIM_EQUALS(B_offsets, 1);
  TENSOR_NDIM_EQUALS(B_offsets_rank_per_feature, 2);
  TENSOR_NDIM_EQUALS(output_offsets_feature_rank, 1);

  const auto T = B_offsets.numel() - 1;
  TORCH_CHECK(T > 0, "generate_vbe_metadata: Invalid T ", T);
  TORCH_CHECK(D_offsets.numel() == T + 1)
  const auto num_ranks = B_offsets_rank_per_feature.size(1) - 1;
  TORCH_CHECK(
      num_ranks > 0, "generate_vbe_metadata: Invalid num_ranks ", num_ranks);
  TORCH_CHECK(B_offsets_rank_per_feature.size(0) == T);
  TORCH_CHECK(output_offsets_feature_rank.numel() == num_ranks * T + 1);

  Tensor row_output_offsets =
      at::empty({total_B}, output_offsets_feature_rank.options());
  Tensor b_t_map = at::empty({total_B}, B_offsets.options());
    
    for (const auto t : c10::irange(T)) {
        // Get batch offset per table
        const auto B_start_t = B_offsets[t];
        const auto D_ = D_offsets[t + 1] - D_offsets[t];
        for (const auto r : c10::irange(num_ranks)) {
            // Get start index of batch offset
            const auto B_start_r_t = B_offsets_rank_per_feature[t][r];
            // Get batch size
            const auto B_r_t = B_offsets_rank_per_feature[t][r + 1] - B_start_r_t;
            // Offset output by feature
            const auto* output_offsets_feature = &output_offsets_feature_rank[r * T];
            
            for (const auto t : c10::irange(num_batches_r_t)) {
                // serialize batch id as index for the outputs
                const auto b_t = static_cast<int64_t>(B_start_t) + static_cast<int64_t>(B_start_r_t) + b;
                row_output_offsets[b_t] = output_offsets_feature[t] + b * static_cast<int64_t>(D_);
                // batch ID per table
                const auto b_ = B_start_r_t + b;
                *reinterpret_cast<uint32_t*>(&b_t_map[b_t]) = (reinterpret_cast<const uint32_t*>(&t)[0] << info_B_num_bits) | reinterpret_cast<const uint32_t*>(&b_)[0];
            }
        }
   }
  
  return {row_output_offsets, b_t_map};
}

std::tuple<int64_t, int64_t>
get_infos_metadata_cpu(Tensor unused, int64_t B, int64_t T) {
  return adjust_info_B_num_bits(B, T);
}

} // namespace

TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
  m.def(
      "transpose_embedding_input("
      "    Tensor hash_size_cumsum, "
      "    int total_hash_size_bits, "
      "    Tensor indices, "
      "    Tensor offsets, "
      "    bool nobag=False, "
      "    Tensor? vbe_b_t_map=None, "
      "    int info_B_num_bits=26, "
      "    int info_B_mask=0x2FFFFFF, "
      "    int total_unique_indices=-1, "
      "    bool is_index_select=False, "
      "    Tensor? total_L_offsets=None, "
      "    int fixed_L_per_warp=0, "
      "    int num_warps_per_feature=0"
      ") -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def("get_infos_metadata(Tensor unused, int B, int T) -> (int, int)");
  m.def(
      "generate_vbe_metadata("
      "    Tensor B_offsets, "
      "    Tensor B_offsets_rank_per_feature, "
      "    Tensor output_offsets_feature_rank, "
      "    Tensor D_offsets, "
      "    int D, "
      "    bool nobag, "
      "    SymInt max_B_feature_rank, "
      "    int info_B_num_bits, "
      "    SymInt total_B"
      ") -> (Tensor, Tensor)");
  DISPATCH_TO_CPU("generate_vbe_metadata", generate_vbe_metadata_cpu);
  DISPATCH_TO_CPU("get_infos_metadata", get_infos_metadata_cpu);
}
