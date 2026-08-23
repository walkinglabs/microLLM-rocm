#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <microllm/core/tensor.h>

namespace microllm::autograd {

struct GradientAccumulationRecord {
    std::string target_operation;
    std::string first_source;
    std::string last_add_source;
    Shape shape;
    std::uint64_t first_assignments = 0;
    std::uint64_t add_calls = 0;
    std::uint64_t materializations = 0;
    std::uint64_t sparse_embedding_add_calls = 0;
    std::uint64_t added_elements = 0;
    std::uint64_t materialized_elements = 0;
};

struct GradientAccumulationDiagnostics {
    std::vector<GradientAccumulationRecord> records;
    std::uint64_t first_assignments = 0;
    std::uint64_t add_calls = 0;
    std::uint64_t materializations = 0;
    std::uint64_t sparse_embedding_add_calls = 0;
    std::uint64_t added_elements = 0;
    std::uint64_t materialized_elements = 0;
};

void enable_gradient_accumulation_diagnostics(bool enabled) noexcept;
void reset_gradient_accumulation_diagnostics() noexcept;
[[nodiscard]] GradientAccumulationDiagnostics gradient_accumulation_diagnostics();
// Research control for same-revision A/B. Production default is enabled.
void enable_tied_embedding_sparse_add(bool enabled) noexcept;
[[nodiscard]] bool tied_embedding_sparse_add_enabled() noexcept;
// Research control for same-revision Attention layout A/B. Production default
// is enabled; disabling it restores the explicit transpose materializations.
void enable_attention_rope_layout_fusion(bool enabled) noexcept;
[[nodiscard]] bool attention_rope_layout_fusion_enabled() noexcept;

}  // namespace microllm::autograd
