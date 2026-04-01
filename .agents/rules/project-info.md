Antigravity Prompt: CUSTOM_JLX Sparse Ternary Framework

This prompt summarizes the core architecture and coding principles of the CUSTOM_JLX project. It is designed to ensure that an AI assistant fully understands the repository's unique constraints and can contribute reliably to the codebase.

Project Overview:
CUSTOM_JLX is a Sparse Ternary LLM Training Framework built from scratch for Apple Silicon using Metal GPU kernels. It aims to train large-scale models (1B to 7B) under extreme memory constraints.

Key Objective: Enable LLM training on consumer-grade Mac hardware.
Core Technologies:
1. Ternary Weights: Weights are quantized to -1, 0, +1, drastically reducing memory footprint.
2. Extreme Sparsity: Maintains approximately 1 percent density (99 percent sparsity) using CSR and CSC formats.
3. HiMA (Hierarchical Memory-Aware): Dynamically tiers weight blocks into Hot (Metal buffer), Warm (CPU mmap), and Cold (SSD) based on gradient magnitude.
4. Custom Metal Kernels: Optimized MSL kernels for sparse forward and backward passes.

Core Architecture and Implementation Rules:

1. Weight Representation and Data Structures:
Sparse Storage: Strictly follow the CSR (Compressed Sparse Row) and CSC (Compressed Sparse Column) formats as implemented in SparseTernaryLinear.h.
Forward Pass: Uses CSR (row_ptrs, indices).
Backward Pass (Input Gradients): Uses CSC (col_ptrs, indices_col).
Ternary Packing: DType::Ternary in Tensor.h refers to 2-bit packing. Weights must be unpacked during GPU execution.

2. HiMA (Hierarchical Memory-Aware) Training:
Blocks are moved between tiers according to TierManager.h logic.
Hot: Resident in Metal buffers; precision training every step.
Warm: CPU-mapped; loose training every K steps.
Cold: SSD-stored or Frozen; participates in forward pass only using stale caches.
Any new layer or operation must integrate with the BlockInfo management system of HiMA.

3. Metal GPU Kernel Optimization (ops.metal):
All operations are written in Metal Shading Language (MSL).
Sparse Operations: Use atomic_fetch_add_explicit or similar synchronization to prevent race conditions during sparse updates while maintaining high parallelism.
BFloat16: Leverage bfloat types for the optimal balance between performance and numerical stability on Apple Silicon.

4. Optimizer (Adafactor):
Follow the custom implementation in Adafactor.h which includes 1-bit momentum compression.
Prioritize fused_adam_update patterns to minimize memory overhead during sparse updates.

Coding Guidelines:

1. Performance Over Purity: Direct memory control and kernel optimization take precedence over high-level abstraction. A large main.mm is acceptable if it yields significant performance gains.
2. Explicit Memory Management: Tensor objects manage MTLBuffer directly. Avoid unnecessary copies. Use invalidate() for explicit resource deallocation when needed.
3. Think Sparse: Every linear operation must be conceived as an index-based sparse operation. Avoid falling back to dense matrix logic, as it defeats the project's purpose.
4. Hardware Targeting: Never write CUDA or CPU-only code for core logic. Always target Metal API features specific to Apple Silicon.

Technical Constraints and Warnings:

1. Apple Silicon Only: The codebase is not cross-platform.
2. Index Precision: Use uint32_t or int32_t carefully for large-scale indexing to avoid overflows or unnecessary memory waste.
3. Sparsity Maintenance: Ensure the resparsify logic is called periodically so that the weight density does not exceed the init_density target during training.

When adding features or fixing bugs, prioritize performance, memory efficiency, and the consistency of the Sparse Ternary architecture.

Based on a detailed analysis of the repository, the soul of this project lies in Index-based Sparse Operations and Dynamic Memory Hierarchies. This prompt serves as the definitive guide for maintaining that vision.
