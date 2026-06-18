# qr_v2

## Description

Implement batched square compact-Householder QR factorization.

Input is A, a batch x n x n CUDA tensor in torch.float32.

Return (H, tau) in the same compact Householder convention as torch.geqrf(A). H is a batch x n x n FP32 tensor containing R in its upper triangle and Householder vectors below the diagonal. tau is a batch x n FP32 tensor containing reflector coefficients. The checker materializes Q = torch.linalg.householder_product(H, tau), uses R_factor = triu(H), and validates the LAPACK-style QR factorization residual R_factor - Q.T @ A and orthogonality of Q. Since R_factor is extracted with triu, triangularity is part of the factorization check: if Q.T @ A has meaningful lower-triangular leakage, then it cannot match R_factor. The checker reports that lower-triangular leakage and the reconstruction residual as diagnostics.

This shape set targets optimizer-style matrix statistics where gradients are viewed as [for_each..., basis_dim, contracted_dim], statistics are formed as G @ G.T, and QR is run on square basis_dim x basis_dim matrices. Batched 512 x 512 is especially important, while 1024, 2048, and 4096 cover larger square factors.

Test and benchmark specs include a cond field. In this task cond is a deterministic input-scaling knob, not an exact requested condition number: dense cases multiply columns by logspace(0, -cond, n), so larger cond creates a wider dynamic range across columns. Some stress cases use their own structure, such as rank-deficient, near-rank-deficient, banded, row-scaled, near-collinear, upper-triangular, or clustered-scale inputs.

The mixed case builds a heterogeneous batch: each matrix is independently assigned a conditioning profile (a well-conditioned dense majority interleaved with the ill-conditioned stress structures above) at a random position in the batch. This mirrors the real optimizer-statistics regime, where the per-layer or per-block factors batched into one call have widely varying conditioning, rather than all sharing one structure. The benchmark set (not just the test set) now includes both mixed batches and fully ill-conditioned homogeneous batches, so conditioning robustness is ranked, not only gated: an implementation cannot inspect a few matrices, decide the whole batch is well-conditioned, and route it to a path that is only valid for well-conditioned inputs, and the runtime cost of the accurate path on hard inputs is part of the score. Each matrix must be factored correctly on its own merits.

Correctness is a hard gate against the original FP32 input and the FP32 torch.geqrf compact-factor contract. Low-bit FP16, FP8, or NVFP4 work is allowed only as an internal implementation strategy: returned factors must still be FP32 and must satisfy the same QR invariants as an FP32 factorization. Residuals are measured in FP64 to reduce checker noise, but the target tolerance is still FP32 accuracy. The numerical property tolerance is purely relative, with no QR atol. The hard gates are the LAPACK-style factor residual, which uses rtol = 20 * n * eps32, and orthogonality, which uses rtol = 100 * n * eps32, each applied to the corresponding matrix L1 norm. Triangularity is reported as lower-triangular leakage in Q.T @ A and is already implied by the factor residual against triu(H).

Among passing submissions, ranking is by runtime using the geometric mean of benchmark cases. We will also celebrate notable submissions beyond the main leaderboard: the fastest, the most elegant, and the strangest working kernels.

### Benchmark Shapes

- {"batch":20,"cond":1,"n":32}
- {"batch":40,"cond":1,"n":176}
- {"batch":40,"cond":1,"n":352}
- {"batch":640,"cond":2,"n":512}
- {"batch":60,"cond":2,"n":1024}
- {"batch":8,"cond":1,"n":2048}
- {"batch":2,"cond":1,"n":4096}
- {"batch":640,"case":"mixed","cond":2,"n":512}
- {"batch":60,"case":"mixed","cond":2,"n":1024}
- {"batch":640,"case":"rankdef","cond":0,"n":512}
- {"batch":640,"case":"clustered","cond":0,"n":512}
- {"batch":60,"case":"nearrank","cond":0,"n":1024}
