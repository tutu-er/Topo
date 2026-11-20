# Identifying Connections Near the Root

Root-near internal supports, the common upstream support, and singleton terminal supports have different observation and selection properties. The [locked study](PROTOCOL.md) evaluates them separately. The [original derivation and audit plan](https://github.com/tutu-er/Topo/blob/e66aae90cf49093483ac3dc8ece2e8d411296a99/rnj_wzzt_dr/experiments/wzzt_long_grid_20261117/ROOT_IDENTIFIABILITY.md) gives the complete rationale.

## Root-anchored tree distances

On an exact additive tree, positive weights $a,b$ give the shared-path matrix $G=aR+bX$. The distance between measured terminals $i,j$ is $d_{ij}=G_{ii}+G_{jj}-2G_{ij}$. The distance from the root to terminal $i$ is $d_{0i}=G_{ii}$. These distances recover each shared path by $G_{ij}=(d_{0i}+d_{0j}-d_{ij})/2$. The common upstream term cancels from terminal pairs and remains in the root distances. The RNJ implementation uses root depth alongside terminal information.

The observable tree combines unmeasured series segments that have the same downstream set. It also omits branches that feed no measured terminals. Report recovery against this observable structure. Under the linearized model and the declared per-unit base $Z_{\rm base}$, the response coefficients for an observable branch are $r_C=2R_{C,\Omega}/Z_{\rm base}$ and $x_C=2X_{C,\Omega}/Z_{\rm base}$.

## Power variation and meter error

A support $C$, with indicator vector $z_C$, produces an active-power feature column formed from $(P_t^\top z_C)z_C$ at every time $t$. Its squared norm is $|C|\sum_t(P_t^\top z_C)^2$. Larger supports can produce a stronger voltage signal because more terminals experience the drop. Ancestor and descendant supports can also have highly correlated profiles. To assess separability in a known network, project that branch's P and Q columns off the other branch columns. A small eigenvalue of their residual two-column Gram matrix indicates a weakly observable resistance/reactance combination.

Root meter error contributes the same first-order voltage-drop error to all terminals at one slot. For common root error variance $\sigma_0^2$ and independent terminal variance $\sigma_v^2$, the information about a shared terminal shift is $n/(\sigma_v^2+n\sigma_0^2)$. It saturates as the number of terminals grows when root error is present. The root-meter control changes measurement error while keeping physical root voltage and loads fixed.

## Comparing recovery

Report exact internal recovery and exact full-support recovery separately. Stratify true supports by root depth and downstream size, and report false supports by size or predicted depth. Compare the R/X matrix error separately from resistance/reactance errors on matched branches. A native solve that improves training loss may still yield an incorrect support. The analysis tracks whether correct supports first appeared in an RNJ family, a pool trial, an unknown-support solve, or the E reconstruction, and records any effect of output contraction.

The runs compare finite budgets on a shared set of inputs and networks. They provide evidence about these data conditions and do not establish a universal ranking of near-root recovery.
