# Fixes to the Zeeman / transition-amplitude notebooks (2026-09-02)

This documents two bugs found in `Rb87 Zeeman Shifts.ipynb` and
`transition_matrix_elements.ipynb`, found by re-reading the derivation in
`magnetically_induced_transitions.tex` and comparing the code against it and
against Steck, *Rubidium 87 D Line Data*. Every change in the notebooks is
marked inline with a `# FIX (2026-09-02): ...` comment giving the same
explanation as below, so the comments and this file should stay in sync.

Both notebooks still use `sympy.physics.wigner` for the 3j/6j symbols and
`scipy.linalg.eigh` for diagonalization; nothing about the physical model
(the Hamiltonian terms, the hyperfine constants, or the dipole reduced-matrix
values from Steck's Table 7) was changed. Only the two bugs below were fixed.

## Issue 1: (F, m_F) labels on the diagrams don't match the plotted state

**Symptom:** the energy-level plot in `Rb87 Zeeman Shifts.ipynb` colors each
curve by `F`, and the transition-strength legends in
`transition_matrix_elements.ipynb` label each curve `g(F, m_F) -> e(F', m_F')`
— but these labels can be wrong, especially away from B=0 and for the
M-degenerate sublevels at B=0.

**Root cause:** `get_Zeeman_shift()` builds `H` in the fixed basis order
produced by `create_FM_arrays()` (F ascending, then M ascending within F),
but diagonalizes it with `scipy.linalg.eigh()`, which returns eigenvalues
**sorted by energy**, not in that basis order. The two orders coincide only
by coincidence — e.g. at B=0 several (F, M) states are exactly degenerate
(same F, different M), so which of them lands in which output column is
whatever `eigh()` and (for later B) the code's own
`linear_sum_assignment`-based adiabatic tracking happen to produce, not the
input order. Downstream code in both notebooks assumed eigen-index `i`
*is* basis position `i` (e.g. `color_vec[int(F[i])]` in the Zeeman plot, or
`gnums[ii]` in the transition-strength legends where `ii` is actually an
eigenvector index, not a basis index) — that assumption is what produced
wrong labels.

Note that this only ever affected *labels*: the eigenvalues/eigenvectors
themselves, and every transition-probability number computed from them, were
already correct, because the code always looped over *all* eigen-indices
exhaustively and contracted eigenvectors (which are stored in the fixed
basis order regardless of energy sorting) with the dipole matrix in that same
basis order.

**Fix:** since `H` is *exactly diagonal* in the (F, M) basis at B=0 (the
Zeeman terms are all proportional to B), the eigenvector returned for
eigen-index `i` at B=0 is (up to sign) exactly one basis vector. We recover
that basis index via `argmax(|eigenvector components|)` and use it to look
up the true (F, M) label:

```python
def get_state_labels(evec_arr, F, M):
    basis_idx = np.argmax(np.abs(evec_arr[0]), axis=0)
    return F[basis_idx], M[basis_idx]
```

This label is then valid at every other B for free, because
`get_Zeeman_shift()` already tracks each eigen-index's identity across B via
the `linear_sum_assignment` overlap matching — that machinery was already
correct, it just was never used to fix the *initial* label.

Applied in:
- `Rb87 Zeeman Shifts.ipynb`: the plotting cell now colors by
  `F_labels[i]` (from `get_state_labels`) instead of `F[i]`.
- `transition_matrix_elements.ipynb`: a new cell computes
  `g_eig_labels`/`e_eig_labels` (per-eigen-index labels) separately from
  `gnums`/`enums` (which stay in basis order, since they're also used to
  build `no_B_matrix_element_arr` in basis order for the matrix
  contraction — renaming those would have been a correctness bug in the
  other direction). The legend-building cells were switched to use
  `g_eig_labels[ii]` / `e_eig_labels[jj]`.

A related, smaller bug was fixed in `Rb87 Zeeman Shifts.ipynb`'s
`get_Zeeman_shift()`: `evec_arr.append(eigenvectors[col_indices])` indexed
*rows* of the eigenvector matrix instead of columns (each column is one
eigenvector) — it should be `eigenvectors[:, col_indices]`, matching the
already-correct line just below it (`evecs_old = eigenvectors[:,
col_indices]`) and matching how `transition_matrix_elements.ipynb` already
did it. This didn't affect that notebook's own plot (which only uses
`eig_arr`), but would have silently corrupted `evec_arr` for any downstream
use.

## Issue 2: missing/incorrect factors in the transition dipole amplitude

**Symptom:** `no_B_matrix_element()` in `transition_matrix_elements.ipynb`
did not reproduce the amplitude derived in
`magnetically_induced_transitions.tex` / Steck.

**Root cause:** two compounding problems in the same function.

1. `transition_dipole_matrix_element()` returns Steck's Table 7 values,
   4.227 and 2.992 (in units of `e a0`), for the D2 (`J'=3/2`) and D1
   (`J'=1/2`) lines respectively. These are the ***J*-reduced** dipole matrix
   elements `<J||er||J'>` — the ratio 4.227/2.992 = 1.4128 ≈ √2 is exactly
   the L,S→J recoupling factor between `J'=3/2` and `J'=1/2` for the same
   `L`, `L'`, which only appears if the L,S-decoupling step has already been
   folded into the tabulated number (an `L`-reduced element `<L||er||L'>`
   would be identical for the D1 and D2 lines, since both share the same
   `L=0, L'=1`). Getting from `<J||er||J'>` to `<F,M|er_q|F',M'>` therefore
   needs only **one** more spectator-theorem step (decoupling `F = I + J`
   with `I` as the spectator — see the intermediate result in
   `magnetically_induced_transitions.tex`, "Step 3", just before the second
   spectator-theorem application).

   The code instead applied a **second**, redundant spectator step
   (`second_spectator`, decoupling `J = L + S` again) on top of the already
   J-reduced tabulated value. This double-counts the L-S recoupling.
   Numerically this multiplies every amplitude by the same
   `sqrt((2J+1)(2J'+1)) * {L' J' S; J L 1}` factor regardless of `F, F'`, so
   it does not change the *relative* branching between different `F'` (a
   sum-rule check, `sum_F' (2F'+1) {J' F' I; F J 1}^2 = 1/(2J+1)`, holds for
   the single-spectator-step formula independent of F, and the two-step
   formula just rescales every term by the same wrong constant) — but it
   *does* corrupt the absolute magnitude of the dipole moment, which matters
   anywhere the code (or the memory/storage analysis in the .tex) uses this
   value as a physical Rabi frequency or decay rate rather than a relative
   strength.

2. A `# Test` block further down in the same function then overwrote
   `parity`, `mere_integral`, and `second_spectator`:
   ```python
   parity = (-1)**(1+F + Fp + J + Ip - M)
   mere_integral = 1
   second_spectator = (2*J+1)**0.5
   ```
   This is what actually executed (Python re-binds the names), and it is
   worse than problem 1: `mere_integral = 1` discards the physical dipole
   moment entirely, so the returned amplitude was not in any physical unit,
   and the replacement `second_spectator = sqrt(2J+1)` does not correspond
   to any term in the derivation.

**Fix:** removed the `# Test` override block and the redundant
`second_spectator` step, and corrected `parity` to the phase that belongs
with the single remaining spectator step
(`(-1)^{F - M + F' + J + 1 + I}`, from
`magnetically_induced_transitions.tex`, Step 3, before the L,S-decoupling is
applied a second time):

```python
def no_B_matrix_element(L,J,F,M,q,Lp,Jp,Fp,Mp):
    I = 3/2; Ip = I; S = 1/2; Sp = S
    parity = (-1)**(F - M + Fp + J + 1 + I)
    mere_integral = transition_dipole_matrix_element(L,J,Lp,Jp)
    wigner_eckart = wigner_3j(F,1,Fp,-M,q,Mp)
    first_spectator = ((2*F+1)*(2*Fp+1))**(1/2)*wigner_6j(Jp,Fp,I,F,J,1)
    return parity*mere_integral*wigner_eckart*first_spectator
```

Sanity check (not in the notebook, done separately): summing
`|<F,M|er_0|F',M>|^2` over `M', q` with this corrected formula reproduces
the well-known sum rule that the total dipole strength leaving a given `F`
is independent of `F` (0.5 in units of `<J||er||J'>^2`, for both F=1 and F=2
of the D2 line's ground state) — the two-spectator-step version instead gave
0.667 for both (a constant, F-independent rescaling, confirming it's a
normalization bug rather than a branching-ratio bug).

## What to re-check

Both notebooks were re-executed end-to-end after these changes to confirm
they still run without errors. Anyone using cached/exported numbers from
before this fix (e.g. transition strengths, or curve labels copied out of
these notebooks) should treat them as suspect and regenerate them.
