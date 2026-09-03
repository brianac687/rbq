"""
Rb-87 D2 line (5S_1/2 -> 5P_3/2) hyperfine transition frequency calculator.

Given a ground-state hyperfine level F_g (5S_1/2, F = 1 or 2) and an
excited-state hyperfine level F_e (5P_3/2, F = 0, 1, 2, or 3), returns the
frequency of the F_g -> F_e transition.

Method
------
The D2 "carrier" frequency (384 230 484.468 5 MHz) is defined, by
convention, as the transition frequency between the *centroids* of the
ground- and excited-state hyperfine manifolds.

Rather than computing those centroids from degeneracy weights (2F+1), this
version uses the centroid-distance values printed directly on the diagram
for each F level (the "crossing" intervals shown next to dashed reference
lines on the 5P_3/2 side, and the two intervals shown on the 5S_1/2 side).
Each F level's tabulated number IS its signed distance from the centroid
of its manifold, read straight off the chart:

    f(F_g, F_e) = f_D2 + dist_e(F_e) - dist_g(F_g)

Reference: Steck, "Rubidium 87 D Line Data" (Los Alamos), and the values
shown in the uploaded level diagram.
"""

from dataclasses import dataclass

# ---------------------------------------------------------------------
# Input data taken directly from the diagram (all in MHz)
# ---------------------------------------------------------------------

# D2 line center frequency (5S1/2 centroid -> 5P3/2 centroid), in MHz
F_D2_CENTER_MHz = 384_230_484.4685  # 384.230 484 468 5 THz

# 5S_1/2 ground state: signed distance of each F level from the manifold
# centroid, read directly from the diagram (positive = above centroid).
#   F=2 is 2.563 005 979 089 11 GHz above centroid
#   F=1 is 4.271 676 631 815 19 GHz below centroid
GROUND_CENTROID_DIST_MHz = {
    2: 2_563.005_979_089_109,
    1: -4_271.676_631_815_181,
}

# 5P_3/2 excited state: signed distance of each F level from the manifold
# centroid, read directly from the diagram (positive = above centroid).
#   F=3 is 193.7408 MHz above centroid
#   F=2 is  72.9113 MHz below centroid
#   F=1 is 229.8518 MHz below centroid
#   F=0 is 302.0738 MHz below centroid
EXC_CENTROID_DIST_MHz = {
    3: 193.7407,
    2: -72.9112,
    1: -229.8518,
    0: -302.0738,
}


def hyperfine_transition_frequency_MHz(F_g: int, F_e: int) -> float:
    """
    Return the 5S1/2(F_g) -> 5P3/2(F_e) transition frequency in MHz.

    Parameters
    ----------
    F_g : int
        Ground state total angular momentum quantum number (1 or 2).
    F_e : int
        Excited state total angular momentum quantum number (0, 1, 2, or 3).
    """
    if F_g not in GROUND_CENTROID_DIST_MHz:
        raise ValueError(f"F_g must be 1 or 2 for 5S_1/2, got {F_g}")
    if F_e not in EXC_CENTROID_DIST_MHz:
        raise ValueError(f"F_e must be 0, 1, 2, or 3 for 5P_3/2, got {F_e}")

    dist_ground = GROUND_CENTROID_DIST_MHz[F_g]
    dist_excited = EXC_CENTROID_DIST_MHz[F_e]

    return F_D2_CENTER_MHz + dist_excited - dist_ground


@dataclass
class TransitionResult:
    F_g: int
    F_e: int
    frequency_MHz: float
    frequency_GHz: float
    frequency_THz: float
    wavelength_nm: float
    frequency_diff_MHz: float  # difference from D2 center


def get_transition(F_g: int, F_e: int) -> TransitionResult:
    """Convenience wrapper returning frequency in multiple units + wavelength."""
    f_MHz = hyperfine_transition_frequency_MHz(F_g, F_e)
    c = 299_792_458.0  # m/s
    wavelength_nm = c / (f_MHz * 1e6) * 1e9
    return TransitionResult(
        F_g=F_g,
        F_e=F_e,
        frequency_MHz=f_MHz,
        frequency_GHz=f_MHz / 1e3,
        frequency_THz=f_MHz / 1e6,
        wavelength_nm=wavelength_nm,
        frequency_diff_MHz=f_MHz - F_D2_CENTER_MHz,
    )

if __name__ == "__main__":
    print(f"{'F_g':>3} -> {'F_e':>3}   {'Frequency (MHz)':>18}   {'Wavelength (nm)':>15}")
    print("-" * 60)
    for F_g in (1, 2):
        for F_e in (0, 1, 2, 3):
            r = get_transition(F_g, F_e)
            print(f"{F_g:>3} -> {F_e:>3}   {r.frequency_MHz:18.4f}   {r.wavelength_nm:15.6f}")

    # Example: the common D2 cooling transition F=2 -> F=3
    r = get_transition(2, 3)
    print("\nExample (cooling transition, F=2 -> F=3):")
    print(f"  f = {r.frequency_MHz:.4f} MHz = {r.frequency_GHz:.6f} GHz")
    print(f"  lambda = {r.wavelength_nm:.6f} nm")
    print(f"  f = {r.frequency_MHz - F_D2_CENTER_MHz:.6f} MHz")