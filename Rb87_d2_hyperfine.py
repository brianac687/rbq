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
F_D1_CENTER_MHz = 377_107_463.38011  # 377.107 463 380 THz

# Signed distance of each F level from its manifold centroid, read directly
# off the diagram (positive = above centroid), keyed by state label.
#
#   5S_1/2: F=2 is 2.563 005 979 089 11 GHz above centroid
#           F=1 is 4.271 676 631 815 19 GHz below centroid
#   5P_1/2: F=2 is 305.4447 MHz above centroid
#           F=1 is 509.0679 MHz below centroid
#   5P_3/2: F=3 is 193.7408 MHz above centroid
#           F=2 is  72.9113 MHz below centroid
#           F=1 is 229.8518 MHz below centroid
#           F=0 is 302.0738 MHz below centroid
LEVEL_CENTROID_DIST_MHz = {
    "5S1/2": {
        2: 2_563.005_979_089_109_34,
        1: -4_271.676_631_815_181_56,
    },
    "5P1/2": {
        2: 305.4447,
        1: -509.0679,
    },
    "5P3/2": {
        3: 193.740746,
        2: -72.911232,
        1: -229.851856,
        0: -302.073888,
    },
}

# Line center frequency (ground-state centroid -> excited-state centroid),
# in MHz, keyed by (ground_label, excited_label).
LINE_CENTER_MHz = {
    ("5S1/2", "5P3/2"): F_D2_CENTER_MHz,
    ("5S1/2", "5P1/2"): F_D1_CENTER_MHz,
}


def hyperfine_transition_frequency_MHz(
    ground_label: str, F_g: int, excited_label: str, F_e: int
) -> float:
    """
    Return the <ground_label>(F_g) -> <excited_label>(F_e) transition
    frequency in MHz.

    Parameters
    ----------
    ground_label : str
        Ground state label, e.g. "5S1/2".
    F_g : int
        Ground state total angular momentum quantum number.
    excited_label : str
        Excited state label, e.g. "5P3/2" (or "5P1/2" once its centroid
        distances and line center are added above).
    F_e : int
        Excited state total angular momentum quantum number.
    """
    if ground_label not in LEVEL_CENTROID_DIST_MHz:
        raise ValueError(f"Unknown ground state label {ground_label!r}")
    if excited_label not in LEVEL_CENTROID_DIST_MHz:
        raise ValueError(f"Unknown excited state label {excited_label!r}")

    ground_dists = LEVEL_CENTROID_DIST_MHz[ground_label]
    excited_dists = LEVEL_CENTROID_DIST_MHz[excited_label]

    if F_g not in ground_dists:
        raise ValueError(
            f"F_g must be one of {sorted(ground_dists)} for {ground_label}, got {F_g}"
        )
    if F_e not in excited_dists:
        raise ValueError(
            f"F_e must be one of {sorted(excited_dists)} for {excited_label}, got {F_e}"
        )

    line_key = (ground_label, excited_label)
    if line_key not in LINE_CENTER_MHz:
        raise ValueError(f"No line center defined for {ground_label} -> {excited_label}")

    line_center = LINE_CENTER_MHz[line_key]
    dist_ground = ground_dists[F_g]
    dist_excited = excited_dists[F_e]

    return line_center + dist_excited - dist_ground


@dataclass
class TransitionResult:
    ground_label: str
    F_g: int
    excited_label: str
    F_e: int
    frequency_MHz: float
    frequency_GHz: float
    frequency_THz: float
    wavelength_nm: float
    frequency_diff_MHz: float  # difference from the line center


def get_transition(
    ground_label: str, F_g: int, excited_label: str, F_e: int
) -> TransitionResult:
    """Convenience wrapper returning frequency in multiple units + wavelength."""
    f_MHz = hyperfine_transition_frequency_MHz(ground_label, F_g, excited_label, F_e)
    line_center = LINE_CENTER_MHz[(ground_label, excited_label)]
    c = 299_792_458.0  # m/s
    wavelength_nm = c / (f_MHz * 1e6) * 1e9
    return TransitionResult(
        ground_label=ground_label,
        F_g=F_g,
        excited_label=excited_label,
        F_e=F_e,
        frequency_MHz=f_MHz,
        frequency_GHz=f_MHz / 1e3,
        frequency_THz=f_MHz / 1e6,
        wavelength_nm=wavelength_nm,
        frequency_diff_MHz=f_MHz - line_center,
    )

if __name__ == "__main__":
    print(f"{'F_g':>3} -> {'F_e':>3}   {'Frequency (MHz)':>18}   {'Wavelength (nm)':>15}")
    print("-" * 60)
    for F_g in (1, 2):
        for F_e in (0, 1, 2, 3):
            r = get_transition("5S1/2", F_g, "5P3/2", F_e)
            print(f"{F_g:>3} -> {F_e:>3}   {r.frequency_MHz:18.4f}   {r.wavelength_nm:15.6f}")

    # Example: the common D2 cooling transition F=2 -> F=3
    r = get_transition("5S1/2", 2, "5P3/2", 3)
    print("\nExample (cooling transition, F=2 -> F=3):")
    print(f"  f = {r.frequency_MHz:.4f} MHz = {r.frequency_GHz:.6f} GHz")
    print(f"  lambda = {r.wavelength_nm:.6f} nm")
    print(f"  f = {r.frequency_diff_MHz:.6f} MHz")