r"""
This model adds a q**-4 Porod regime and hexagonal Bragg peaks with the
first ten peaks present.

It describes objects assembled together with internal 2D hexagonal order.
The Porod term accounts for the contribution at low q due to the surface
of the assemblies.

Definition
----------

Bragg peaks are modeled using pseudo-Voigt peak function.

The peak positions of a 2D hexagonal lattice follow
q_hk = q10 * sqrt(h**2 + h*k + k**2), so the first ten allowed reflections
(h,k) correspond to h**2+h*k+k**2 = 1, 3, 4, 7, 9, 12, 13, 16, 19, 21,
i.e. the (10), (11), (20), (21), (30), (22), (31), (40), (32) and (41) peaks.

Validation
----------


References
----------

1. G Porod. *Kolloid Zeit* 124 (1951) 83

2. L A Feigin, D I Svergun, G W Taylor
   Structure Analysis by Small-Angle X-ray and Neutron Scattering
   Springer (1987)

3. Aaron L. Stancik, Eric B. Brauns
   A simple asymmetric lineshape for fitting infrared absorption spectra
   Vibrational Spectroscopy 47 (2008) 66-69


Authorship and Verification
----------------------------

* **Authors:** Jules Marcone (julesmarcone@gmail.com) **Date:** 30 May 2023
               Marianne Imperor-Clerc (marianne.imperor@cnrs.fr)
               Anirban Mandal (mandalanirban2023@gmail.com)
* **Author:**  Steve King **Date:** 24 June 2020

* **Last Modified by:** MIC **Date:** 16 June 2026

* **Last Reviewed by:** **Date:**

"""

import numpy as np
from numpy import errstate, inf

name = "porod_peak_voigt_hexagonal_10peaks"
title = "Porod function with added hexagonal pseudo-Voigt Bragg peaks (10 peaks)"
description = """\
          I(q) = scale(scale_Porod/q^4+sum_of_hexagonal_peaks) + background
"""

category = "shape-independent"

# The ten first allowed reflections of a 2D hexagonal lattice, as
# (label, h**2 + h*k + k**2).  The peak positions are
# q_hk = q10 * sqrt(h**2 + h*k + k**2).
PEAKS = (
    ("q10", 1),
    ("q11", 3),
    ("q20", 4),
    ("q21", 7),
    ("q30", 9),
    ("q22", 12),
    ("q31", 13),
    ("q40", 16),
    ("q32", 19),
    ("q41", 21),
)

# sqrt(2*ln(2)) = 1.17741, the HWHM-to-sigma conversion for a Gaussian.
SIGMA_PER_HWHM = 1.0 / np.sqrt(2.0 * np.log(2.0))

parameters = [
    ["scale_Porod", "", 0.05, [0, inf], "", "Scale factor for Porod"],
    ["a_cell", "Ang", 900, [20, 10000], "", "hexagonal cell parameter"],
    ["w_f", "", 0.8, [0, 1], "", "lorentzian/gaussian weighting factor"],
    ["hwhm_q10", "1/Ang", 1, [0, 1], "", "HWHM of q10 peak"],
    ["hwhm_q11", "1/Ang", 1, [0, 1], "", "HWHM of q11 peak"],
    ["hwhm_q20", "1/Ang", 1, [0, 1], "", "HWHM of q20 peak"],
    ["hwhm_q21", "1/Ang", 1, [0, 1], "", "HWHM of q21 peak"],
    ["hwhm_q30", "1/Ang", 1, [0, 1], "", "HWHM of q30 peak"],
    ["hwhm_q22", "1/Ang", 1, [0, 1], "", "HWHM of q22 peak"],
    ["hwhm_q31", "1/Ang", 1, [0, 1], "", "HWHM of q31 peak"],
    ["hwhm_q40", "1/Ang", 1, [0, 1], "", "HWHM of q40 peak"],
    ["hwhm_q32", "1/Ang", 1, [0, 1], "", "HWHM of q32 peak"],
    ["hwhm_q41", "1/Ang", 1, [0, 1], "", "HWHM of q41 peak"],
    ["scale_q10", "", 1, [0, inf], "", "Scale factor for q10 peak"],
    ["scale_q11", "", 1, [0, inf], "", "Scale factor for q11 peak"],
    ["scale_q20", "", 1, [0, inf], "", "Scale factor for q20 peak"],
    ["scale_q21", "", 0, [0, inf], "", "Scale factor for q21 peak"],
    ["scale_q30", "", 0, [0, inf], "", "Scale factor for q30 peak"],
    ["scale_q22", "", 0, [0, inf], "", "Scale factor for q22 peak"],
    ["scale_q31", "", 0, [0, inf], "", "Scale factor for q31 peak"],
    ["scale_q40", "", 0, [0, inf], "", "Scale factor for q40 peak"],
    ["scale_q32", "", 0, [0, inf], "", "Scale factor for q32 peak"],
    ["scale_q41", "", 0, [0, inf], "", "Scale factor for q41 peak"],
]

# Guard against the table and the peak list drifting apart when a peak is
# added or removed.
assert len(parameters) == 3 + 2 * len(PEAKS)


def Ipeak(q, wf, q0, hwhm):
    """
    When $w_f$ = 1 a Lorentzian peak is returned, and when $w_f$ = 0 a
    Gaussian peak is returned.

    The peak is taken to be centered at $q_0$ with a HWHM (half-width
    half-maximum) for the Lorentzian and $sigma$=HWHM/1.17741, where $sigma$
    is the standard deviation of the Gaussian. In other words, the widths of
    the Lorentzian and the Gaussian have been coupled for convenience of
    parameterisation.

    A peak of zero width carries no intensity, so zero is returned rather
    than a division by zero.
    """
    if hwhm <= 0.0:
        return np.zeros_like(q, dtype=float)
    reduced = (q - q0) / hwhm
    lorentzian = 1.0 / (1.0 + reduced**2)
    gaussian = np.exp(-0.5 * (reduced / SIGMA_PER_HWHM) ** 2)
    return wf * lorentzian + (1.0 - wf) * gaussian


def Iq(
    q,
    scale_Porod,
    a_cell,
    w_f,
    hwhm_q10,
    hwhm_q11,
    hwhm_q20,
    hwhm_q21,
    hwhm_q30,
    hwhm_q22,
    hwhm_q31,
    hwhm_q40,
    hwhm_q32,
    hwhm_q41,
    scale_q10,
    scale_q11,
    scale_q20,
    scale_q21,
    scale_q30,
    scale_q22,
    scale_q31,
    scale_q40,
    scale_q32,
    scale_q41,
):
    """
    scale_Porod: scale coefficent for Porod term
    a_cell: 2D hexagonal cell parameter
    w_f: weighting coefficient in the pseudo Voigt peak function
    $w_f=1$ for Lorentzian and $w_f=0$ for Gaussian peak.
    hwhm_q10: HWHM of 10 hexagonal peak
    scale_q10: Scale factor for q10 peak
    hwhm_q11: HWHM of 11 hexagonal peak
    scale_q11: Scale factor for q11 peak
    hwhm_q20: HWHM of 20 hexagonal peak
    scale_q20: Scale factor for q20 peak
    hwhm_q21: HWHM of 21 hexagonal peak
    scale_q21: Scale factor for q21 peak
    hwhm_q30: HWHM of 30 hexagonal peak
    scale_q30: Scale factor for q30 peak
    hwhm_q22: HWHM of 22 hexagonal peak
    scale_q22: Scale factor for q22 peak
    hwhm_q31: HWHM of 31 hexagonal peak
    scale_q31: Scale factor for q31 peak
    hwhm_q40: HWHM of 40 hexagonal peak
    scale_q40: Scale factor for q40 peak
    hwhm_q32: HWHM of 32 hexagonal peak
    scale_q32: Scale factor for q32 peak
    hwhm_q41: HWHM of 41 hexagonal peak
    scale_q41: Scale factor for q41 peak
    """
    hwhm_values = (
        hwhm_q10,
        hwhm_q11,
        hwhm_q20,
        hwhm_q21,
        hwhm_q30,
        hwhm_q22,
        hwhm_q31,
        hwhm_q40,
        hwhm_q32,
        hwhm_q41,
    )
    scale_values = (
        scale_q10,
        scale_q11,
        scale_q20,
        scale_q21,
        scale_q30,
        scale_q22,
        scale_q31,
        scale_q40,
        scale_q32,
        scale_q41,
    )

    q = np.asarray(q, dtype=float)
    q10 = 4 * np.pi / (np.sqrt(3) * a_cell)

    with errstate(divide="ignore"):
        intensity = (scale_Porod / q) ** 4

    for index, (_, hk_index) in enumerate(PEAKS):
        scale_peak = scale_values[index]
        if scale_peak == 0.0:
            continue
        q_hk = q10 * np.sqrt(hk_index)
        intensity = intensity + scale_peak * Ipeak(q, w_f, q_hk, hwhm_values[index])

    return intensity


Iq.vectorized = True  # Iq accepts an array of q values

tests = [
    [
        {
            "scale": 0.00001,
            "background": 0.01,
            "scale_Porod": 1.0,
            "scale_q10": 0.0,
            "scale_q11": 0.0,
            "scale_q20": 0.0,
            "scale_q21": 0.0,
            "scale_q30": 0.0,
            "scale_q22": 0.0,
            "scale_q31": 0.0,
            "scale_q40": 0.0,
            "scale_q32": 0.0,
            "scale_q41": 0.0,
        },
        0.04,
        3.916250,
    ],
]
