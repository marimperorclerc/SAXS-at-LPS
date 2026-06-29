r"""
This model describes a single pseudo-Voigt Bragg/form-factor peak.

It is a stripped-down, lattice-free model containing only one peak whose position is a
free parameter. It is intended to be summed with other models, so no Porod regime is
included here (the Porod q**-4 contribution is expected to come from the model it is
summed with). A typical use is to account for a small peak at low q (e.g. arising from
a form factor) that is not otherwise modelled.

Definition
----------

The peak is modeled using a pseudo-Voigt peak function. The peak position ``peak_pos``
is an independent fit parameter and is not tied to any unit cell.

Validation
----------


References
----------


1. L A Feigin, D I Svergun, G W Taylor 
   Structure Analysis by Small-Angle X-ray and Neutron Scattering 
   Springer (1987)

2. Aaron L. Stancik, Eric B. Brauns
   A simple asymmetric lineshape for fitting infrared absorption spectra
   Vibrational Spectroscopy 47 (2008) 66-69

   
Authorship and Verification
----------------------------

* **Author:**  Steve King **Date:** 24 June 2020 

* **Authors:** Marianne Imperor-Clerc (marianne.imperor@cnrs.fr)
               Anirban Mandal (mandalanirban2023@gmail.com)

* **Last Modified by:** MIC **Date:** 16 June 2026

* **Last Reviewed by:** **Date:**

"""

import numpy as np
from numpy import inf, errstate

name = "peak_voigt"
title = "Single pseudo-Voigt peak"
description = """\
          I(q) = scale*peak + background
"""

category = "shape-independent"

parameters = [["w_f", "", 0.8, [0, 1], "", "lorentzian/gaussian weighting factor"],
              ["peak_pos", "1/Ang", 0.05, [0, inf], "", "Position of the peak"],
              ["hwhm_peak", "1/Ang", 0.01, [0, 1], "", "HWHM of the peak"]]


def Ipeak(q, wf, q0, hwhm):
    """
    When $w_f$ = 1 a Lorentzian peak is returned, and when $w_f$ = 0 a
    Gaussian peak is returned.

    The peak is taken to be centered at $q_0$ with a HWHM (half-width
    half-maximum) for the Lorentzian and $sigma$=HWHM/1.17741, where $sigma$ is the standard deviation
    of the Gaussian. In other words, the widths of the Lorentzian and the
    Gaussian have been coupled for convenience of parameterisation.
    
    """
    cste=np.sqrt(2*np.log(2))
    #cste=1.17741
    sigma=hwhm/cste
    intensity = (wf*(1/(1+((q-q0)**2.0/hwhm**2.0))))+((1.0-wf)*np.exp((-0.5*(q-q0)**2.0)/(sigma**2.0)))
    return intensity


def Iq(q, w_f, peak_pos, hwhm_peak):
    """
    w_f: weighting coefficient in the pseudo Voigt peak function 
    $w_f=1$ for Lorentzian and $w_f=0$ for Gaussian peak.
    peak_pos: position of the peak
    hwhm_peak: HWHM of the peak
    """
    
    with errstate(divide='ignore'):
        L = Ipeak(q, w_f, peak_pos, hwhm_peak)

        return L

Iq.vectorized = True  # Iq accepts an array of q values

tests = [
    [{"scale": 1.0,
      "background": 0.0,
      "peak_pos": 0.05,
      "hwhm_peak": 0.01
      },
      0.05, 1.0],
]
