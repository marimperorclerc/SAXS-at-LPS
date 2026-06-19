r"""
This model adds a q**-4 Porod regime and hexagonal Bragg peaks with the first five peaks present.

It describes objects assembled together with internal 2D hexagonal order. 
The Porod term accounts for the contribution at low q due to the surface of the assemblies.

Definition
----------

Bragg peaks are modeled using pseudo-Voigt peak function.

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
* **Author:**  Steve King **Date:** 24 June 2020 

* **Last Modified by:** MIC **Date:** 16 June 2026

* **Last Reviewed by:** **Date:**

"""

import numpy as np
from numpy import inf, errstate

name = "porod_peak_voigt_hexagonal"
title = "Porod function with added hexagonal pseudo-Voigt Bragg peaks"
description = """\
          I(q) = scale(scale_Porod/q^4+sum_of_hexagonal_peaks) + background
"""

category = "shape-independent"

parameters = [["scale_Porod", "", 0.05, [0, inf], "", "Scale factor for Porod"],
              ["a_cell", "Ang", 40, [20, 200], "", "hexagonal cell parameter"],
              ["w_f", "", 0.8, [0, 1], "", "lorentzian/gaussian weighting factor"],
              ["hwhm_q10", "1/Ang", 0.01, [0, 1], "", "HWHM of q10 peak"],
              ["hwhm_q11", "1/Ang", 0.01, [0, 1], "", "HWHM of q11 peak"],
              ["hwhm_q20", "1/Ang", 0.01, [0, 1], "", "HWHM of q20 peak"],
              ["hwhm_q21", "1/Ang", 0.01, [0, 1], "", "HWHM of q21 peak"],
              ["hwhm_q30", "1/Ang", 0.01, [0, 1], "", "HWHM of q30 peak"],
              ["scale_q10", "", 1, [0,inf], "", "Scale factor for q10 peak"],
              ["scale_q11", "", 1, [0,inf], "", "Scale factor for q11 peak"],
              ["scale_q20", "", 1, [0,inf], "", "Scale factor for q20 peak"],
              ["scale_q21", "", 0, [0,inf], "", "Scale factor for q21 peak"],
              ["scale_q30", "", 0, [0,inf], "", "Scale factor for q30 peak"]]



def Ipeak(q, wf, q0, hwhm):
    """
    When $w_f$ = 1 a Lorentzian peak is returned, and when $w_f$ = 0 a
    Gaussian peak is returned.

    The peak is taken to be centered at $q_0$ with a HWHM (half-width
    half-maximum) for the Lorentzian and $sigma$=HWHM/1.17741, where $sigma$ is the standard deviation
    of the Gaussian. In other words, the widths of the Lorentzian and the
    Gaussian have been coupled for convenience of parameterisation.
    
    1.17741=np.sqrt(2*np.ln(2))
    """
    #cste=np.sqrt(2*np.ln(2))
    cste=1.17741
    sigma=hwhm/cste
    intensity = (wf*(1/(1+((q-q0)**2.0/hwhm**2.0))))+((1.0-wf)*np.exp((-0.5*(q-q0)**2.0)/(sigma**2.0)))
    return intensity


def Iq(q,scale_Porod,a_cell, w_f, hwhm_q10, hwhm_q11, hwhm_q20, hwhm_q21, hwhm_q30, scale_q10, scale_q11, scale_q20, scale_q21, scale_q30):
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
    """
    
    with errstate(divide='ignore'):
        q10=4*np.pi/(np.sqrt(3)*a_cell)
        q11=np.sqrt(3)*q10
        q20=2*q10
        q21=np.sqrt(7)*q10
        q30=3*q10
        porod = (scale_Porod/q)**4
        L10 = Ipeak(q,w_f,q10,hwhm_q10)
        L11 = Ipeak(q,w_f,q11,hwhm_q11)
        L20 = Ipeak(q,w_f,q20,hwhm_q20)
        L21 = Ipeak(q,w_f,q21,hwhm_q21)
        L30 = Ipeak(q,w_f,q30,hwhm_q30)

        return porod+scale_q10*L10+scale_q11*L11+scale_q20*L20+scale_q21*L21+scale_q30*L30

Iq.vectorized = True  # Iq accepts an array of q values

tests = [
    [{"scale": 0.00001, 
      "background":0.01,
      "scale_Porod": 1.,
      "scale_q10": 0.,
      "scale_q11": 0.,
      "scale_q20": 0.,
      "scale_q21": 0.,
      "scale_q30": 0.,
      }, 
      0.04, 3.916250],
]
