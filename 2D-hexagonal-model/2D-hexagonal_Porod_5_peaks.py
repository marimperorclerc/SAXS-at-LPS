r"""
This model adds a q**-4 Porod regime to an hexagonal layer of objects, with the first five peaks present

References
----------

#. G Porod. *Kolloid Zeit*. 124 (1951) 83
#. L A Feigin, D I Svergun, G W Taylor. *Structure Analysis by Small-Angle
   X-ray and Neutron Scattering*. Springer. (1987)

Authorship and Verification
----------------------------

* **Authors: Jules Marcone, Marianne Imperor-Clerc, 30May2023**
* **Last Modified by: Paul Butler, 28Mar2020**
* **Last Reviewed by:**
"""

import numpy as np
from numpy import inf, errstate

name = "2D-hexagonal_Porod_5_peaks"
title = "Porod function with added hexagonal layer"
description = """\
          I(q) = scale(scale_Porod/q^4+sum_of_hexagonal_peaks) + background
"""

category = "custom"

parameters = [["scale_Porod", "", 0.05, [0, inf], "", "Scale factor for Porod"],
              ["a_cell", "Ang", 40, [20, 200], "", "hexagonal cell parameter"],
              ["hwhm_q10", "1/Ang", 0.01, [0, 1], "", "HWHM of q10 peak"],
              ["hwhm_q11", "1/Ang", 0.01, [0, 1], "", "HWHM of q11 peak"],
              ["hwhm_q20", "1/Ang", 0.01, [0, 1], "", "HWHM of q20 peak"],
              ["hwhm_q21", "1/Ang", 0.01, [0, 1], "", "HWHM of q21 peak"],
              ["hwhm_q30", "1/Ang", 0.01, [0, 1], "", "HWHM of q30 peak"],
              ["scale_q10", "", 1, [0, inf], "", "Scale factor for q10 peak"],
              ["scale_q11", "", 1, [0, inf], "", "Scale factor for q11 peak"],
              ["scale_q20", "", 1, [0,inf], "", "Scale factor for q20 peak"],
              ["scale_q21", "", 0, [0,inf], "", "Scale factor for q21 peak"],
              ["scale_q30", "", 0, [0,inf], "", "Scale factor for q30 peak"]]

def Iq(q,scale_Porod,a_cell, hwhm_q10, hwhm_q11, hwhm_q20, hwhm_q21, hwhm_q30, scale_q10, scale_q11, scale_q20, scale_q21, scale_q30):
    """
    @param q: Input q-value
    """
    with errstate(divide='ignore'):
      q0 = 4*np.pi/(np.sqrt(3)*a_cell)
      porod = (scale_Porod/q)**4
      L10 = 1/(1+((q-q0)/(hwhm_q10))**2)
      L11 = 1/(1+((q-np.sqrt(3)*q0)/(hwhm_q11))**2)
      L20 = 1/(1+((q-2*q0)/(hwhm_q20))**2)
      L21 = 1/(1+((q-np.sqrt(7)*q0)/(hwhm_q21))**2)
      L30 = 1/(1+((q-3*q0)/(hwhm_q30))**2)
      return porod+scale_q10*L10+scale_q11*L11+scale_q20*L20+scale_q21*L21+scale_q30*L30

Iq.vectorized = True  # Iq accepts an array of q values

def random():
    """Return a random parameter set for the model."""
    sld, solvent = np.random.uniform(-0.5, 12, size=2)
    radius = 10**np.random.uniform(1, 4.7)
    Vf = 10**np.random.uniform(-3, -1)
    scale = 1e-4 * Vf * 2*np.pi*(sld-solvent)**2/(3*radius)
    a_cell = np.random.uniform(1,100)
    hwhm_q10 = np.random.uniform(0.001,0.1)
    hwhm_q11 = np.random.uniform(0.001,0.1)
    hwhm_q20 = np.random.uniform(0.001,0.1)
    scale_q10 = np.random.uniform(0.01,100)
    scale_q11 = np.random.uniform(0.01,100)
    scale_q20 = np.random.uniform(0.01,100)
    pars = dict(
        scale=scale,
        a_cell = a_cell,
        hwhm_q10 = hwhm_q10,
        hwhm_q11 = hwhm_q11,
        hwhm_q20 = hwhm_q20,
        scale_q10 = scale_q10,
        scale_q11 = scale_q11,
        scale_q20 = scale_q20,
    )
    return pars

tests = [
    [{'scale': 0.00001, 'background':0.01}, 0.04, 3.916250],
    [{}, 0.0, inf],
]
