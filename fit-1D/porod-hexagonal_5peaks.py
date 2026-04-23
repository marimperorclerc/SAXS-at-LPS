r"""
This model adds a q**-4 Porod regime to an hexagonal layer of objects, with the first three peaks present

References
----------

#. G Porod. *Kolloid Zeit*. 124 (1951) 83
#. L A Feigin, D I Svergun, G W Taylor. *Structure Analysis by Small-Angle
   X-ray and Neutron Scattering*. Springer. (1987)

Authorship and Verification
----------------------------

* **Author:**
* **Last Modified by: Paul Butler, 28Mar2020**
* **Last Reviewed by:**
"""

import numpy as np
from numpy import inf, errstate

name = "porod-hexagonal_5peaks"
title = "Porod function with 5 peaks"
description = """\
          I(q) = scale/q^4 + background
"""

category = "custom"

parameters = [["a_cell", "Ang", 100, [0, inf], "", "hexagonal cell parameter"],
              ["hwhm_q10", "1/Ang", 0.005, [-inf, inf], "", "HWHM of q10 peak"],
              ["hwhm_q11", "1/Ang", 0.005, [-inf, inf], "", "HWHM of q11 peak"],
              ["hwhm_q20", "1/Ang", 0.005, [-inf, inf], "", "HWHM of q20 peak"],
              ["hwhm_q21", "1/Ang", 0.005, [-inf, inf], "", "HWHM of q21 peak"],
              ["hwhm_q30", "1/Ang", 0.005, [-inf, inf], "", "HWHM of q30 peak"],
              ["scale_q10", "", 1.e+7, [0, inf], "", "Scale factor for q10 peak"],
              ["scale_q11", "", 1.e+6, [0, inf], "", "Scale factor for q11 peak"],
              ["scale_q20", "", 1.e+6, [0,inf], "", "Scale factor for q20 peak"],
              ["scale_q21", "", 1.e+4, [0, inf], "", "Scale factor for q21 peak"],
              ["scale_q30", "", 1.e+4, [0, inf], "", "Scale factor for q30 peak"]]

def Iq(q, a_cell, hwhm_q10, hwhm_q11, hwhm_q20, hwhm_q21, hwhm_q30, scale_q10, scale_q11, scale_q20, scale_q21, scale_q30):
    """
    @param q: Input q-value
    """
    with errstate(divide='ignore'):
      coeff=1.e-5
      q0 = 4*np.pi/(np.sqrt(3)*a_cell)  
      porod = (1./q)**4
      L10 = 1./(1+((q-q0)/(hwhm_q10))**2)
      L11 = 1./(1+((q-np.sqrt(3)*q0)/(hwhm_q11))**2)
      L20 = 1./(1+((q-2*q0)/(hwhm_q20))**2)
      L21 = 1./(1+((q-np.sqrt(7)*q0)/(hwhm_q21))**2)
      L30 = 1./(1+((q-3*q0)/(hwhm_q30))**2)
      return coeff*(porod+scale_q10*L10+scale_q11*L11+scale_q20*L20+scale_q21*L21+scale_q30*L30)

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
    hwhm_q21 = np.random.uniform(0.001,0.1)
    hwhm_q30 = np.random.uniform(0.001,0.1)
    scale_q10 = np.random.uniform(0.01,100)
    scale_q11 = np.random.uniform(0.01,100)
    scale_q20 = np.random.uniform(0.01,100)
    scale_q21 = np.random.uniform(0.01,100)
    scale_q30 = np.random.uniform(0.01,100)
    pars = dict(
        scale=scale,
        a_cell = a_cell,
        hwhm_q10 = hwhm_q10,
        hwhm_q11 = hwhm_q11,
        hwhm_q20 = hwhm_q20,
        hwhm_q21 = hwhm_q21,
        hwhm_q30 = hwhm_q30,
        scale_q10 = scale_q10,
        scale_q11 = scale_q11,
        scale_q20 = scale_q20,
        scale_q21 = scale_q21,
        scale_q30 = scale_q30,
    )
    return pars

tests = [
    [{'scale': 0.00001, 'background':0.01}, 0.04, 3.916250],
    [{}, 0.0, inf],
]
