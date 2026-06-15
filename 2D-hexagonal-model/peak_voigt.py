r"""
This model describes a pseudo-Voigt shaped peak on a flat background.

Definition
----------

This pseudo-Voigt peak function is a weighted linear summation of
Lorentzian (L) and Gaussian (G) peak shapes. The usefulness of this
function is that it produces a peak shape with asymmetry.
 
The scattering intensity $I(q)$ is calculated as

.. math::

    I(q) = scale . [{W_f.I(q)_L} + {(1-W_f).I(q)_G}] + background
	
where $W_f$ is a weighting factor and

.. math::

    I(q)_L = frac{1}{igl(1+igl(frac{q-q_0}{HWHM}igr)^2igr)}


    I(q)_G = expleft[ -frac12 (q-q_0)^2 / sigma^2 
ight]

The peak is taken to be centered at $q_0$ with a HWHM (half-width
half-maximum) of 1.177 $sigma$, where $sigma$ is the standard deviation
of the Gaussian. In other words, the widths of the Lorentzian and the
Gaussian have been coupled for convenience of parameterisation.

When $W_f$ = 1 a Lorentzian peak is returned, and when $W_f$ = 0 a
Gaussian peak is returned.

For practical purposes 0 < $sigma$ < 0.1 else no peak is generated.

For 2D data the scattering intensity is calculated in the same way as 1D,
where the $q$ vector is defined as

.. math::

    q = sqrt{q_x^2 + q_y^2}

References
----------
Aaron L. Stancik, Eric B. Brauns
A simple asymmetric lineshape for fitting infrared absorption spectra
Vibrational Spectroscopy 47 (2008) 66-69

Authorship and Verification
---------------------------

* **Author:** Steve King **Date:** 19/11/2019
* **Last Modified by:** Steve King **Date:** 24/06/2020
* **Last Reviewed by:** **Date:**

"""

import numpy as np
from numpy import inf

name = "peak_voigt"
title = "A Voigt peak on a flat background"
description = """
        Evaluates a pseudo-Voigt shaped peak."""

category = "shape-independent"

#             ["name", "units", default, [lower, upper], "type", "description"],
parameters = [["wf", "", 0.5, [0, 1], "",
               "Weighting factor"],
              ["q0", "1/Ang", 0.05, [-inf, inf], "",
               "Peak position in q"],
              ["sigma", "1/Ang", 0.005, [0, inf], "",
               "Peak width (Std dev)"]]

def Iq(q, wf, q0, sigma):
    hwhm = 1.177*sigma
    intens = (wf*(1/(1+((q-q0)**2.0/hwhm**2.0))))+((1.0-wf)*np.exp((-0.5*(q-q0)**2.0)/(sigma**2.0)))
    return intens

Iq.vectorized = True  # Iq accepts an array of q values

demo = dict(scale=100, background=0.1,
            wf=0.5, q0=0.05, sigma=0.005)

tests = [
    [{'scale': 1.0, 'background' : 0.001, 'wf' : 0.5,
      'q0' : 0.05, 'sigma' : 0.005},
     [0.0005, 0.0514693877551], [0.00796878321113, 0.950526807316]],
	]
