#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Tests for the dimensionless numbers in pydhn.fluids"""

import unittest

import numpy as np

from pydhn.default_values import CP_FLUID
from pydhn.default_values import K_FLUID
from pydhn.default_values import MU_FLUID
from pydhn.fluids.dimensionless_numbers import compute_nusselt
from pydhn.fluids.dimensionless_numbers import laminar_nusselt

# Fixed geometry/flow used across tests. Diameter and length do not enter the
# Nusselt correlations directly; the friction factor is held constant so that
# Nu is a function of Reynolds alone and its regime behaviour can be checked.
DIAMETER = 0.05
LENGTH = 100.0
FRICTION_FACTOR = 0.03
PRANDTL = CP_FLUID * MU_FLUID / K_FLUID


def _nu(reynolds):
    return compute_nusselt(
        reynolds=reynolds,
        diameter=DIAMETER,
        length=LENGTH,
        friction_factor=FRICTION_FACTOR,
    )


def _gnielinski_reference(reynolds, prandtl, friction_factor):
    """Independent implementation of the Gnielinski correlation."""
    fb = friction_factor / 8.0
    num = fb * (reynolds - 1000.0) * prandtl
    denom = 1.0 + 12.7 * np.sqrt(fb) * (prandtl ** (2 / 3) - 1.0)
    return num / denom


class NusseltTestCase(unittest.TestCase):
    def test_laminar_regime(self):
        """Below the laminar threshold Nu is the constant 4.36."""
        reynolds = np.array([1.0, 100.0, 1000.0, 2000.0, 2300.0])
        np.testing.assert_array_almost_equal(
            _nu(reynolds), laminar_nusselt(), decimal=12
        )

    def test_turbulent_closed_form(self):
        """In the turbulent regime Nu matches the Gnielinski correlation."""
        # Hardcoded anchor (Re=1e4, Pr~2.797, f=0.03), computed by hand from the
        # correlation, to catch edits that change both implementations at once.
        np.testing.assert_almost_equal(_nu(1e4), 53.45087269308247, decimal=8)
        # Self-consistency across the whole valid Reynolds range
        reynolds = np.array([3000.0, 1e4, 1e5, 1e6, 5e6])
        np.testing.assert_allclose(
            _nu(reynolds),
            _gnielinski_reference(reynolds, PRANDTL, FRICTION_FACTOR),
            rtol=1e-10,
        )

    def test_transition_continuity(self):
        """
        Nu is continuous at both ends of the transition band (2300, 3000).
        """
        # Lower boundary: transition starts from the laminar value
        np.testing.assert_almost_equal(_nu(2300.0 + 1e-6), laminar_nusselt(), decimal=4)
        # Upper boundary: transition meets the turbulent branch
        np.testing.assert_almost_equal(_nu(3000.0 - 1e-6), _nu(3000.0), decimal=4)

    def test_transition_monotonic(self):
        """Nu increases strictly across the transition band."""
        reynolds = np.linspace(2300.0, 3000.0, 50)
        self.assertTrue(np.all(np.diff(_nu(reynolds)) > 0))

    def test_scalar_array_consistency(self):
        """Scalar input returns a float equal to the array result."""
        scalar = _nu(5000.0)
        self.assertIsInstance(scalar, float)
        np.testing.assert_almost_equal(scalar, _nu(np.array([5000.0]))[0], decimal=12)


if __name__ == "__main__":
    unittest.main()
