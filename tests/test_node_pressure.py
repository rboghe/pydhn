#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Tests for nodal pressure assignment in pydhn.solving"""

import unittest

import numpy as np

from pydhn import ConstantWater
from pydhn.networks import star_network
from pydhn.solving import solve_hydraulics
from pydhn.solving.hydraulic_simulation import _assign_node_pressure


def _max_edge_inconsistency(net):
    """Largest mismatch between the nodal pressure drop and the edge delta_p."""
    errors = [
        net[u]["pressure"] - net[v]["pressure"] - net[(u, v)]["delta_p"]
        for u, v in net._graph.edges()
    ]
    return np.max(np.abs(errors))


class NodePressureTestCase(unittest.TestCase):
    def setUp(self):
        self.net = star_network()
        solve_hydraulics(self.net, ConstantWater(), error_threshold=1e-9, verbose=0)

    def test_default_source_consistent(self):
        """With source=None the nodal pressures match edge delta_p everywhere."""
        _assign_node_pressure(self.net)
        np.testing.assert_almost_equal(
            _max_edge_inconsistency(self.net), 0.0, decimal=6
        )

    def test_explicit_source(self):
        """
        Passing an explicit source node must not raise: the given node keeps its
        pre-set pressure and the rest are propagated consistently from it.
        """
        source = "S5"
        self.net[source]["pressure"] = 5e5
        _assign_node_pressure(self.net, source=source)
        self.assertEqual(self.net[source]["pressure"], 5e5)
        np.testing.assert_almost_equal(
            _max_edge_inconsistency(self.net), 0.0, decimal=6
        )


if __name__ == "__main__":
    unittest.main()
