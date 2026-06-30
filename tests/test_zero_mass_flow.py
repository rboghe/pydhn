#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Tests for zero-mass-flow filling in pydhn.solving.thermal_simulation"""

import unittest

import numpy as np

from pydhn.networks import star_network
from pydhn.solving.thermal_simulation import _fill_zero_mass_flow


def _fill(net, zero_names):
    """Run _fill_zero_mass_flow on a network with the named edges zeroed."""
    edges, _ = net.edges("mass_flow")
    nodes, _ = net.nodes()
    names = net.get_edges_attribute_array("name")
    mass_flow = np.ones(len(edges))
    zero_mask = np.isin(names, zero_names)
    mass_flow[zero_mask] = 0.0
    out = _fill_zero_mass_flow(net, edges, nodes, mass_flow.copy())
    return out, zero_mask


class FillZeroMassFlowTestCase(unittest.TestCase):
    def test_even_degree_subgraph(self):
        """
        A zero-flow subgraph whose nodes all have even degree (here the closed
        loop S2-S3-S5-S4) must be filled without error.
        """
        net = star_network()
        out, zero_mask = _fill(net, ["SP2", "SP3", "SP4", "SP5"])
        self.assertFalse(np.isin(0.0, out))
        np.testing.assert_array_equal(np.abs(out[zero_mask]), 1e-16)
        np.testing.assert_array_equal(out[~zero_mask], 1.0)

    def test_odd_degree_subgraph(self):
        """A zero-flow subgraph with odd-degree nodes is filled as well."""
        net = star_network()
        out, zero_mask = _fill(net, ["SP1"])
        self.assertFalse(np.isin(0.0, out))
        np.testing.assert_array_equal(np.abs(out[zero_mask]), 1e-16)
        np.testing.assert_array_equal(out[~zero_mask], 1.0)


if __name__ == "__main__":
    unittest.main()
