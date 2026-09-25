#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
Tests for the temporary replacement of zero mass flows in thermal simulations.

To prevent singular solution matrices, solve_thermal() replaces zero mass flows
with small values. Their signs must not turn idle nodes into artificial sources
or sinks.
"""

import unittest

import numpy as np

from pydhn import ConstantWater
from pydhn import Network
from pydhn import Soil
from pydhn.networks import star_network
from pydhn.solving import solve_hydraulics
from pydhn.solving import solve_thermal
from pydhn.solving.thermal_simulation import _fill_zero_mass_flow

# Use a value different than the default one
MASS_FLOW_MIN = 3e-12


def network_with_flows(edge_specs):
    """Builds a network of pipes from (name, start, end, mass_flow) tuples."""
    net = Network()
    nodes = sorted({n for _, u, v, _ in edge_specs for n in (u, v)})
    for i, node in enumerate(nodes):
        net.add_node(name=node, x=float(i), y=0.0, z=0.0)
    for name, u, v, _ in edge_specs:
        net.add_pipe(name=name, start_node=u, end_node=v, line="supply")
    flows = {name: mdot for name, _, _, mdot in edge_specs}
    _, names = net.edges("name")
    return net, np.array([flows[n] for n in names])


def fill(net, mass_flow):
    return _fill_zero_mass_flow(net, net.edges(), mass_flow.copy(), MASS_FLOW_MIN)


def flows_by_name(net, mass_flow, names):
    _, all_names = net.edges("name")
    flows = dict(zip(all_names, mass_flow))
    return np.array([flows[n] for n in names])


class FillZeroMassFlowTestCase(unittest.TestCase):
    def check_fill(self, mass_flow, filled):
        # Only the zeros are replaced, all with the requested magnitude
        idle = mass_flow == 0.0
        np.testing.assert_array_equal(np.abs(filled[idle]), MASS_FLOW_MIN)
        np.testing.assert_array_equal(filled[~idle], mass_flow[~idle])

    def check_open_path(self, net, mass_flow, filled, endpoints):
        # Within the zero flow edges, only the path ends are unbalanced
        idle = mass_flow == 0.0
        residual = net.incidence_matrix[:, idle] @ filled[idle]
        nodes, _ = net.nodes()
        self.assertSetEqual(set(nodes[residual != 0.0]), endpoints)
        np.testing.assert_array_equal(
            np.sort(residual[residual != 0.0]), [-MASS_FLOW_MIN, MASS_FLOW_MIN]
        )

    def test_closed_loops(self):
        """
        Two overlapping loops, where B -> X <- D and B -> Y <- D carry no
        flow. Both edges point into X (and Y), so one of each pair must be
        reversed to form a loop:

            ┌──────────> X <────────┐
            │                       │
            │      (1.0)            │
            │   ┌─────────A         │
            │   │         ^         │
            │   │         │(1.0)    │
            │   V         │         │
            ├── B <────── D ────────┤
            │      (-1.0) │         │
            │             │         │
            │             V         │
            └───────────> Y <───────┘
        """
        net, mass_flow = network_with_flows(
            [
                ("AB", "A", "B", 1.0),
                ("DB", "D", "B", -1.0),
                ("DA", "D", "A", 1.0),
                ("BX", "B", "X", 0.0),
                ("DX", "D", "X", 0.0),
                ("BY", "B", "Y", 0.0),
                ("DY", "D", "Y", 0.0),
            ]
        )
        filled = fill(net, mass_flow)
        self.check_fill(mass_flow, filled)
        # Closed loops conserve mass in every node
        np.testing.assert_array_equal(net.incidence_matrix @ (filled - mass_flow), 0)
        for pair in (["BX", "DX"], ["BY", "DY"]):
            np.testing.assert_array_equal(
                np.sort(flows_by_name(net, filled, pair)),
                [-MASS_FLOW_MIN, MASS_FLOW_MIN],
            )

    def test_open_path(self):
        """
        The zero flow path D -> Y -> X -> B must get a single direction:

                   (1.0)          (1.0)
               ┌───────────> A ───────────┐
               │                          │
               │           (1.0)          v
               D <─────────────────────── B
               │                          ^
               │                          │
               └─────> Y ─────> X ────────┘
        """
        net, mass_flow = network_with_flows(
            [
                ("AB", "A", "B", 1.0),
                ("BD", "B", "D", 1.0),
                ("DA", "D", "A", 1.0),
                ("DY", "D", "Y", 0.0),
                ("YX", "Y", "X", 0.0),
                ("XB", "X", "B", 0.0),
            ]
        )
        filled = fill(net, mass_flow)
        self.check_fill(mass_flow, filled)
        path = flows_by_name(net, filled, ["DY", "YX", "XB"])
        self.assertEqual(len(set(np.sign(path))), 1)
        self.check_open_path(net, mass_flow, filled, {"B", "D"})

    def test_open_path_mixed_orientation(self):
        """
        The zero flow path between B and D has edges pointing away from X3 in
        both directions, so one half must be reversed:

                    (1.0)             (1.0)
               ┌─────────────> A ──────────────┐
               │                               │
               │             (1.0)             v
               D <──────────────────────────── B
               ^                               ^
               │                               │
               X5 <─── X4 <─── X3 ───> X2 ───> X1
        """
        net, mass_flow = network_with_flows(
            [
                ("AB", "A", "B", 1.0),
                ("BD", "B", "D", 1.0),
                ("DA", "D", "A", 1.0),
                ("X1_B", "X1", "B", 0.0),
                ("X2_X1", "X2", "X1", 0.0),
                ("X3_X2", "X3", "X2", 0.0),
                ("X3_X4", "X3", "X4", 0.0),
                ("X4_X5", "X4", "X5", 0.0),
                ("X5_D", "X5", "D", 0.0),
            ]
        )
        filled = fill(net, mass_flow)
        self.check_fill(mass_flow, filled)
        to_b = np.sign(flows_by_name(net, filled, ["X3_X2", "X2_X1", "X1_B"]))
        to_d = np.sign(flows_by_name(net, filled, ["X3_X4", "X4_X5", "X5_D"]))
        np.testing.assert_array_equal(to_b, to_b[0])
        np.testing.assert_array_equal(to_d, -to_b[0])
        self.check_open_path(net, mass_flow, filled, {"B", "D"})

    def test_idle_consumer_branch(self):
        """The thermal solver must converge with a consumer switched off."""
        net = star_network()
        _, names = net.edges("name")
        sub1 = np.where(names == "SUB1")[0]
        net.set_edge_attribute(0.0, "heat_demand", mask=sub1)
        net.set_edge_attribute(0.0, "mass_flow_min", mask=sub1)
        solve_hydraulics(net, ConstantWater(), error_threshold=1e-6, verbose=0)

        mass_flow = net.get_edges_attribute_array("mass_flow")
        self.assertSetEqual(set(names[mass_flow == 0.0]), {"SP6", "SUB1", "RP6"})
        self.check_fill(mass_flow, fill(net, mass_flow))

        results = solve_thermal(
            net, ConstantWater(), Soil(), error_threshold=1e-13, verbose=0
        )
        self.assertTrue(results["history"]["thermal converged"])


if __name__ == "__main__":
    unittest.main()
