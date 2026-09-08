#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
Tests for the temporary replacement of zero mass flows in thermal simulations.

To prevent singular solution matrices, `solve_thermal` replaces zero mass flows 
with small values. The sign of the replacement values should be fixed such 
that there are never zero-flow nodes with only outgoing flows.
"""

import unittest

import numpy as np

from pydhn import ConstantWater, Network, Soil
from pydhn.networks import star_network
from pydhn.solving import solve_hydraulics, solve_thermal
from pydhn.solving.thermal_simulation import _fill_zero_mass_flow

# Use a value different than the default one
MASS_FLOW_MIN = 3e-12


def _fill(net, mass_flow):
    """Return a mass flow vector with filled zero entries."""
    nodes, _ = net.nodes("temperature")
    return _fill_zero_mass_flow(
        net,
        net.edges(),
        nodes,
        mass_flow.copy(),
        mass_flow_min=MASS_FLOW_MIN,
    )


def _named_flows(net, mass_flow, edge_names):
    """Return mass flows in the requested edge-name order."""
    # Build a name-to-flow lookup.
    _, names = net.edges("name")
    flow_by_name = dict(zip(names, mass_flow))

    # Ignore the network's internal edge order.
    return np.array([flow_by_name[name] for name in edge_names])


def _network_with_flows(edge_specs):
    """
    Build a toy network and its mass flow array from named edge specs.

    `edge_specs` contains `(name, start, end, mass_flow)` entries.
    """
    net = Network()
    nodes = sorted({node for _, start, end, _ in edge_specs for node in (start, end)})
    for i, node in enumerate(nodes):
        net.add_node(name=node, x=float(i), y=0.0, z=0.0)

    flow_by_name = {}
    for name, start, end, mass_flow in edge_specs:
        net.add_pipe(
            name=name,
            start_node=start,
            end_node=end,
            length=10.0,
            diameter=0.02,
            roughness=0.045,
            line="supply",
        )
        flow_by_name[name] = mass_flow

    _, names = net.edges("name")
    return net, np.array([flow_by_name[name] for name in names])


def _partial_zero_flow_loops_network():
    """
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
              
              
    Return two overlapping loops with zero-flow paths.

    `A -> B <- D -> A` carries flow (negative in edge DB).
    `B -> X <- D` and `B -> Y <- D` are zero-flow paths.

    Because the stored edge orientations converge at C and Y, a valid fill
    must assign negative mass flow to one of the opposing edges (e.g., `DX` or 
    `DY`) to form a valid circulation.
    """
    return _network_with_flows(
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


def _full_path_reversed_zero_flow_network():
    """
           (1.0)          (1.0)
       ┌───────────> A ───────────┐
       │                          │
       │           (1.0)          v
       D <─────────────────────── B
       │                          ^
       │                          │
       └─────> Y ─────> X ────────┘

    Return an active loop with a three-edge zero-flow path. 
    
    `A -> B -> D -> A` carries unit flow. 
    The zero-flow path has stored orientation `D -> Y -> X -> B`. 
    
    When zero flows are replaced with signed epsilon flows, the three edges 
    must be oriented consistently along a single Eulerian path. Depending on 
    the Eulerian traversal, this may either preserve the stored orientation 
    (`D -> Y -> X -> B`) or reverse the entire path (`B -> X -> Y -> D`).
    
    In either case, the internal nodes X and Y remain balanced: each has one 
    incoming and one outgoing epsilon flow.
    """
    return _network_with_flows(
        [
            # Active unit-flow loop: A -> B -> D -> A
            ("AB", "A", "B", 1.0),
            ("BD", "B", "D", 1.0),
            ("DA", "D", "A", 1.0),
            # Zero-flow path: all 3 edges point D -> Y -> X -> B
            ("DY", "D", "Y", 0.0),
            ("YX", "Y", "X", 0.0),
            ("XB", "X", "B", 0.0),
        ]
    )


def _mixed_orientation_zero_flow_path_network():
    """
            (1.0)             (1.0)
       ┌─────────────> A ──────────────┐
       │                               │
       │             (1.0)             v
       D <──────────────────────────── B
       ^                               ^
       │                               │
       X5 <─── X4 <─── X3 ───> X2 ───> X1
       
    Return an active loop with a mixed-orientation six-edge zero-flow path.

    `A -> B -> D -> A` carries unit flow.

    The zero-flow edges form the undirected path
    `B - X1 - X2 - X3 - X4 - X5 - D`. Their stored orientations point
    away from X3 toward both endpoints:
    `X3 -> X2 -> X1 -> B` and `X3 -> X4 -> X5 -> D`.

    When zero flows are replaced with signed epsilon flows, all six edges
    must be oriented consistently along one Eulerian path between B and D.
    Whichever endpoint is chosen first, one three-edge half follows its
    stored orientation while the other three-edge half is traversed in
    reverse. Thus the path exercises a mixture of positive and negative
    epsilon assignments without depending on the traversal direction.
    """
    return _network_with_flows(
        [
            # Active unit-flow loop: A -> B -> D -> A
            ("AB", "A", "B", 1.0),
            ("BD", "B", "D", 1.0),
            ("DA", "D", "A", 1.0),
            # Zero-flow path: stored orientations diverge from X3 toward B
            ("X1_B", "X1", "B", 0.0),
            ("X2_X1", "X2", "X1", 0.0),
            ("X3_X2", "X3", "X2", 0.0),
            # ...and from X3 toward D
            ("X3_X4", "X3", "X4", 0.0),
            ("X4_X5", "X4", "X5", 0.0),
            ("X5_D", "X5", "D", 0.0),
        ]
    )


def _idle_consumer_network():
    """Return a hydraulic solution with SUB1 and its branch switched off."""
    net = star_network()

    # Disable the consumer demand.
    _, names = net.edges("name")
    sub1 = np.where(names == "SUB1")[0]
    net.set_edge_attribute(0.0, "heat_demand", mask=sub1)

    # Allow the branch flow to reach zero.
    net.set_edge_attribute(0.0, "mass_flow_min", mask=sub1)

    # Compute the resulting hydraulic state.
    solve_hydraulics(net, ConstantWater(), error_threshold=1e-6, verbose=0)

    return net


class FillZeroMassFlowTestCase(unittest.TestCase):
    def assert_only_zero_flows_are_replaced(self, mass_flow, filled):
        """Assert that only exact zero flows are replaced."""
        # Identify the edges the helper may modify.
        idle = mass_flow == 0.0
        self.assertTrue(np.any(idle), "fixture must contain zero-flow edges")

        # Every zero must be filled.
        self.assertFalse(np.any(filled[idle] == 0.0))

        # Every fill must have the requested magnitude.
        np.testing.assert_array_equal(
            np.abs(filled[idle]),
            MASS_FLOW_MIN,
        )

        # Existing hydraulic flows must remain untouched.
        np.testing.assert_array_equal(
            filled[~idle],
            mass_flow[~idle],
        )

    def assert_open_path_endpoints(
        self, net, mass_flow, filled, expected_endpoints
    ):
        """Assert that only the expected open-path endpoints are unbalanced."""
        # Restrict the balance to originally idle edges.
        idle = mass_flow == 0.0

        # Compute the epsilon-flow residual at every node.
        residual = net.incidence_matrix[:, idle] @ filled[idle]

        # Get node names in incidence-matrix order.
        nodes, _ = net.nodes("temperature")

        # Nonzero residuals are the path endpoints.
        endpoints = set(nodes[residual != 0.0])

        self.assertSetEqual(endpoints, set(expected_endpoints))

        # A path has one source and one sink.
        endpoint_residual = residual[residual != 0.0]
        self.assertEqual(np.sum(endpoint_residual > 0.0), 1)
        self.assertEqual(np.sum(endpoint_residual < 0.0), 1)

        # Both endpoint residuals have epsilon magnitude.
        np.testing.assert_array_equal(
            np.abs(endpoint_residual),
            MASS_FLOW_MIN,
        )

    def test_closed_zero_flow_loops_form_circulation(self):
        """Orient overlapping zero-flow paths into a closed circulation."""
        net, mass_flow = _partial_zero_flow_loops_network()

        # Fill the four zero-flow edges.
        filled = _fill(net, mass_flow)

        # Only zero flows may change.
        self.assert_only_zero_flows_are_replaced(mass_flow, filled)

        # Isolate the epsilon perturbation.
        perturbation = filled - mass_flow

        # The closed zero-flow component must conserve mass exactly.
        np.testing.assert_array_equal(
            net.incidence_matrix @ perturbation,
            0.0,
        )

        # Both stored edges point into X.
        at_x = _named_flows(net, filled, ["BX", "DX"])

        # One of them must therefore be traversed in reverse.
        np.testing.assert_array_equal(
            np.sort(at_x),
            [-MASS_FLOW_MIN, MASS_FLOW_MIN],
        )

        # Both stored edges also point into Y.
        at_y = _named_flows(net, filled, ["BY", "DY"])

        # Again, one must be traversed in reverse.
        np.testing.assert_array_equal(
            np.sort(at_y),
            [-MASS_FLOW_MIN, MASS_FLOW_MIN],
        )

    def test_uniformly_oriented_open_path_gets_consistent_direction(self):
        """Orient all edges consistently along a three-edge open path."""
        net, mass_flow = _full_path_reversed_zero_flow_network()

        # Fill the zero-flow path.
        filled = _fill(net, mass_flow)

        # Only zero flows may change.
        self.assert_only_zero_flows_are_replaced(mass_flow, filled)

        # Read the path in stored orientation.
        path_flow = _named_flows(net, filled, ["DY", "YX", "XB"])

        # The Eulerian path may run in either direction.
        first_sign = np.sign(path_flow[0])

        # All three edges must nevertheless agree.
        np.testing.assert_array_equal(
            np.sign(path_flow),
            np.full(3, first_sign),
        )

        # Only B and D may remain unbalanced within the idle subgraph.
        self.assert_open_path_endpoints(
            net,
            mass_flow,
            filled,
            expected_endpoints={"B", "D"},
        )

    def test_mixed_orientation_open_paths(
        self,
    ):
        """Orient a mixed six-edge path continuously between its endpoints."""
        net, mass_flow = _mixed_orientation_zero_flow_path_network()

        # Fill the six zero-flow edges.
        filled = _fill(net, mass_flow)

        # Only zero flows may change.
        self.assert_only_zero_flows_are_replaced(mass_flow, filled)

        # These three stored orientations point from X3 toward B.
        toward_b = _named_flows(
            net,
            filled,
            ["X3_X2", "X2_X1", "X1_B"],
        )

        # These three stored orientations point from X3 toward D.
        toward_d = _named_flows(
            net,
            filled,
            ["X3_X4", "X4_X5", "X5_D"],
        )

        # Each half must be internally consistent.
        self.assertTrue(np.all(np.sign(toward_b) == np.sign(toward_b[0])))
        self.assertTrue(np.all(np.sign(toward_d) == np.sign(toward_d[0])))

        # A B-D traversal follows one half and reverses the other.
        self.assertEqual(np.sign(toward_b[0]), -np.sign(toward_d[0]))

        # Therefore exactly three fills are positive.
        idle_flow = filled[mass_flow == 0.0]
        self.assertEqual(np.sum(idle_flow > 0.0), 3)

        # And exactly three fills are negative.
        self.assertEqual(np.sum(idle_flow < 0.0), 3)

        # Only B and D may remain unbalanced within the idle subgraph.
        self.assert_open_path_endpoints(
            net,
            mass_flow,
            filled,
            expected_endpoints={"B", "D"},
        )

    def test_thermal_solver_handles_real_zero_flow_branch(self):
        """Run the thermal solver with a hydraulically idle consumer branch."""
        net = _idle_consumer_network()

        # Read the hydraulic result.
        _, names = net.edges("name")
        mass_flow = net.get_edges_attribute_array("mass_flow")

        # Check that the intended branch is actually idle.
        idle_names = {
            name
            for name, flow in zip(names, mass_flow)
            if flow == 0.0
        }
        self.assertSetEqual(idle_names, {"SP6", "SUB1", "RP6"})

        # Exercise the fill directly as well.
        filled = _fill(net, mass_flow)

        # Every zero must become a signed epsilon.
        self.assert_only_zero_flows_are_replaced(mass_flow, filled)

        # The thermal solve must handle the idle branch.
        results = solve_thermal(
            net,
            ConstantWater(),
            Soil(),
            error_threshold=1e-13,
            verbose=0,
        )

        # The end-to-end thermal solve must converge.
        self.assertTrue(results["history"]["thermal converged"])


if __name__ == "__main__":
    unittest.main()
