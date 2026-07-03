#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Tests for the network incidence and cycle matrices (pydhn.utilities.matrices)"""

import unittest

import numpy as np

from pydhn import Network
from pydhn.networks import star_network
from pydhn.utilities.matrices import compute_cycle_matrix


def two_producer_network():
    """
    A small meshed network with two producers (one pressure-controlled, one
    mass-flow-controlled) and one consumer, exercising the multi-producer path
    of compute_network_cycle_matrix.
    """
    net = Network()
    for n in ["S1", "S2", "S3", "R1", "R2", "R3"]:
        net.add_node(name=n, x=0.0, y=0.0, z=0.0)
    # Meshed supply and return triangles (each contributes an internal loop)
    pipes = [
        ("SP1", "S1", "S2", "supply"),
        ("SP2", "S2", "S3", "supply"),
        ("SP3", "S1", "S3", "supply"),
        ("RP1", "R2", "R1", "return"),
        ("RP2", "R3", "R2", "return"),
        ("RP3", "R3", "R1", "return"),
    ]
    for name, u, v, line in pipes:
        net.add_pipe(name=name, start_node=u, end_node=v, length=10, diameter=0.02, line=line)
    net.add_consumer(name="SUB1", start_node="S3", end_node="R3")
    net.add_producer(name="main", start_node="R1", end_node="S1")
    net.add_producer(
        name="prod2",
        start_node="R2",
        end_node="S2",
        setpoint_type_hyd="mass_flow",
        setpoint_value_hyd=0.5,
    )
    return net


# Both fixtures are connected graphs, so the cycle space has dimension E - N + 1.
NETWORKS = {"star (1 producer)": star_network, "meshed (2 producers)": two_producer_network}


class CycleMatrixTestCase(unittest.TestCase):
    def test_rows_are_circulations(self):
        """Each cycle-matrix row is a valid circulation: incidence @ Bᵀ = 0."""
        for label, factory in NETWORKS.items():
            net = factory()
            with self.subTest(network=label):
                np.testing.assert_allclose(
                    net.incidence_matrix @ net.cycle_matrix.T, 0.0, atol=1e-12
                )

    def test_cyclomatic_number(self):
        """The number of independent loops equals E - N + 1."""
        for label, factory in NETWORKS.items():
            net = factory()
            with self.subTest(network=label):
                self.assertEqual(
                    net.cycle_matrix.shape[0], net.n_edges - net.n_nodes + 1
                )

    def test_cycles_are_independent(self):
        """The cycle basis has full row rank."""
        for label, factory in NETWORKS.items():
            net = factory()
            B = net.cycle_matrix
            with self.subTest(network=label):
                self.assertEqual(np.linalg.matrix_rank(B), B.shape[0])

    def test_nx_cycle_basis(self):
        """compute_cycle_matrix(method='nx') is a valid full cycle basis too."""
        for label, factory in NETWORKS.items():
            net = factory()
            B = compute_cycle_matrix(net, method="nx")
            with self.subTest(network=label):
                np.testing.assert_allclose(net.incidence_matrix @ B.T, 0.0, atol=1e-12)
                self.assertEqual(B.shape[0], net.n_edges - net.n_nodes + 1)
                self.assertEqual(np.linalg.matrix_rank(B), B.shape[0])

    def test_consumers_cycle_matrix(self):
        """
        consumers_cycle_matrix holds one independent circulation per consumer
        and per secondary producer (producer-to-consumer loops).
        """
        for label, factory in NETWORKS.items():
            net = factory()
            B = net.consumers_cycle_matrix
            expected = len(net.consumers_mask) + len(net.producers_mask) - 1
            with self.subTest(network=label):
                np.testing.assert_allclose(net.incidence_matrix @ B.T, 0.0, atol=1e-12)
                self.assertEqual(B.shape[0], expected)
                self.assertEqual(np.linalg.matrix_rank(B), B.shape[0])


class IncidenceMatrixTestCase(unittest.TestCase):
    def test_oriented_columns_sum_to_zero(self):
        """
        Every edge column of the oriented incidence matrix has a single +1 and
        a single -1 (tail and head), so columns sum to zero and abs-sum to two.
        """
        net = two_producer_network()
        A = net.incidence_matrix
        self.assertEqual(A.shape, (net.n_nodes, net.n_edges))
        np.testing.assert_array_equal(A.sum(axis=0), np.zeros(net.n_edges))
        np.testing.assert_array_equal(np.abs(A).sum(axis=0), 2 * np.ones(net.n_edges))


if __name__ == "__main__":
    unittest.main()
