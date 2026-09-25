#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Tests for time step IDs and zero mass flows in solve_thermal()"""

import unittest
import warnings
from itertools import product
from unittest.mock import patch

import numpy as np

from pydhn import ConstantWater
from pydhn import Network
from pydhn import Soil
from pydhn.components.vector_functions import COMPONENT_FUNCTIONS_DICT
from pydhn.networks import star_network
from pydhn.solving import solve_hydraulics
from pydhn.solving import solve_thermal

FLUID, SOIL = ConstantWater(), Soil()


def idle_loop():
    """
    Loop A -> B -> C -> A with zero mass flow. A -> B is a Lagrangian pipe
    with hot water on the A side and cold water on the B side.
    """
    net = Network()
    for name in ("A", "B", "C"):
        net.add_node(name, temperature=50.0)
    net.add_lagrangian_pipe(
        "P",
        "A",
        "B",
        mass_flow=0.0,
        length=10.0,
        diameter=0.02,
        stepsize=10.0,
        reynolds=0.0,
        friction_factor=0.0,
    )
    pipe = net["A", "B"]
    pipe._volumes = np.full(2, pipe._internal_volume / 2)
    pipe._temperatures = np.array([80.0, 30.0])
    pipe._wall_temperatures = pipe._temperatures.copy()
    net.add_pipe("R", "B", "C", mass_flow=0.0, length=1.0)
    net.add_producer("H", "C", "A", mass_flow=0.0)
    return net


class ThermalSimulationTestCase(unittest.TestCase):
    def test_automatic_ts_id(self):
        """
        Without a ts_id, networks with dynamic components warn and continue
        from the last ID used, which can also be an explicit one.
        """
        net = idle_loop()
        pipe = net["A", "B"]
        for expected in (0, 1):
            with self.assertWarnsRegex(UserWarning, "No ts_id given"):
                solve_thermal(net, FLUID, SOIL, verbose=0)
            self.assertEqual(pipe._last_ts, expected)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            solve_thermal(net, FLUID, SOIL, ts_id=10, verbose=0)
        with self.assertWarnsRegex(UserWarning, "No ts_id given"):
            solve_thermal(net, FLUID, SOIL, verbose=0)
        self.assertEqual(pipe._last_ts, 11)

    def test_no_warning_without_dynamic_components(self):
        net = star_network()
        solve_hydraulics(net, FLUID, verbose=0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            solve_thermal(net, FLUID, SOIL, verbose=0)

    def test_idle_dynamic_components(self):
        """
        An idle Lagrangian pipe gives its outlet temperature at the end chosen
        by the solver, both with the vectorized and the scalar model. Its zero
        mass flow is then restored.
        """
        vector = COMPONENT_FUNCTIONS_DICT["lagrangian_pipe"]
        scalar = {"delta_p": vector["delta_p"]}
        for functions, direction in product((vector, scalar), (1, -1)):
            with self.subTest(vectorized=functions is vector, direction=direction):
                net = idle_loop()
                flows = np.full(net.n_edges, direction * 1e-16)
                with patch.dict(
                    COMPONENT_FUNCTIONS_DICT, {"lagrangian_pipe": functions}
                ), patch(
                    "pydhn.solving.thermal_simulation._fill_zero_mass_flow",
                    return_value=flows,
                ):
                    results = solve_thermal(
                        net, FLUID, SOIL, ts_id=0, verbose=0, error_threshold=1e-20
                    )
                self.assertTrue(results["history"]["thermal converged"])
                # Cold water leaves from B, hot water from A
                outlet, t_out = ("B", 30.0) if direction > 0 else ("A", 80.0)
                self.assertAlmostEqual(net[outlet]["temperature"], t_out, delta=1)
                mass_flow = net.get_edges_attribute_array("mass_flow")
                np.testing.assert_array_equal(mass_flow, 0.0)

    def test_mass_flows_restored_after_errors(self):
        for error in (RuntimeError, KeyboardInterrupt):
            with self.subTest(error=error.__name__):
                net = idle_loop()
                with patch(
                    "pydhn.solving.thermal_simulation.compute_edge_temperatures",
                    side_effect=error,
                ), self.assertRaises(error):
                    solve_thermal(net, FLUID, SOIL, ts_id=0, verbose=0)
                mass_flow = net.get_edges_attribute_array("mass_flow")
                np.testing.assert_array_equal(mass_flow, 0.0)

    def test_idle_consumer_with_imposed_heat(self):
        """An idle consumer exchanges no heat, even with a delta_q setpoint."""
        net = star_network()
        consumer = net["S8", "R8"]
        consumer.set("control_type", "mass_flow")
        consumer.set("setpoint_type_hyd", "mass_flow")
        consumer.set("setpoint_value_hyd", 0.0)
        consumer.set("setpoint_type_hx", "delta_q")
        consumer.set("setpoint_value_hx", -5000.0)
        solve_hydraulics(net, FLUID, verbose=0)
        solve_thermal(net, FLUID, SOIL, verbose=0)
        self.assertEqual(consumer["delta_q"], 0.0)
        self.assertEqual(consumer["outlet_temperature"], net["S8"]["temperature"])


if __name__ == "__main__":
    unittest.main()
