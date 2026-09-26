#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Tests for compute_edge_temperatures() with a subset of the edges"""

import unittest
import warnings

import numpy as np

from pydhn import ConstantWater
from pydhn.components import Consumer
from pydhn.default_values import CP_FLUID
from pydhn.networks import star_network
from pydhn.soils import KusudaSoil
from pydhn.solving import solve_hydraulics
from pydhn.solving.temperature import compute_edge_temperatures


class ControlledConsumer(Consumer):
    """Consumer imposing an outlet temperature of 40°C through control logic"""

    _controlled_keys = Consumer._controlled_keys | {
        "setpoint_type_hx",
        "setpoint_value_hx",
    }

    def _run_control_logic(self, key, cp_fluid=CP_FLUID):
        if key == "setpoint_type_hx":
            return "t_out"
        if key == "setpoint_value_hx":
            return 40.0
        return super()._run_control_logic(key, cp_fluid)


class PartialSelectionTestCase(unittest.TestCase):
    def test_partial_selection_matches_full_network(self):
        """
        Computing only some edges gives the same results as computing all of
        them, with a soil that depends on the time step and a consumer
        controlled through its control logic.
        """
        net = star_network()
        with warnings.catch_warnings():
            # Replacing the component of an existing edge raises a warning
            warnings.simplefilter("ignore")
            net.add_component("SUB1", "S7", "R7", ControlledConsumer(name="SUB1"))
        fluid = ConstantWater()
        solve_hydraulics(net, fluid, verbose=0)
        hours = np.arange(8760)
        soil = KusudaSoil(t_air=10 + 10 * np.sin(2 * np.pi * hours / 8760))

        full = compute_edge_temperatures(net, fluid, soil, ts_id=5000)
        names = net.get_edges_attribute_array("name")
        mask = np.flatnonzero(np.isin(names, ["SP1", "SUB1"]))
        partial = compute_edge_temperatures(net, fluid, soil, mask=mask, ts_id=5000)

        self.assertEqual(full[1][names == "SUB1"], 40.0)
        for values_full, values_partial in zip(full, partial):
            np.testing.assert_allclose(values_partial[mask], values_full[mask])


if __name__ == "__main__":
    unittest.main()
