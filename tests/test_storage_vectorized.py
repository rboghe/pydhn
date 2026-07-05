#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
Equivalence tests between the vectorized storage thermal model
(pydhn.components.stratified_storage_thermal) and the tank-by-tank
reference implementation (StratifiedStorage._compute_temperatures).

Since the batched Thomas sweeps and per-tank scalars replicate the scalar
arithmetic operation by operation, results are required to be bitwise
identical, not merely close.
"""

import unittest
from contextlib import contextmanager
from copy import deepcopy

import numpy as np

from pydhn.classes import Network
from pydhn.components.vector_functions import COMPONENT_FUNCTIONS_DICT
from pydhn.fluids import Water
from pydhn.soils import Soil
from pydhn.solving.temperature import compute_edge_temperatures

SOIL = Soil(temp=8)
STEPS = 12


@contextmanager
def scalar_fallback():
    """Temporarily unregister the vectorized thermal function."""
    entry = COMPONENT_FUNCTIONS_DICT["stratified_storage"]
    COMPONENT_FUNCTIONS_DICT["stratified_storage"] = None
    try:
        yield
    finally:
        COMPONENT_FUNCTIONS_DICT["stratified_storage"] = entry


def build_tanks_net():
    """Four heterogeneous tanks as edges of a bare network."""
    net = Network()
    specs = [
        dict(volume=0.5, height=1.0, n_layers=1, u_value=1.0),
        dict(volume=2.0, height=2.0, n_layers=5, u_value=0.0, stepsize=300.0),
        dict(volume=10.0, height=3.0, n_layers=10, delta_k=0.5),
        dict(volume=1.0, height=2.5, n_layers=17, u_value=2.0, t_ambient=5.0),
    ]
    for i, spec in enumerate(specs):
        net.add_node(f"A{i}", x=0.0, y=0.0, z=0.0)
        net.add_node(f"B{i}", x=1.0, y=0.0, z=0.0)
        net.add_stratified_storage(f"T{i}", f"A{i}", f"B{i}", **spec)
    return net


def drive(net, fluid, mass_flows, node_temps, ts_ids):
    """Impose flows and node temperatures, then step the thermal models."""
    records = []
    for mdot, t_nodes, ts in zip(mass_flows, node_temps, ts_ids):
        net.set_edge_attributes(mdot, "mass_flow")
        net.set_node_attributes(t_nodes, "temperature")
        outs = compute_edge_temperatures(net, fluid, SOIL, ts_id=ts)
        states = [
            net[(u, v)]._layer_temperatures.copy() for u, v in net.edges()
        ]
        records.append((outs, states))
    return records


class StorageVectorEquivalence(unittest.TestCase):
    def assert_equivalent(self, mass_flows, ts_ids):
        rng = np.random.default_rng(1)
        node_temps = [rng.uniform(25, 90, 8) for _ in mass_flows]
        net_v, net_s = build_tanks_net(), build_tanks_net()
        rec_v = drive(net_v, Water(), mass_flows, node_temps, ts_ids)
        with scalar_fallback():
            rec_s = drive(net_s, Water(), mass_flows, node_temps, ts_ids)

        names = ["t_in", "t_out", "t_avg", "t_out_der", "delta_q"]
        for k, ((outs_v, st_v), (outs_s, st_s)) in enumerate(zip(rec_v, rec_s)):
            for name, a, b in zip(names, outs_v, outs_s):
                np.testing.assert_array_equal(
                    a, b, err_msg=f"step {k}: {name} differs"
                )
            for s, (a, b) in enumerate(zip(st_v, st_s)):
                np.testing.assert_array_equal(
                    a, b, err_msg=f"step {k} tank {s}: layers differ"
                )

    def test_mixed_flows(self):
        # Charge, discharge, idle and inversion-triggering cold charges; the
        # large flows force different sub-step counts per tank
        rng = np.random.default_rng(2)
        mass_flows = [
            rng.uniform(-1.0, 1.0, 4) * (rng.random(4) > 0.25) for _ in range(STEPS)
        ]
        self.assert_equivalent(mass_flows, range(STEPS))

    def test_repeated_ts(self):
        rng = np.random.default_rng(3)
        mass_flows = [rng.uniform(-0.5, 0.5, 4) for _ in range(STEPS)]
        ts_ids = np.repeat(np.arange(STEPS // 2), 2)
        self.assert_equivalent(mass_flows, ts_ids)

    def test_full_simulation(self):
        # The sandwich network of test_stratified_storage, simulated with
        # both paths: results must stay bitwise identical through the
        # network Newton feedback
        from pydhn.solving import SimpleStep

        net = Network()
        for name in ("N1", "N2", "N2b", "N3", "N3b", "N4"):
            net.add_node(name, x=0.0, y=0.0, z=0.0)
        net.add_pipe("SP1", "N1", "N2", length=50, diameter=0.05, line="supply")
        net.add_pipe("SP2", "N2", "N2b", length=10, diameter=0.05, line="supply")
        net.add_pipe("RP1", "N3b", "N3", length=10, diameter=0.05, line="return")
        net.add_pipe("RP2", "N3", "N4", length=50, diameter=0.05, line="return")
        net.add_consumer(
            name="C1", start_node="N2", end_node="N3",
            setpoint_type_hyd="mass_flow", setpoint_value_hyd=0.4,
            setpoint_type_hx="delta_q", setpoint_value_hx=-5000.0,
            control_type="mass_flow", stepsize=600.0,
        )
        net.add_producer(
            name="M", start_node="N4", end_node="N1",
            setpoint_type_hyd="pressure", setpoint_value_hyd=-1e5,
            setpoint_type_hx="t_out", setpoint_value_hx=80.0, stepsize=600.0,
        )
        net.add_stratified_storage(
            name="T", start_node="N2b", end_node="N3b",
            volume=2.0, n_layers=10, setpoint_value_hyd=0.1, stepsize=600.0,
        )

        def run(net):
            loop = SimpleStep(
                hydraulic_sim_kwargs={"error_threshold": 1, "verbose": 0},
                thermal_sim_kwargs={"error_threshold": 1e-6, "verbose": 0},
                with_thermal=True,
            )
            fluid = Water()
            out = []
            for k in range(6):
                if k == 3:  # switch from charging to discharging
                    net[("N2b", "N3b")].set("setpoint_value_hyd", -0.1)
                res = loop.execute(net=net, fluid=fluid, soil=SOIL, ts_id=k)
                out.append(res)
            return out, net[("N2b", "N3b")]._layer_temperatures.copy()

        net_v, net_s = deepcopy(net), deepcopy(net)
        res_v, layers_v = run(net_v)
        with scalar_fallback():
            res_s, layers_s = run(net_s)

        np.testing.assert_array_equal(layers_v, layers_s)
        for k, (rv, rs) in enumerate(zip(res_v, res_s)):
            for key in ("outlet_temperature", "temperature", "delta_q"):
                np.testing.assert_array_equal(
                    rv["edges"][key], rs["edges"][key],
                    err_msg=f"step {k}: {key} differs",
                )


if __name__ == "__main__":
    unittest.main()
