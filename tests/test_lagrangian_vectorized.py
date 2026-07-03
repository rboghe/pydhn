#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
Equivalence tests between the vectorized lagrangian thermal model
(pydhn.components.lagrangian_thermal) and the pipe-by-pipe reference
implementation (LagrangianPipe._compute_temperatures).

Both paths are driven through compute_edge_temperatures() on identical
network copies over multiple time steps, covering positive, negative, zero
and overshooting mass flows, exact-fill volumes, and repeated time steps
(Newton iterations). Outputs and the full internal parcel state must match.
"""

import unittest
from contextlib import contextmanager
from copy import deepcopy

import numpy as np

from pydhn.classes import Network
from pydhn.components.vector_functions import COMPONENT_FUNCTIONS_DICT
from pydhn.fluids import ConstantWater
from pydhn.fluids import Water
from pydhn.soils import Soil
from pydhn.solving.temperature import compute_edge_temperatures

RTOL, ATOL = 1e-9, 1e-12
STEPSIZE = 60.0


@contextmanager
def scalar_fallback():
    """Temporarily unregister the vectorized thermal function."""
    entry = COMPONENT_FUNCTIONS_DICT["lagrangian_pipe"]
    COMPONENT_FUNCTIONS_DICT["lagrangian_pipe"] = {"delta_p": entry["delta_p"]}
    try:
        yield
    finally:
        COMPONENT_FUNCTIONS_DICT["lagrangian_pipe"] = entry


def build_chain(n_pipes=5, stepsize=STEPSIZE):
    """A chain of heterogeneous lagrangian pipes."""
    rng = np.random.default_rng(42)
    net = Network()
    for i in range(n_pipes + 1):
        net.add_node(f"N{i}", x=float(i), y=0.0, z=0.0)
    for i in range(n_pipes):
        net.add_lagrangian_pipe(
            name=f"P{i}",
            start_node=f"N{i}",
            end_node=f"N{i + 1}",
            length=rng.uniform(20, 200),
            diameter=rng.uniform(0.02, 0.2),
            insulation_thickness=rng.uniform(0.01, 0.05),
            k_insulation=rng.uniform(0.02, 0.05),
            internal_pipe_thickness=rng.uniform(0.002, 0.01),
            casing_thickness=rng.uniform(0.002, 0.01),
            depth=rng.uniform(0.5, 1.5),
            stepsize=stepsize,
        )
    return net


def drive(net, fluid, soil, mass_flows, node_temps, ts_ids):
    """
    Run compute_edge_temperatures for each step with imposed mass flows and
    node temperatures, returning outputs and internal states of every step.
    """
    records = []
    for mdot, t_nodes, ts in zip(mass_flows, node_temps, ts_ids):
        net.set_edge_attributes(mdot, "mass_flow")
        net.set_node_attributes(t_nodes, "temperature")
        # Keep reynolds and friction_factor in sync as the solver would
        mask = net.mask(attr="component_type", value="lagrangian_pipe")
        COMPONENT_FUNCTIONS_DICT["lagrangian_pipe"]["delta_p"](
            net, fluid, mask=mask
        )
        outs = compute_edge_temperatures(net, fluid, soil, ts_id=ts)
        states = [
            (c._volumes.copy(), c._temperatures.copy(), c._wall_temperatures.copy())
            for c in (net[(u, v)] for u, v in net.edges()[mask])
        ]
        records.append((outs, states))
    return records


class LagrangianVectorEquivalence(unittest.TestCase):
    def assert_equivalent(self, net, fluid, soil, mass_flows, node_temps, ts_ids):
        net_v, net_s = deepcopy(net), deepcopy(net)
        rec_v = drive(net_v, fluid, soil, mass_flows, node_temps, ts_ids)
        with scalar_fallback():
            rec_s = drive(net_s, fluid, soil, mass_flows, node_temps, ts_ids)

        names = ["t_in", "t_out", "t_avg", "t_out_der", "delta_q"]
        for k, ((outs_v, states_v), (outs_s, states_s)) in enumerate(
            zip(rec_v, rec_s)
        ):
            for name, a, b in zip(names, outs_v, outs_s):
                np.testing.assert_allclose(
                    a, b, rtol=RTOL, atol=ATOL,
                    err_msg=f"step {k}: {name} differs",
                )
            for p, (sv, ss) in enumerate(zip(states_v, states_s)):
                for name, a, b in zip(["volumes", "temps", "walls"], sv, ss):
                    self.assertEqual(
                        len(a), len(b), f"step {k} pipe {p}: {name} length"
                    )
                    np.testing.assert_allclose(
                        a, b, rtol=RTOL, atol=ATOL,
                        err_msg=f"step {k} pipe {p}: {name} differ",
                    )

    def _random_walk_temps(self, net, steps, seed, lo=40, hi=80):
        rng = np.random.default_rng(seed)
        names, _ = net.nodes(data="temperature")
        n = len(names)
        return [rng.uniform(lo, hi, n) for _ in range(steps)]

    def test_positive_flow(self):
        net = build_chain()
        rng = np.random.default_rng(0)
        mdots = [rng.uniform(0.05, 2.0, 5) for _ in range(8)]
        temps = self._random_walk_temps(net, 8, 1)
        self.assert_equivalent(net, Water(), Soil(temp=8), mdots, temps, range(8))

    def test_negative_flow(self):
        net = build_chain()
        rng = np.random.default_rng(2)
        mdots = [-rng.uniform(0.05, 2.0, 5) for _ in range(8)]
        temps = self._random_walk_temps(net, 8, 3)
        self.assert_equivalent(net, Water(), Soil(temp=8), mdots, temps, range(8))

    def test_mixed_and_alternating_flow(self):
        net = build_chain()
        rng = np.random.default_rng(4)
        mdots = [rng.uniform(-1.0, 1.0, 5) for _ in range(12)]
        temps = self._random_walk_temps(net, 12, 5)
        self.assert_equivalent(net, Water(), Soil(temp=8), mdots, temps, range(12))

    def test_zero_flow(self):
        net = build_chain()
        rng = np.random.default_rng(6)
        # Some pipes still, some flowing, pattern changing over time
        mdots = [rng.uniform(0.1, 1.0, 5) * (rng.random(5) > 0.5) for _ in range(8)]
        temps = self._random_walk_temps(net, 8, 7)
        self.assert_equivalent(net, Water(), Soil(temp=8), mdots, temps, range(8))

    def test_overshoot(self):
        # Mass flows large enough to flush more than one pipe volume per step
        net = build_chain()
        volumes = np.array([net[(u, v)]._internal_volume for u, v in net.edges()])
        mdots = [volumes * 1000.0 / STEPSIZE * f for f in (1.5, 3.0, 0.2, 2.0)]
        temps = self._random_walk_temps(net, 4, 8)
        self.assert_equivalent(net, Water(), Soil(temp=8), mdots, temps, range(4))

    def test_exact_fill(self):
        # With a constant-property fluid the inlet parcel can exactly match
        # the pipe volume, hitting the dedicated branch of the scalar model
        net = build_chain()
        fluid = ConstantWater()
        volumes = np.array([net[(u, v)]._internal_volume for u, v in net.edges()])
        exact = volumes * fluid.rho / STEPSIZE
        assert np.all(
            np.abs(exact * STEPSIZE / fluid.rho) == volumes
        ), "test setup: fill not exact"
        temps = self._random_walk_temps(net, 3, 9)
        self.assert_equivalent(
            net, fluid, Soil(temp=8), [exact, exact, exact], temps, range(3)
        )

    def test_repeated_ts(self):
        # The same ts_id repeated (Newton iterations) must restore the state
        net = build_chain()
        rng = np.random.default_rng(10)
        mdots = [rng.uniform(0.05, 1.0, 5) for _ in range(6)]
        temps = self._random_walk_temps(net, 6, 11)
        ts_ids = [0, 0, 0, 1, 1, 2]
        self.assert_equivalent(net, Water(), Soil(temp=8), mdots, temps, ts_ids)

    def test_full_simulation(self):
        # End-to-end SimpleStep run on a looped net with producer and consumers
        from pydhn.solving import SimpleStep

        net = Network()
        for name, x, y in [
            ("S0", 0, 0), ("S1", 1, 0), ("S2", 2, 0),
            ("R0", 0, 1), ("R1", 1, 1), ("R2", 2, 1),
        ]:
            net.add_node(name, x=float(x), y=float(y), z=0.0)
        for i in range(2):
            net.add_lagrangian_pipe(
                name=f"SP{i}", start_node=f"S{i}", end_node=f"S{i + 1}",
                length=100.0, diameter=0.05, line="supply", stepsize=STEPSIZE,
            )
            net.add_lagrangian_pipe(
                name=f"RP{i}", start_node=f"R{i + 1}", end_node=f"R{i}",
                length=100.0, diameter=0.05, line="return", stepsize=STEPSIZE,
            )
        net.add_producer(
            name="main", start_node="R0", end_node="S0",
            setpoint_type_hyd="pressure", setpoint_value_hyd=-1e5,
            setpoint_type_hx="t_out", setpoint_value_hx=75.0,
            stepsize=STEPSIZE,
        )
        for i in (1, 2):
            net.add_consumer(
                name=f"C{i}", start_node=f"S{i}", end_node=f"R{i}",
                setpoint_type_hyd="mass_flow", setpoint_value_hyd=0.3,
                setpoint_type_hx="delta_q", setpoint_value_hx=-2.0,
                control_type="mass_flow", stepsize=STEPSIZE,
            )

        def run(net):
            loop = SimpleStep(
                hydraulic_sim_kwargs={"error_threshold": 1, "verbose": 0},
                thermal_sim_kwargs={"error_threshold": 1e-9, "verbose": 0},
                with_thermal=True,
            )
            fluid, soil = Water(), Soil(temp=5)
            return [
                loop.execute(net=net, fluid=fluid, soil=soil, ts_id=k)
                for k in range(5)
            ]

        net_v, net_s = deepcopy(net), deepcopy(net)
        res_v = run(net_v)
        with scalar_fallback():
            res_s = run(net_s)
        for k, (rv, rs) in enumerate(zip(res_v, res_s)):
            for key in ("outlet_temperature", "temperature", "delta_q"):
                np.testing.assert_allclose(
                    rv["edges"][key], rs["edges"][key], rtol=1e-8, atol=1e-8,
                    err_msg=f"step {k}: {key} differs",
                )


if __name__ == "__main__":
    unittest.main()
