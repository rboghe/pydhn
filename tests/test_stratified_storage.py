#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
Tests for the StratifiedStorage component: standby decay against the exact
backward-Euler recurrence with an independently computed UA, exact energy
conservation, stratification and plug-flow behaviour, buoyant mixing,
repeated time steps, and a mixed steady-state/dynamic network simulation.
"""

import unittest

import numpy as np

from pydhn.classes import Network
from pydhn.components import StratifiedStorage
from pydhn.fluids import ConstantWater
from pydhn.fluids import Water
from pydhn.soils import Soil

SOIL = Soil(temp=8)


def make_tank(**kwargs):
    defaults = dict(volume=2.0, height=2.0, n_layers=10, stepsize=600.0)
    tank = StratifiedStorage(**{**defaults, **kwargs})
    tank.set("mass_flow", 0.0)
    return tank


def run_steps(tank, fluid, mdots, t_ins, ts_ids=None):
    outs = []
    for k, (mdot, t_in) in enumerate(zip(mdots, t_ins)):
        tank.set("mass_flow", mdot)
        ts = ts_ids[k] if ts_ids is not None else k
        outs.append(tank._compute_temperatures(fluid, SOIL, t_in=t_in, ts_id=ts))
    return outs


class StratifiedStorageTestCase(unittest.TestCase):
    def test_standby_decay(self):
        """
        A single-layer idle tank must follow the backward-Euler recurrence
        of the fully-mixed tank, with UA computed independently from the
        geometry, and decay monotonically towards the ambient temperature.
        """
        fluid = ConstantWater()
        volume, height, u, t_amb, dt = 2.0, 2.0, 1.5, 20.0, 600.0
        tank = make_tank(
            volume=volume, height=height, n_layers=1, u_value=u, t_ambient=t_amb
        )
        area = volume / height
        diameter = np.sqrt(4 * area / np.pi)
        ua = u * (np.pi * diameter * height + 2 * area)
        capacity = fluid.rho * volume * fluid.cp

        temps = [tank._layer_temperatures[0]]  # starts at 50 °C
        run_steps(tank, fluid, [0.0] * 20, [np.nan] * 20)
        expected = temps[0]
        for _ in range(20):
            expected = (capacity / dt * expected + ua * t_amb) / (capacity / dt + ua)
        np.testing.assert_allclose(tank._layer_temperatures[0], expected, rtol=1e-12)
        self.assertGreater(tank._layer_temperatures[0], t_amb)

    def test_energy_conservation(self):
        """
        With a lossless envelope, the reported delta_q must match the change
        of internal energy of the tank exactly, including sub-stepped steps.
        """
        fluid = ConstantWater()
        tank = make_tank(u_value=0.0)
        capacity = fluid.rho * tank._layer_volume * fluid.cp
        # Large flows trigger sub-stepping (one layer is 0.2 m³)
        mdots = [0.1, 0.5, -0.4, 0.0, 1.5, -1.0]
        t_ins = [80.0, 75.0, 30.0, 40.0, 85.0, 25.0]

        u0 = capacity * tank._layer_temperatures.sum()
        outs = run_steps(tank, fluid, mdots, t_ins)
        u1 = capacity * tank._layer_temperatures.sum()

        exchanged = 3600.0 * sum(dq for *_, dq in outs)  # J, negative = charge
        np.testing.assert_allclose(u1 - u0, -exchanged, rtol=1e-10)

    def test_stratification_and_plug_flow(self):
        """
        Charging half a cold tank with hot water must give a stratified,
        non-increasing profile with hot top and cold bottom, while the
        outlet stays cold until the front arrives.
        """
        fluid = ConstantWater()
        tank = make_tank(volume=2.0, n_layers=20, u_value=0.0)
        tank._layer_temperatures[:] = 20.0
        tank._last_layer_temperatures[:] = 20.0

        # Charge half the tank volume: 1 m³ over 5 steps of 600 s
        mdot = 1.0 * fluid.rho / (5 * 600.0)
        outs = run_steps(tank, fluid, [mdot] * 5, [80.0] * 5)

        temps = tank._layer_temperatures
        self.assertTrue(np.all(np.diff(temps) <= 1e-12))  # non-increasing
        self.assertGreater(temps[0], 75.0)
        self.assertLess(temps[-1], 25.0)
        # Cold water is expelled first
        self.assertLess(outs[0][1], 21.0)

    def test_buoyancy_mixing(self):
        """
        A temperature inversion must be mixed away conserving energy, and
        the helper must leave stable profiles untouched.
        """
        stable = np.array([70.0, 60.0, 50.0])
        np.testing.assert_array_equal(
            StratifiedStorage._mix_inversions(stable), stable
        )
        mixed = StratifiedStorage._mix_inversions(np.array([50.0, 70.0, 60.0]))
        np.testing.assert_allclose(mixed, [60.0, 60.0, 60.0])

        fluid = ConstantWater()
        tank = make_tank(u_value=0.0)
        inverted = np.linspace(20.0, 80.0, 10)  # cold on top
        tank._layer_temperatures = inverted.copy()
        run_steps(tank, fluid, [0.0], [np.nan])
        temps = tank._layer_temperatures
        self.assertTrue(np.all(np.diff(temps) <= 1e-12))
        np.testing.assert_allclose(temps.mean(), inverted.mean(), rtol=1e-12)

    def test_repeated_ts_restores_state(self):
        """
        Calling the same ts_id twice (Newton iterations) must give the same
        final state as calling it once.
        """
        fluid = Water()
        tank_a, tank_b = make_tank(), make_tank()
        run_steps(tank_a, fluid, [0.5, 0.5, -0.3], [80.0, 70.0, 30.0], [0, 0, 1])
        run_steps(tank_b, fluid, [0.5, -0.3], [70.0, 30.0], [0, 1])
        np.testing.assert_array_equal(
            tank_a._layer_temperatures, tank_b._layer_temperatures
        )

    def test_network_simulation(self):
        """
        A sandwich network with steady-state pipes, a consumer and a storage
        charging then discharging: both solvers converge, mass is conserved
        and the tank state responds accordingly.
        """
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

        loop = SimpleStep(
            hydraulic_sim_kwargs={"error_threshold": 1, "verbose": 0},
            thermal_sim_kwargs={"error_threshold": 1e-6, "verbose": 0},
            with_thermal=True,
        )
        fluid, tank = Water(), net[("N2b", "N3b")]

        def steps(n, start):
            for k in range(start, start + n):
                res = loop.execute(net=net, fluid=fluid, soil=SOIL, ts_id=k)
                self.assertTrue(res["history"]["hydraulics converged"])
                self.assertTrue(res["history"]["thermal converged"])
                _, mdot = net.edges("mass_flow")
                np.testing.assert_allclose(
                    net.incidence_matrix @ mdot, 0.0, atol=1e-6
                )

        # Charging: the top of the tank approaches the supply temperature
        steps(5, 0)
        self.assertGreater(tank._layer_temperatures[0], 70.0)
        self.assertAlmostEqual(net[("N2b", "N3b")]["mass_flow"], 0.1)

        # Discharging: the tank drains into the return line and cools down
        mean_before = tank._layer_temperatures.mean()
        net.set_edge_attributes([-0.1], "setpoint_value_hyd", mask=np.array(
            [i for i, (u, v) in enumerate(net.edges()) if (u, v) == ("N2b", "N3b")]
        ))
        steps(5, 5)
        self.assertAlmostEqual(net[("N2b", "N3b")]["mass_flow"], -0.1)
        self.assertLess(tank._layer_temperatures.mean(), mean_before)


if __name__ == "__main__":
    unittest.main()
