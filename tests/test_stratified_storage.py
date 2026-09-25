#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
Tests for the StratifiedStorage component. Where possible, results are
compared with analytical solutions or matrix exponentials of the layer
balances.
"""

import unittest
from itertools import product

import numpy as np
from scipy.linalg import expm

from pydhn.classes import Network
from pydhn.components import StratifiedStorage
from pydhn.components.stratified_storage_thermal import _mix_inversions
from pydhn.fluids import ConstantWater
from pydhn.fluids import Water
from pydhn.soils import Soil
from pydhn.solving import SimpleStep
from pydhn.solving import solve_thermal

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
        """An idle single layer tank decays exponentially to the ambient."""
        fluid = ConstantWater()
        volume, height, u, t_amb, dt = 2.0, 2.0, 1.5, 20.0, 600.0
        tank = make_tank(
            volume=volume, height=height, n_layers=1, u_value=u, t_ambient=t_amb
        )
        area = volume / height
        diameter = np.sqrt(4 * area / np.pi)
        ua = u * (np.pi * diameter * height + 2 * area)
        capacity = fluid.rho * volume * fluid.cp

        run_steps(tank, fluid, [0.0] * 20, [np.nan] * 20)
        expected = t_amb + (50.0 - t_amb) * np.exp(-ua * 20 * dt / capacity)
        self.assertAlmostEqual(tank._layer_temperatures[0], expected, delta=1e-9)

    def test_energy_conservation(self):
        """Without losses, delta_q matches the change of stored energy."""
        fluid = ConstantWater()
        tank = make_tank(u_value=0.0)
        capacity = fluid.rho * tank._layer_volume * fluid.cp
        # Large flows need many substeps (one layer is 0.2 m³)
        mdots = [0.1, 0.5, -0.4, 0.0, 1.5, -1.0]
        t_ins = [80.0, 75.0, 30.0, 40.0, 85.0, 25.0]

        u0 = capacity * tank._layer_temperatures.sum()
        outs = run_steps(tank, fluid, mdots, t_ins)
        u1 = capacity * tank._layer_temperatures.sum()
        exchanged = 3600.0 * sum(out[4] for out in outs)
        np.testing.assert_allclose(u1 - u0, -exchanged, rtol=1e-10)

    def test_stratification(self):
        """
        Charging half a cold tank gives hot water on top and cold water at the
        bottom, while the outlet stays cold.
        """
        fluid = ConstantWater()
        tank = make_tank(n_layers=20, u_value=0.0, temperature=20.0)
        # Charge 1 m³ in 5 steps
        mdot = 1.0 * fluid.rho / (5 * 600.0)
        outs = run_steps(tank, fluid, [mdot] * 5, [80.0] * 5)
        temps = tank._layer_temperatures
        self.assertTrue(np.all(np.diff(temps) <= 1e-12))
        self.assertGreater(temps[0], 75.0)
        self.assertLess(temps[-1], 25.0)
        self.assertLess(outs[0][1], 21.0)

    def test_mixing(self):
        """Inversions are mixed conserving energy."""
        zeros = np.zeros(3)
        stable = np.array([70.0, 60.0, 50.0])
        np.testing.assert_array_equal(_mix_inversions(stable, zeros)[0], stable)
        mixed = _mix_inversions(np.array([50.0, 70.0, 60.0]), zeros)[0]
        np.testing.assert_allclose(mixed, [60.0, 60.0, 60.0])

        inverted = np.linspace(20.0, 80.0, 10)
        tank = make_tank(u_value=0.0, initial_layer_temperatures=inverted)
        run_steps(tank, ConstantWater(), [0.0], [np.nan])
        temps = tank._layer_temperatures
        self.assertTrue(np.all(np.diff(temps) <= 1e-12))
        self.assertAlmostEqual(temps.mean(), inverted.mean(), places=10)

    def test_initial_inversions(self):
        """
        Inversions in the initial profile are mixed before the integration, so
        the results are the same as starting from the mixed profile.
        """
        profiles = [
            ([20.0, 80.0], [50.0, 50.0]),
            ([80.0, 20.0, 60.0, 30.0], [80.0, 40.0, 40.0, 30.0]),
        ]
        for (initial, mixed), fluid, dt, mdot in product(
            profiles,
            (ConstantWater(k=0.0), Water()),
            (1e-4, 1.0, 600.0),
            (0.1, -0.1, 0.0),
        ):
            with self.subTest(initial=initial, fluid=fluid, dt=dt, mdot=mdot):
                tank, reference = [
                    make_tank(
                        volume=10.0,
                        n_layers=len(profile),
                        stepsize=dt,
                        u_value=0.0,
                        initial_layer_temperatures=profile,
                    )
                    for profile in (initial, mixed)
                ]
                tank.set("mass_flow", mdot)
                reference.set("mass_flow", mdot)
                t_mean = np.mean(initial)
                capacity = fluid.get_rho(t_mean) * fluid.get_cp(t_mean)
                capacity *= tank._layer_volume
                # Repeating the step with another inlet starts again from the
                # initial profile
                for t_in in (80.0, 70.0):
                    outs = tank._compute_temperatures(fluid, SOIL, t_in, ts_id=0)
                    expected = reference._compute_temperatures(fluid, SOIL, t_in, 0)
                    np.testing.assert_array_equal(outs, expected)
                    np.testing.assert_array_equal(
                        tank._layer_temperatures, reference._layer_temperatures
                    )
                    np.testing.assert_array_equal(
                        tank._last_layer_temperatures, initial
                    )
                    stored = capacity * (tank._layer_temperatures - initial).sum()
                    self.assertAlmostEqual(stored + 3600.0 * outs[4], 0, delta=1e-5)

    def test_repeated_ts(self):
        """Repeating a time step gives the same result as running it once."""
        fluid = Water()
        tank_a, tank_b = make_tank(), make_tank()
        run_steps(tank_a, fluid, [0.5, 0.5, -0.3], [80.0, 70.0, 30.0], [0, 0, 1])
        run_steps(tank_b, fluid, [0.5, -0.3], [70.0, 30.0], [0, 1])
        np.testing.assert_array_equal(
            tank_a._layer_temperatures, tank_b._layer_temperatures
        )

    def test_initial_temperatures(self):
        tank = make_tank(temperature=80.0)
        np.testing.assert_array_equal(tank._layer_temperatures, np.full(10, 80.0))
        profile = np.linspace(80.0, 30.0, 10)
        tank = make_tank(temperature=10.0, initial_layer_temperatures=profile)
        # The profile is copied
        profile[:] = 0.0
        self.assertEqual(tank["temperature"], 55.0)
        np.testing.assert_array_equal(
            tank._layer_temperatures, np.linspace(80.0, 30.0, 10)
        )
        # Idle tanks exchange no heat with the network
        outs = tank._compute_temperatures(ConstantWater(), SOIL, np.nan, 0)
        self.assertEqual(outs[4], 0.0)

    def test_invalid_parameters(self):
        invalid = [
            ("volume", 0.0),
            ("height", -1.0),
            ("stepsize", 0.0),
            ("n_layers", 0),
            ("u_value", -1.0),
            ("delta_k", -1.0),
            ("setpoint_type_hyd", "pressure"),
        ]
        for key, value in invalid:
            with self.subTest(key=key, value=value):
                with self.assertRaises(ValueError):
                    make_tank(**{key: value})
                tank = make_tank()
                with self.assertRaises(ValueError):
                    tank.set(key, value)

    def test_geometry_updates(self):
        """Changing the geometry gives the same results as a new tank."""
        tank = make_tank(u_value=0.0)
        for key, value in [("volume", 4.0), ("height", 3.0), ("u_value", 2.0)]:
            tank.set(key, value)
        fresh = make_tank(volume=4.0, height=3.0, u_value=2.0)
        a = tank._compute_temperatures(ConstantWater(), SOIL, np.nan, 0)
        b = fresh._compute_temperatures(ConstantWater(), SOIL, np.nan, 0)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(
            tank._layer_temperatures, fresh._layer_temperatures
        )
        with self.assertRaises(ValueError):
            tank.set("n_layers", 5)
        # Setting the temperature does not change the layers
        before = tank._layer_temperatures.copy()
        tank.set("temperature", 90.0)
        np.testing.assert_array_equal(tank._layer_temperatures, before)

    def test_single_layer_with_flow_and_losses(self):
        """A single layer follows the analytical solution with flow and losses."""
        fluid = ConstantWater()
        for mdot in (0.5, -0.5, 3.0):
            with self.subTest(mdot=mdot):
                tank = make_tank(n_layers=1, u_value=1.5, t_ambient=20.0)
                tank.set("mass_flow", mdot)
                t_in, dt = 80.0, tank["stepsize"]
                area = tank["volume"] / tank["height"]
                ua = 1.5 * (2 * np.sqrt(np.pi * area) * tank["height"] + 2 * area)
                capacity = fluid.rho * tank["volume"] * fluid.cp
                adv = abs(mdot) * fluid.cp
                equilibrium = (adv * t_in + ua * 20.0) / (adv + ua)
                rate = (adv + ua) * dt / capacity
                # Time average of the exponential decay
                phi = -np.expm1(-rate) / rate
                expected = equilibrium + (50.0 - equilibrium) * np.exp(-rate)
                average = equilibrium + (50.0 - equilibrium) * phi
                outs = tank._compute_temperatures(fluid, SOIL, t_in, 0)
                self.assertAlmostEqual(outs[2], expected, delta=1e-3)
                self.assertAlmostEqual(outs[1], average, delta=1e-3)
                self.assertAlmostEqual(
                    outs[3], adv / (adv + ua) * (1 - phi), delta=3e-5
                )
                # With one layer, the average outlet temperature is also the
                # average temperature of the losses
                losses = ua * dt * (outs[1] - 20.0)
                balance = capacity * (outs[2] - 50.0) + 3600 * outs[4] + losses
                self.assertAlmostEqual(balance, 0.0, delta=1e-5)

    def test_conduction(self):
        """Conduction between idle layers matches the matrix exponential."""
        fluid = ConstantWater()
        initial = np.array([80.0, 65.0, 40.0, 20.0])
        tank = make_tank(
            n_layers=4, u_value=0.0, delta_k=500.0, initial_layer_temperatures=initial
        )
        capacity = fluid.rho * 2.0 / 4 * fluid.cp
        g = (fluid.k + 500.0) * (2.0 / 2.0) / (2.0 / 4)
        matrix = np.diag([-1.0, -2.0, -2.0, -1.0])
        matrix += np.diag(np.ones(3), 1) + np.diag(np.ones(3), -1)
        expected = expm(matrix * g * 600.0 / capacity) @ initial
        tank._compute_temperatures(fluid, SOIL, np.nan, 0)
        np.testing.assert_allclose(tank._layer_temperatures, expected, atol=1e-4)
        self.assertAlmostEqual(tank._layer_temperatures.sum(), initial.sum())

    def test_flow_with_conduction_and_losses(self):
        """
        Layers and average outlet temperature match the matrix exponential of
        the layer balances, extended with the ambient and inlet temperatures
        and with the integral of the outlet temperature.
        """
        fluid = ConstantWater()
        initial = np.array([80.0, 65.0, 40.0, 25.0])
        for mdot, t_in in ((0.5, 90.0), (-0.5, 10.0)):
            with self.subTest(mdot=mdot):
                tank = make_tank(
                    n_layers=4,
                    u_value=1.5,
                    delta_k=50.0,
                    initial_layer_temperatures=initial,
                )
                tank.set("mass_flow", mdot)
                capacity = fluid.rho * 0.5 * fluid.cp
                g = (fluid.k + 50.0) / 0.5
                ua = np.full(4, 1.5 * 2 * np.sqrt(np.pi) * 0.5)
                ua[[0, -1]] += 1.5
                adv = abs(mdot) * fluid.cp
                matrix = np.zeros((6, 6))
                for i in range(4):
                    matrix[i, i] = -(ua[i] + adv)
                    matrix[i, 4] = ua[i] * 20.0
                    for j in (i - 1, i + 1):
                        if 0 <= j < 4:
                            matrix[i, i] -= g
                            matrix[i, j] += g
                    upstream = i - 1 if mdot > 0 else i + 1
                    if 0 <= upstream < 4:
                        matrix[i, upstream] += adv
                    else:
                        matrix[i, 4] += adv * t_in
                matrix[:4] /= capacity
                matrix[5, 3 if mdot > 0 else 0] = 1.0
                exact = expm(600.0 * matrix) @ np.r_[initial, 1.0, 0.0]
                outs = tank._compute_temperatures(fluid, SOIL, t_in, 0)
                np.testing.assert_allclose(
                    tank._layer_temperatures, exact[:4], atol=1e-3
                )
                self.assertAlmostEqual(outs[1], exact[5] / 600.0, delta=1e-3)

    def test_large_flows(self):
        """
        From a partial to many turnovers of the tank volume in a step, the
        results match the matrix exponential of pure advection.
        """
        fluid = ConstantWater(k=0.0)
        n, dt, volume = 10, 3600.0, 2.0
        for mdot in (0.1, 1.0, 5.0, 50.0):
            with self.subTest(mdot=mdot):
                tank = make_tank(stepsize=dt, u_value=0.0)
                tank.set("mass_flow", mdot)
                rate = mdot * dt / (fluid.rho * volume / n)
                matrix = np.zeros((n + 2, n + 2))
                matrix[:n, :n] = -rate * np.eye(n) + rate * np.eye(n, k=-1)
                matrix[0, n] = rate * 80.0
                matrix[n + 1, n - 1] = 1.0
                propagator = expm(matrix)
                exact = propagator @ np.r_[np.full(n, 50.0), 1.0, 0.0]
                outs = tank._compute_temperatures(fluid, SOIL, 80.0, 0)
                np.testing.assert_allclose(
                    tank._layer_temperatures, exact[:n], atol=1e-3
                )
                self.assertAlmostEqual(outs[1], exact[-1], delta=1e-3)
                self.assertAlmostEqual(outs[3], propagator[-1, n] / 80.0, delta=3e-5)
                stored = fluid.rho * volume * fluid.cp * (outs[2] - 50.0)
                np.testing.assert_allclose(stored, -3600.0 * outs[4], rtol=1e-10)

    def test_outlet_derivative(self):
        """dT_out/dT_in matches finite differences, also when layers mix."""
        for fluid in (ConstantWater(), Water()):
            for mdot, t_in in ((0.5, 85.0), (-0.5, 20.0), (0.5, 10.0), (-0.5, 95.0)):
                with self.subTest(fluid=fluid, mdot=mdot, t_in=t_in):
                    tank = make_tank(
                        n_layers=4, initial_layer_temperatures=[80.0, 65.0, 40.0, 25.0]
                    )
                    tank.set("mass_flow", mdot)
                    center = tank._compute_temperatures(fluid, SOIL, t_in, 0)
                    plus = tank._compute_temperatures(fluid, SOIL, t_in + 1e-3, 0)
                    minus = tank._compute_temperatures(fluid, SOIL, t_in - 1e-3, 0)
                    self.assertAlmostEqual(
                        center[3], (plus[1] - minus[1]) / 2e-3, delta=1e-7
                    )

    def test_time_refinement_with_mixing(self):
        """
        With inlet temperatures that cause mixing, one step gives the same
        results as 200 shorter ones.
        """
        fluid = ConstantWater()
        for mdot, t_in in ((0.5, 10.0), (-0.5, 95.0)):
            with self.subTest(mdot=mdot):
                profiles, outlets = [], []
                for count in (1, 200):
                    tank = make_tank(
                        n_layers=4,
                        u_value=0.0,
                        stepsize=600.0 / count,
                        initial_layer_temperatures=[80.0, 65.0, 40.0, 25.0],
                    )
                    outs = run_steps(tank, fluid, [mdot] * count, [t_in] * count)
                    profiles.append(tank._layer_temperatures)
                    outlets.append(np.mean([out[1] for out in outs]))
                np.testing.assert_allclose(profiles[0], profiles[1], atol=0.005)
                self.assertAlmostEqual(outlets[0], outlets[1], delta=0.001)

    def test_recirculation(self):
        """A tank in a loop with a heater converges."""
        net = Network()
        for name in ("A", "B", "C"):
            net.add_node(name, temperature=50.0)
        net.add_stratified_storage(
            "T", "A", "B", volume=0.1, n_layers=1, mass_flow=1.0, stepsize=3600.0
        )
        net.add_pipe("P", "B", "C", length=0.01, diameter=0.1, mass_flow=1.0)
        net.add_producer(
            "H",
            "C",
            "A",
            mass_flow=1.0,
            setpoint_type_hx="delta_t",
            setpoint_value_hx=0.1,
        )
        results = solve_thermal(
            net, ConstantWater(), SOIL, ts_id=0, max_iters=5, verbose=0
        )
        self.assertTrue(results["history"]["thermal converged"])

    def test_network_simulation(self):
        """
        A tank next to a consumer charges and then discharges, with converged
        simulations and mass conservation.
        """
        net = Network()
        for name in ("N1", "N2", "N2b", "N3", "N3b", "N4"):
            net.add_node(name, x=0.0, y=0.0, z=0.0)
        net.add_pipe("SP1", "N1", "N2", length=50, diameter=0.05, line="supply")
        net.add_pipe("SP2", "N2", "N2b", length=10, diameter=0.05, line="supply")
        net.add_pipe("RP1", "N3b", "N3", length=10, diameter=0.05, line="return")
        net.add_pipe("RP2", "N3", "N4", length=50, diameter=0.05, line="return")
        net.add_consumer(
            name="C1",
            start_node="N2",
            end_node="N3",
            setpoint_type_hyd="mass_flow",
            setpoint_value_hyd=0.4,
            setpoint_type_hx="delta_q",
            setpoint_value_hx=-5000.0,
            control_type="mass_flow",
            stepsize=600.0,
        )
        net.add_producer(
            name="M",
            start_node="N4",
            end_node="N1",
            setpoint_type_hyd="pressure",
            setpoint_value_hyd=-1e5,
            setpoint_type_hx="t_out",
            setpoint_value_hx=80.0,
            stepsize=600.0,
        )
        net.add_stratified_storage(
            name="T",
            start_node="N2b",
            end_node="N3b",
            volume=2.0,
            n_layers=10,
            setpoint_value_hyd=0.1,
            stepsize=600.0,
        )
        loop = SimpleStep(
            hydraulic_sim_kwargs={"error_threshold": 1, "verbose": 0},
            thermal_sim_kwargs={"error_threshold": 1e-6, "verbose": 0},
            with_thermal=True,
        )
        tank = net[("N2b", "N3b")]

        def run(steps):
            for ts_id in steps:
                results = loop.execute(net=net, fluid=Water(), soil=SOIL, ts_id=ts_id)
                self.assertTrue(results["history"]["hydraulics converged"])
                self.assertTrue(results["history"]["thermal converged"])
                _, mdot = net.edges("mass_flow")
                np.testing.assert_allclose(net.incidence_matrix @ mdot, 0, atol=1e-6)

        # Charging: the top of the tank approaches the supply temperature
        run(range(5))
        self.assertGreater(tank._layer_temperatures[0], 70.0)
        self.assertAlmostEqual(tank["mass_flow"], 0.1)

        # Discharging: the tank cools down
        t_mean = tank._layer_temperatures.mean()
        tank.set("setpoint_value_hyd", -0.1)
        run(range(5, 10))
        self.assertAlmostEqual(tank["mass_flow"], -0.1)
        self.assertLess(tank._layer_temperatures.mean(), t_mean)


if __name__ == "__main__":
    unittest.main()
