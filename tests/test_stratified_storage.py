#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
Tests for the StratifiedStorage component: standby decay against the exact
analytical solution with an independently computed UA, exact energy
conservation, stratification and plug-flow behaviour, buoyant mixing,
repeated time steps, and a mixed steady-state/dynamic network simulation.
"""

import unittest
from copy import deepcopy
from warnings import catch_warnings
from warnings import filterwarnings

import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm

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
        A single-layer idle tank must follow the analytical exponential decay
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
        expected = t_amb + (temps[0] - t_amb) * np.exp(-ua * 20 * dt / capacity)
        np.testing.assert_allclose(
            tank._layer_temperatures[0], expected, atol=1e-9, rtol=0
        )
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

    def test_water_property_freezing_energy_diagnostic(self):
        """Quantify stepwise freezing against an integrated heat-capacity inventory.

        The diagnostic uses e(T) = integral(rho(T) * cp(T), dT) per unit volume.
        It tests the approximation, not a compressible or enthalpy-flow model.
        """
        fluid = Water()
        tank = make_tank(u_value=0.0, temperature=50.0)
        exchanged = throughput = 0.0
        for step, mdot in enumerate([0.1] * 10 + [-0.1] * 10):
            before = tank._layer_temperatures.copy()
            capacity = (
                fluid.get_rho(before.mean())
                * fluid.get_cp(before.mean())
                * tank._layer_volume
            )
            tank.set("mass_flow", mdot)
            outs = tank._compute_temperatures(
                fluid, SOIL, 80.0 if mdot > 0 else 30.0, step
            )
            q = outs[4] * 3600.0
            # Each individual step conserves its own frozen-property energy.
            balance = capacity * (tank._layer_temperatures - before).sum() + q
            self.assertAlmostEqual(balance, 0.0, delta=1e-5)
            exchanged += q
            throughput += abs(q)

        # A single nonlinear energy inventory need not close across those steps.
        stored = tank._layer_volume * sum(
            quad(lambda t: float(fluid.get_rho(t) * fluid.get_cp(t)), 50.0, t)[0]
            for t in tank._layer_temperatures
        )
        relative_residual = (stored + exchanged) / throughput
        self.assertAlmostEqual(relative_residual, -0.001067, delta=1e-5)

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

    def test_initial_temperature_and_profile(self):
        tank = make_tank(temperature=80.0)
        np.testing.assert_array_equal(tank._layer_temperatures, np.full(10, 80.0))
        profile = np.linspace(80.0, 30.0, 10)
        tank = make_tank(temperature=10.0, initial_layer_temperatures=profile)
        profile[:] = 0.0
        self.assertEqual(tank["temperature"], 55.0)
        np.testing.assert_array_equal(
            tank._layer_temperatures, np.linspace(80.0, 30.0, 10)
        )
        self.assertEqual(
            tank._compute_temperatures(ConstantWater(), SOIL, np.nan, 0)[4], 0.0
        )

    def test_invalid_parameters_and_updates(self):
        invalid = [
            ("volume", 0.0),
            ("volume", -1.0),
            ("height", 0.0),
            ("stepsize", 0.0),
            ("stepsize", -1.0),
            ("stepsize", np.inf),
            ("n_layers", 0),
            ("n_layers", 2.5),
            ("n_layers", True),
            ("u_value", -1.0),
            ("delta_k", -1.0),
            ("t_ambient", np.nan),
            ("mass_flow", np.nan),
            ("temperature", np.inf),
            ("setpoint_type_hyd", "pressure"),
        ]
        for key, value in invalid:
            with self.subTest(key=key, value=value):
                with self.assertRaises(ValueError):
                    make_tank(**{key: value})
                tank = make_tank()
                before = tank[key]
                with self.assertRaises(ValueError):
                    tank.set(key, value)
                self.assertEqual(tank[key], before)
        for profile in ([50.0], np.full(10, np.nan), np.ones((10, 1))):
            with self.assertRaises(ValueError):
                make_tank(initial_layer_temperatures=profile)

    def test_geometry_updates(self):
        tank = make_tank(u_value=0.0)
        for key, value in [("volume", 4.0), ("height", 3.0), ("u_value", 2.0)]:
            tank.set(key, value)
        fresh = make_tank(volume=4.0, height=3.0, u_value=2.0)
        fluid = ConstantWater()
        a = tank._compute_temperatures(fluid, SOIL, np.nan, 0)
        b = fresh._compute_temperatures(fluid, SOIL, np.nan, 0)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(
            tank._layer_temperatures, fresh._layer_temperatures
        )
        self.assertLess(a[2], 50.0)
        with self.assertRaisesRegex(ValueError, "construct a new tank"):
            tank.set("n_layers", 5)
        tank.set("n_layers", 10)
        # Network reporting must never reset the internal temperature profile.
        before = tank._layer_temperatures.copy()
        tank.set("temperature", 90.0)
        np.testing.assert_array_equal(tank._layer_temperatures, before)

    def test_single_layer_analytical_flow_and_losses(self):
        fluid = ConstantWater()
        for mdot in (0.5, -0.5, 3.0):
            with self.subTest(mdot=mdot):
                tank = make_tank(n_layers=1, u_value=1.5, t_ambient=20.0)
                tank.set("mass_flow", mdot)
                tin, dt = 80.0, tank["stepsize"]
                # Independently compute the full cylindrical envelope.
                area = tank["volume"] / tank["height"]
                ua = 1.5 * (2 * np.sqrt(np.pi * area) * tank["height"] + 2 * area)
                capacity = fluid.rho * tank["volume"] * fluid.cp
                adv = abs(mdot) * fluid.cp
                equilibrium = (adv * tin + ua * 20.0) / (adv + ua)
                rate = (adv + ua) * dt / capacity
                phi = -np.expm1(-rate) / rate
                expected = equilibrium + (50.0 - equilibrium) * np.exp(-rate)
                average = equilibrium + (50.0 - equilibrium) * phi
                outs = tank._compute_temperatures(fluid, SOIL, tin, 0)
                # The 0.1 time-constant limit targets sub-millikelvin accuracy.
                self.assertAlmostEqual(outs[2], expected, delta=1e-3)
                self.assertAlmostEqual(outs[1], average, delta=1e-3)
                self.assertAlmostEqual(
                    outs[3], adv / (adv + ua) * (1 - phi), delta=3e-5
                )
                # For ONE layer, the time-averaged outlet (outs[1]) also gives
                # its time-averaged loss temperature. End-state t_avg does not.
                balance = (
                    capacity * (outs[2] - 50.0)
                    + 3600 * outs[4]
                    + ua * dt * (outs[1] - 20.0)
                )
                self.assertAlmostEqual(balance, 0.0, delta=1e-5)
                endpoint_balance = balance + ua * dt * (outs[2] - outs[1])
                self.assertGreater(abs(endpoint_balance), 1000.0)

    def test_conduction_against_matrix_exponential(self):
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
        np.testing.assert_allclose(
            tank._layer_temperatures, expected, atol=1e-4, rtol=0
        )
        self.assertAlmostEqual(tank._layer_temperatures.sum(), initial.sum(), places=10)

    def test_multilayer_flow_against_matrix_exponential(self):
        fluid = ConstantWater()
        initial = np.array([80.0, 65.0, 40.0, 25.0])
        for mdot, tin in ((0.5, 90.0), (-0.5, 10.0)):
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
                # Augment the ODE with a constant source and outlet integral.
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
                        matrix[i, 4] += adv * tin
                matrix[:4] /= capacity
                matrix[5, 3 if mdot > 0 else 0] = 1.0
                exact = expm(600.0 * matrix) @ np.r_[initial, 1.0, 0.0]
                outs = tank._compute_temperatures(fluid, SOIL, tin, 0)
                np.testing.assert_allclose(
                    tank._layer_temperatures, exact[:4], atol=1e-3, rtol=0
                )
                self.assertAlmostEqual(outs[1], exact[5] / 600.0, delta=1e-3)

    def test_large_turnovers_against_matrix_exponential(self):
        """Resolve hourly charging from partial turnover to near-equilibrium flow."""
        fluid = ConstantWater(k=0.0)
        n, dt, volume = 10, 3600.0, 2.0
        for mdot in (0.1, 1.0, 5.0, 50.0):
            with self.subTest(mdot=mdot):
                tank = make_tank(stepsize=dt, u_value=0.0)
                tank.set("mass_flow", mdot)
                rate = mdot * dt / (fluid.rho * volume / n)
                # Exact advection with a constant inlet and an outlet integral.
                matrix = np.zeros((n + 2, n + 2))
                matrix[:n, :n] = -rate * np.eye(n) + rate * np.eye(n, k=-1)
                matrix[0, n] = rate * 80.0
                matrix[n + 1, n - 1] = 1.0
                propagator = expm(matrix)
                exact = propagator @ np.r_[np.full(n, 50.0), 1.0, 0.0]
                outs = tank._compute_temperatures(fluid, SOIL, 80.0, 0)
                np.testing.assert_allclose(
                    tank._layer_temperatures, exact[:n], atol=1e-3, rtol=0
                )
                self.assertAlmostEqual(outs[1], exact[-1], delta=1e-3)
                self.assertAlmostEqual(outs[3], propagator[-1, n] / 80.0, delta=3e-5)
                stored = fluid.rho * volume * fluid.cp * (outs[2] - 50.0)
                np.testing.assert_allclose(stored, -3600.0 * outs[4], rtol=1e-10)

    def test_outlet_derivative_including_mixing(self):
        for fluid in (ConstantWater(), Water()):
            for mdot, tin in ((0.5, 85.0), (-0.5, 20.0), (0.5, 10.0), (-0.5, 95.0)):
                with self.subTest(fluid=fluid, mdot=mdot, tin=tin):
                    tank = make_tank(
                        n_layers=4, initial_layer_temperatures=[80.0, 65.0, 40.0, 25.0]
                    )
                    tank.set("mass_flow", mdot)
                    center = tank._compute_temperatures(fluid, SOIL, tin, 0)
                    plus = tank._compute_temperatures(fluid, SOIL, tin + 1e-3, 0)
                    minus = tank._compute_temperatures(fluid, SOIL, tin - 1e-3, 0)
                    self.assertAlmostEqual(
                        center[3], (plus[1] - minus[1]) / 2e-3, delta=1e-7
                    )

    def test_turnover_boundary_and_time_refinement(self):
        fluid = ConstantWater()
        outs = []
        for turnover in (0.999999, 1.000001):
            tank = make_tank(n_layers=1, u_value=0.0)
            tank.set("mass_flow", turnover * fluid.rho * 2.0 / 600.0)
            outs.append(tank._compute_temperatures(fluid, SOIL, 80.0, 0))
        self.assertLess(abs(outs[0][1] - outs[1][1]), 1e-4)
        self.assertLess(abs(outs[0][2] - outs[1][2]), 1e-4)
        exact = 80.0 - 30.0 * np.exp(-1.0)
        errors = []
        for count in (1, 200):
            tank = make_tank(n_layers=1, u_value=0.0, stepsize=600.0 / count)
            run_steps(tank, fluid, [fluid.rho * 2.0 / 600.0] * count, [80.0] * count)
            errors.append(abs(tank._layer_temperatures[0] - exact))
        self.assertLess(errors[0], 1e-3)
        self.assertLess(errors[1], errors[0] / 20.0)

    def test_mixing_time_refinement(self):
        """Unfavourable inlet temperatures must remain accurate when layers mix."""
        fluid = ConstantWater()
        for mdot, tin in ((0.5, 10.0), (-0.5, 95.0)):
            with self.subTest(mdot=mdot):
                profiles, outlets = [], []
                for count in (1, 200):
                    tank = make_tank(
                        n_layers=4,
                        u_value=0.0,
                        stepsize=600.0 / count,
                        initial_layer_temperatures=[80.0, 65.0, 40.0, 25.0],
                    )
                    outputs = run_steps(tank, fluid, [mdot] * count, [tin] * count)
                    profiles.append(tank._layer_temperatures.copy())
                    outlets.append(np.mean([out[1] for out in outputs]))
                np.testing.assert_allclose(profiles[0], profiles[1], atol=0.005, rtol=0)
                self.assertAlmostEqual(outlets[0], outlets[1], delta=0.001)

    def test_invalid_inlet_does_not_advance_state(self):
        tank = make_tank()
        tank.set("mass_flow", 0.5)
        before = tank._layer_temperatures.copy()
        with self.assertRaises(ValueError):
            tank._compute_temperatures(ConstantWater(), SOIL, np.nan, 0)
        np.testing.assert_array_equal(tank._layer_temperatures, before)
        self.assertIsNone(tank._last_ts)

    @staticmethod
    def _recirculation_network(
        volume=2.0,
        n_layers=10,
        mdot=0.5,
        stepsize=600.0,
        heater_type="t_out",
        heater_value=80.0,
    ):
        net = Network()
        for name in ("A", "B", "C"):
            net.add_node(name, temperature=50.0)
        net.add_stratified_storage(
            "T",
            "A",
            "B",
            volume=volume,
            n_layers=n_layers,
            mass_flow=mdot,
            stepsize=stepsize,
            u_value=0.0,
        )
        net.add_pipe("P", "B", "C", length=0.01, diameter=0.1, mass_flow=mdot)
        net.add_producer(
            "H",
            "C",
            "A",
            mass_flow=mdot,
            setpoint_type_hx=heater_type,
            setpoint_value_hx=heater_value,
        )
        return net

    def test_automatic_timestep_ids(self):
        from pydhn.solving import solve_thermal

        fluid = ConstantWater()
        auto = self._recirculation_network()
        explicit = deepcopy(auto)
        for step in range(3):
            with self.assertWarnsRegex(UserWarning, "No ts_id supplied"):
                a = solve_thermal(auto, fluid, SOIL, verbose=0)
            # Explicit IDs must not trigger the automatic-timestep warning.
            with catch_warnings():
                filterwarnings("error", message="No ts_id supplied", category=UserWarning)
                b = solve_thermal(explicit, fluid, SOIL, ts_id=step, verbose=0)
            self.assertTrue(a["history"]["thermal converged"])
            np.testing.assert_array_equal(
                auto["A", "B"]._layer_temperatures,
                explicit["A", "B"]._layer_temperatures,
            )
            np.testing.assert_array_equal(
                a["edges"]["outlet_temperature"], b["edges"]["outlet_temperature"]
            )
        solve_thermal(auto, fluid, SOIL, ts_id=10, verbose=0)
        with self.assertWarnsRegex(UserWarning, "No ts_id supplied"):
            solve_thermal(auto, fluid, SOIL, verbose=0)
        self.assertEqual(auto["A", "B"]._last_ts, 11)

    def test_recirculation_converges(self):
        from pydhn.solving import solve_thermal

        net = self._recirculation_network(
            volume=0.1,
            n_layers=1,
            mdot=1.0,
            stepsize=3600.0,
            heater_type="delta_t",
            heater_value=0.1,
        )
        result = solve_thermal(
            net, ConstantWater(), SOIL, ts_id=0, max_iters=5, verbose=0
        )
        self.assertTrue(result["history"]["thermal converged"])

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
