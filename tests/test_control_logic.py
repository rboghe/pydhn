#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Tests for the control logic dispatch in component attribute lookups"""

import unittest

import numpy as np

from pydhn.components import BranchPump
from pydhn.components import Component
from pydhn.components import Consumer
from pydhn.default_values import CP_FLUID


class ControlLogicDispatchTestCase(unittest.TestCase):
    def test_special_keys(self):
        comp = Component()
        self.assertEqual(comp["component_type"], "base_component")
        self.assertEqual(comp["component_class"], "branch_component")
        self.assertTrue(comp["is_ideal"])

    def test_plain_and_missing_attributes(self):
        comp = Component(diameter=0.5)
        self.assertEqual(comp["diameter"], 0.5)
        self.assertTrue(np.isnan(comp["not_there"]))

    def test_consumer_energy_control(self):
        """
        With 'energy' control the hydraulic setpoint must be computed from
        the heat demand, overriding the stored attribute, and react to
        set() without any cache invalidation.
        """
        cons = Consumer(
            control_type="energy",
            setpoint_type_hyd="mass_flow",
            heat_demand=-6000.0,
            design_delta_t=20.0,
            setpoint_value_hyd=123.0,
        )
        self.assertAlmostEqual(cons["setpoint_value_hyd"], 300.0 / CP_FLUID)
        cons.set("heat_demand", -12000.0)
        self.assertAlmostEqual(cons["setpoint_value_hyd"], 600.0 / CP_FLUID)

        # With a different control type, the stored attribute is returned
        cons.set("control_type", "mass_flow")
        self.assertEqual(cons["setpoint_value_hyd"], 123.0)

    def test_branch_pump_setpoint_type(self):
        """The hydraulic setpoint type of a pump is always 'pressure'."""
        pump = BranchPump()
        pump.set("setpoint_type_hyd", "mass_flow")
        self.assertEqual(pump["setpoint_type_hyd"], "pressure")

    def test_declared_controlled_keys_limit_dispatch(self):
        calls = []

        class Declared(Component):
            _controlled_keys = frozenset({"magic"})

            def _run_control_logic(self, key):
                calls.append(key)
                return 42.0 if key == "magic" else None

        comp = Declared(other=1.0)
        self.assertEqual(comp["magic"], 42.0)
        self.assertEqual(comp["other"], 1.0)
        self.assertEqual(calls, ["magic"])

    def test_legacy_override_without_declaration(self):
        """
        Custom components that override _run_control_logic without declaring
        _controlled_keys must keep the legacy behavior of having it called
        for every key.
        """

        class Legacy(Component):
            def _run_control_logic(self, key):
                if key == "magic":
                    return 42.0
                return None

        comp = Legacy(other=1.0)
        self.assertEqual(comp["magic"], 42.0)
        self.assertEqual(comp["other"], 1.0)
        self.assertTrue(np.isnan(comp["not_there"]))

    def test_legacy_builtin_subclass_override(self):
        """Inherited key declarations must not suppress a new control method."""
        for base, controlled_key in (
            (Consumer, "setpoint_value_hx"),
            (BranchPump, "setpoint_value_hyd"),
        ):

            class Legacy(base):
                def _run_control_logic(self, key):
                    if key == controlled_key:
                        return self._attrs["target"]
                    return super()._run_control_logic(key)

            class Inherited(Legacy):
                pass

            for cls in (Legacy, Inherited):
                with self.subTest(base=base.__name__, subclass=cls.__name__):
                    comp = cls(target=-1234.0, other=7.0)
                    comp.set(controlled_key, -1.0)
                    self.assertEqual(comp[controlled_key], -1234.0)
                    comp.set("target", -2345.0)
                    self.assertEqual(comp[controlled_key], -2345.0)
                    self.assertEqual(comp["other"], 7.0)
                    self.assertTrue(np.isnan(comp["not_there"]))

    def test_inherited_builtin_control_keeps_declared_keys(self):
        """Inheriting an unchanged control method retains restricted dispatch."""
        for base in (Consumer, BranchPump):

            class Inherited(base):
                pass

            with self.subTest(base=base.__name__):
                comp = Inherited()
                self.assertEqual(comp._controlled_keys, base._controlled_keys)
                self.assertNotIn("other", comp._controlled_keys)

    def test_builtin_subclass_can_declare_controlled_keys(self):
        calls = []

        class Declared(Consumer):
            _controlled_keys = Consumer._controlled_keys | {"magic"}

            def _run_control_logic(self, key):
                calls.append(key)
                if key == "magic":
                    return 42.0
                return super()._run_control_logic(key)

        class Inherited(Declared):
            pass

        for cls in (Declared, Inherited):
            with self.subTest(subclass=cls.__name__):
                calls.clear()
                comp = cls(other=7.0)
                self.assertEqual(comp["magic"], 42.0)
                self.assertEqual(comp["other"], 7.0)
                self.assertEqual(calls, ["magic"])


if __name__ == "__main__":
    unittest.main()
