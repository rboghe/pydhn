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
    def test_special_and_missing_keys(self):
        comp = Component(diameter=0.5)
        self.assertEqual(comp["component_type"], "base_component")
        self.assertEqual(comp["component_class"], "branch_component")
        self.assertTrue(comp["is_ideal"])
        self.assertEqual(comp["diameter"], 0.5)
        self.assertTrue(np.isnan(comp["not_there"]))

    def test_consumer_energy_control(self):
        """
        With "energy" control, the mass flow setpoint is computed from the
        heat demand instead of using the stored value.
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
        cons.set("control_type", "mass_flow")
        self.assertEqual(cons["setpoint_value_hyd"], 123.0)

    def test_branch_pump_setpoint_type(self):
        """The hydraulic setpoint type of a pump is always "pressure"."""
        pump = BranchPump()
        pump.set("setpoint_type_hyd", "mass_flow")
        self.assertEqual(pump["setpoint_type_hyd"], "pressure")

    def test_declared_keys(self):
        """Only the declared keys go through the control logic."""
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
            with self.subTest(cls=cls.__name__):
                calls.clear()
                comp = cls(other=7.0)
                self.assertEqual(comp["magic"], 42.0)
                self.assertEqual(comp["other"], 7.0)
                self.assertEqual(calls, ["magic"])

    def test_undeclared_keys(self):
        """
        Custom components that override the control logic without declaring
        their keys get it called for every key, also when they inherit from
        a component that declares its own.
        """
        for base in (Component, Consumer, BranchPump):

            class Custom(base):
                def _run_control_logic(self, key):
                    if key == "setpoint_value_hx":
                        return self._attrs["target"]
                    return super()._run_control_logic(key)

            class Inherited(Custom):
                pass

            for cls in (Custom, Inherited):
                with self.subTest(base=base.__name__, cls=cls.__name__):
                    comp = cls(target=-1234.0, other=7.0)
                    comp.set("setpoint_value_hx", -1.0)
                    self.assertEqual(comp["setpoint_value_hx"], -1234.0)
                    self.assertEqual(comp["other"], 7.0)
                    self.assertTrue(np.isnan(comp["not_there"]))

    def test_inherited_keys(self):
        """Subclasses that do not override the control logic keep its keys."""
        for base in (Consumer, BranchPump):

            class Inherited(base):
                pass

            self.assertEqual(Inherited._controlled_keys, base._controlled_keys)


if __name__ == "__main__":
    unittest.main()
