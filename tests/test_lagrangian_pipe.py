#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
Mirror-symmetry tests for the LagrangianPipe thermal model.

A pipe driven with negative mass flow must behave as the exact mirror image
of an identical pipe driven with positive flow: same outlet, average and
loss values, and internal parcel state reversed. This catches reference
frame errors in the negative-flow handling (e.g. remapping wall temperatures
on an un-flipped volume grid), which only surface once parcel widths and
wall temperatures are non-uniform.
"""

import unittest

import numpy as np

from pydhn.classes import Network
from pydhn.components import LagrangianPipe
from pydhn.components.vector_functions import COMPONENT_FUNCTIONS_DICT
from pydhn.fluids import Water
from pydhn.soils import Soil
from pydhn.solving.temperature import compute_edge_temperatures

STEPSIZE = 60.0
PIPE_KWARGS = dict(diameter=0.05, length=100.0, stepsize=STEPSIZE)

# Varying flow -> non-uniform parcels; varying inlet -> wall gradients
MDOTS = [0.3, 0.5, 0.1, 0.6, 0.0, 0.2, 0.45, 0.15, 0.35, 0.25]
T_INS = [70.0, 55.0, 80.0, 60.0, 75.0, 50.0, 65.0, 78.0, 58.0, 72.0]


def assert_mirrored(case, fwd_state, rev_state, step):
    for name, a, b in zip(["volumes", "temps", "walls"], fwd_state, rev_state):
        np.testing.assert_allclose(
            a, b[::-1], rtol=1e-12, atol=1e-12,
            err_msg=f"step {step}: {name} not mirrored",
        )


class LagrangianPipeMirrorSymmetry(unittest.TestCase):
    def test_solver_removes_roundoff_imaginary_parts(self):
        matrix_a = np.array([[[-1.0, 1.0], [0.5, -1.5]]])
        vector_b = np.array([[0.0], [4.0]])

        temperatures = LagrangianPipe._solve_diff_sys(
            matrix_A=matrix_a,
            vector_b=vector_b,
            x_0=np.array([60.0]),
            y_0=np.array([50.0]),
            stepsize=60.0,
        )

        for value in temperatures:
            self.assertFalse(np.iscomplexobj(value))

    def test_solver_rejects_significant_imaginary_parts(self):
        with self.assertRaises(FloatingPointError):
            LagrangianPipe._real_if_close(np.array([1.0 + 1e-3j]))

    def test_scalar_model(self):
        fluid, soil = Water(), Soil(temp=8)
        fwd, rev = LagrangianPipe(**PIPE_KWARGS), LagrangianPipe(**PIPE_KWARGS)
        for ts, (mdot, t_in) in enumerate(zip(MDOTS, T_INS)):
            outs = []
            for pipe, sign in ((fwd, 1.0), (rev, -1.0)):
                pipe.set("mass_flow", sign * mdot)
                pipe._compute_delta_p(fluid)  # sets reynolds and friction
                outs.append(
                    pipe._compute_temperatures(fluid, soil, t_in=t_in, ts_id=ts)
                )
            # At zero flow nothing is flipped, so the twin's t_in and t_out
            # legitimately refer to the opposite ends of the pipe
            f, (r_in, r_out, *r_rest) = outs
            expected = (r_out, r_in, *r_rest) if mdot == 0 else outs[1]
            np.testing.assert_allclose(
                f, expected, rtol=1e-12, err_msg=f"step {ts}: outputs"
            )
            assert_mirrored(
                self,
                (fwd._volumes, fwd._temperatures, fwd._wall_temperatures),
                (rev._volumes, rev._temperatures, rev._wall_temperatures),
                ts,
            )

    def test_vectorized_model(self):
        # One network with a forward pipe and its mirror twin driven at once
        fluid, soil = Water(), Soil(temp=8)
        net = Network()
        for n in ("F0", "F1", "R0", "R1"):
            net.add_node(n, x=0.0, y=0.0, z=0.0)
        net.add_lagrangian_pipe("F", "F0", "F1", **PIPE_KWARGS)
        net.add_lagrangian_pipe("R", "R0", "R1", **PIPE_KWARGS)
        mask = net.mask(attr="component_type", value="lagrangian_pipe")

        for ts, (mdot, t_in) in enumerate(zip(MDOTS, T_INS)):
            net.set_edge_attributes([mdot, -mdot], "mass_flow")
            net.set_node_attribute(t_in, "temperature")
            COMPONENT_FUNCTIONS_DICT["lagrangian_pipe"]["delta_p"](
                net, fluid, mask=mask
            )
            t_i, t_o, t_a, _, dq = compute_edge_temperatures(
                net, fluid, soil, ts_id=ts
            )
            # t_out is end-dependent and swaps at zero flow; the state mirror
            # assertion below covers it there
            checks = [(t_a, "t_avg"), (dq, "delta_q")]
            if mdot != 0:
                checks.append((t_o, "t_out"))
            for arr, name in checks:
                np.testing.assert_allclose(
                    arr[0], arr[1], rtol=1e-12, err_msg=f"step {ts}: {name}"
                )
            f, r = net[("F0", "F1")], net[("R0", "R1")]
            assert_mirrored(
                self,
                (f._volumes, f._temperatures, f._wall_temperatures),
                (r._volumes, r._temperatures, r._wall_temperatures),
                ts,
            )


if __name__ == "__main__":
    unittest.main()
