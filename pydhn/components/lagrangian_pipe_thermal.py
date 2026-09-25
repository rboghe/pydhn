#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only


"""
Vectorized thermal model of the LagrangianPipe component.

It computes one time step for all the Lagrangian pipes of a network at once,
giving the same results as LagrangianPipe._compute_temperatures(). The
parcels of all pipes are stored one after the other in flat arrays to compute
heat transfer, and in zero-padded 2D arrays (one row per pipe) to move them.
"""

import numpy as np

from pydhn.components.lagrangian_pipe import LagrangianPipe
from pydhn.fluids.dimensionless_numbers import compute_nusselt
from pydhn.fluids.dimensionless_numbers import compute_prandtl
from pydhn.utilities import safe_divide


def compute_lagrangian_temp_net(net, fluid, soil, ts_id=None):
    """
    Computes one time step for all the Lagrangian pipes of the network and
    updates their internal state. Returns t_in, t_out, t_avg, t_out_der and
    delta_q, following the order of the Lagrangian pipes in the network.
    """
    mask = net.mask(attr="component_type", value="lagrangian_pipe")
    edges = net.edges(mask=mask)
    pipes = [net[(u, v)] for u, v in edges]
    P = len(pipes)

    # Get attributes
    keys = (
        "mass_flow",
        "stepsize",
        "depth",
        "diameter",
        "k_internal_pipe",
        "k_insulation",
        "k_casing",
        "reynolds",
        "length",
        "friction_factor",
        "h_ext",
        "rho_wall",
        "cp_wall",
    )
    (
        mdot_signed,
        stepsize,
        depth,
        diameter,
        k_internal_pipe,
        k_insulation,
        k_casing,
        reynolds,
        length,
        friction_factor,
        h_ext,
        rho_wall,
        cp_wall,
    ) = np.array([[c._attrs[k] for k in keys] for c in pipes], dtype=float).T

    # If it is a repeated step, restore previous conditions, then update the
    # internal memory
    for c in pipes:
        if c._last_ts is not None and c._last_ts == ts_id:
            c._volumes = c._last_volumes.copy()
            c._temperatures = c._last_temperatures.copy()
            c._wall_temperatures = c._last_wall_temperatures.copy()
        c._last_ts = ts_id
        c._last_volumes = c._volumes.copy()
        c._last_temperatures = c._temperatures.copy()
        c._last_wall_temperatures = c._wall_temperatures.copy()

    # Get pipe characteristics
    internal_volume = np.array([c._internal_volume for c in pipes])
    section_area = np.array([c._section_area for c in pipes])
    wall_section_area = np.array([c._wall_section_area for c in pipes])
    r_w = np.array([c._r_w for c in pipes])
    r_s = np.array([c._r_s for c in pipes])
    r_ins = np.array([c._r_ins for c in pipes])
    r_cas = np.array([c._r_cas for c in pipes])

    # Put the parcels of all pipes in flat arrays. Pipes with negative mass
    # flow are flipped, so that parcels are always ordered from the inlet.
    reversed_ = mdot_signed < 0
    mdot = np.abs(mdot_signed)
    counts = np.array([len(c._volumes) for c in pipes])
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))

    def gather(attr):
        values = [getattr(c, attr) for c in pipes]
        return np.concatenate([v[::-1] if r else v for v, r in zip(values, reversed_)])

    volumes = gather("_volumes")
    temps = gather("_temperatures")
    wall_temps = gather("_wall_temperatures")

    # Get the inlet temperature from the upstream node
    nodes, t_nodes = net.nodes(data="temperature")
    t_dict = dict(zip(nodes, t_nodes))
    t_in = np.array([t_dict[v if r else u] for (u, v), r in zip(edges, reversed_)])

    # Get soil temperature
    t_soil = np.broadcast_to(soil.get_temp(depth=depth, ts=ts_id), P)

    # Repeat pipe values for each of their parcels
    def rep(values):
        return np.repeat(values, counts)

    # Get fluid properties for each parcel
    cp_fluid = fluid.get_cp(temps)
    rho_fluid = fluid.get_rho(temps)
    k_fluid = fluid.get_k(temps)
    mu_fluid = fluid.get_mu(temps)

    # Get dimensionless numbers
    nu = compute_nusselt(
        reynolds=rep(reynolds),
        diameter=rep(diameter),
        length=rep(length),
        friction_factor=rep(friction_factor),
        cp_fluid=cp_fluid,
        mu_fluid=mu_fluid,
        k_fluid=k_fluid,
    )
    prandtl = compute_prandtl(cp_fluid=cp_fluid, mu_fluid=mu_fluid, k_fluid=k_fluid)

    # Compute heat transfer coefficients, as in the scalar model
    h_ins = safe_divide(2 * np.pi * k_insulation, np.log(safe_divide(r_ins, r_s)))
    h_cas = safe_divide(2 * np.pi * k_casing, np.log(safe_divide(r_cas, r_ins)))
    h_ext_pipe = h_ext * 2 * np.pi * r_cas
    denom = safe_divide(1, h_ins) + safe_divide(1, h_cas) + safe_divide(1, h_ext_pipe)
    h_ins_cas = safe_divide(1, denom)
    h_s = safe_divide(2 * np.pi * k_internal_pipe, np.log(safe_divide(r_s, r_w)))
    h_w = nu * k_fluid * np.pi
    h_w_cv = safe_divide(nu * k_fluid * np.pi, rep(diameter))
    h_b = safe_divide(1, safe_divide(1, h_w) + safe_divide(1, rep(h_s)))

    # Compute boundary layers
    thermal_layer_thickness = safe_divide(k_fluid, h_w_cv)
    momentum_layer_thickness = thermal_layer_thickness * (prandtl ** (1 / 3))
    momentum_layer_thickness = np.where(
        np.isnan(momentum_layer_thickness), 0, momentum_layer_thickness
    )

    # Compute capacities
    C_wall = rho_wall * wall_section_area * cp_wall
    core_section_area = np.pi * (rep(r_w) - momentum_layer_thickness) ** 2
    C_core = core_section_area * cp_fluid * rho_fluid
    C_b = (rep(section_area) - core_section_area) * cp_fluid * rho_fluid + rep(C_wall)

    # Solve the differential equations of all parcels at once
    N = len(h_b)
    matrix_A = np.ones([N, 2, 2])
    matrix_A[:, 0, 0] = safe_divide(-1 * h_b, C_core)
    matrix_A[:, 0, 1] = safe_divide(h_b, C_core)
    matrix_A[:, 1, 0] = safe_divide(h_b, C_b)
    matrix_A[:, 1, 1] = safe_divide(-1 * (h_b + rep(h_ins_cas)), C_b)
    vector_b = np.array([np.zeros(N), safe_divide(rep(h_ins_cas), C_b) * rep(t_soil)])
    new_temps, new_wall_temps = LagrangianPipe._solve_diff_sys(
        matrix_A=matrix_A,
        vector_b=vector_b,
        x_0=temps,
        y_0=wall_temps,
        stepsize=rep(stepsize),
    )

    # Compute losses of each pipe
    delta_qs = safe_divide(volumes * cp_fluid * (new_temps - temps) * rho_fluid, 3600.0)
    delta_q = np.add.reduceat(delta_qs, starts)

    # Move the parcels in 2D arrays with one row per pipe. The first column is
    # left free for the new parcel entering flowing pipes, and there is room
    # for splitting the parcel that partly leaves the pipe.
    flow = mdot != 0
    new_vol = np.where(flow, safe_divide(mdot * stepsize, fluid.get_rho(t_in)), 0.0)
    W = counts.max() + 2
    rows = np.arange(P)
    vols2 = np.zeros((P, W))
    temps2 = np.zeros((P, W))
    row_of = rep(rows)
    col_of = np.arange(len(volumes)) - rep(starts) + rep(flow)
    vols2[row_of, col_of] = volumes
    temps2[row_of, col_of] = new_temps
    vols2[flow, 0] = new_vol[flow]
    temps2[flow, 0] = t_in[flow]

    # Find the parcel that exceeds the pipe volume and split it in the part
    # that stays and the part that leaves, unless the new parcel exactly fills
    # the pipe
    cumsum = np.cumsum(vols2, axis=1)
    exact_fill = flow & (new_vol == internal_volume)
    split = flow & ~exact_fill
    out_idx = np.argmax(cumsum > internal_volume[:, None], axis=1)
    out_part = np.where(split, cumsum[rows, out_idx] - internal_volume, 0.0)
    in_part = np.where(split, vols2[rows, out_idx] - out_part, 0.0)

    # Insert the staying part at out_idx by shifting the following parcels to
    # the right. The temperature of the split parcel is repeated.
    cols = np.arange(W)[None, :]
    shift = split[:, None] & (cols > out_idx[:, None])
    src = np.clip(cols - shift, 0, W - 1)
    vols3 = np.take_along_axis(vols2, src, axis=1)
    temps3 = np.take_along_axis(temps2, src, axis=1)
    vols3[rows[split], out_idx[split]] = in_part[split]
    vols3[rows[split], out_idx[split] + 1] = out_part[split]

    # Find staying and leaving parcels
    n_stay = np.where(flow, np.where(exact_fill, 1, out_idx + 1), counts)
    staying = cols < n_stay[:, None]
    leaving = (cols < (counts + flow + split)[:, None]) & ~staying

    # Compute outlet temperature. If the mass flow is zero, use the
    # temperatures of the first and last parcels.
    lv = np.where(leaving, vols3, 0.0)
    t_out = np.where(
        flow,
        safe_divide((lv * temps3).sum(axis=1), lv.sum(axis=1)),
        temps3[rows, counts - 1],
    )
    t_in = np.where(flow, t_in, temps3[:, 0])

    # Compute average temperature
    sv = np.where(staying, vols3, 0.0)
    t_avg = (sv * temps3).sum(axis=1) / sv.sum(axis=1)

    # Update internal memory, restoring the reference frame of reversed pipes
    for p, c in enumerate(pipes):
        old = slice(starts[p], starts[p] + counts[p])
        staying_volumes = vols3[p, : n_stay[p]]
        staying_temperatures = temps3[p, : n_stay[p]]
        wall_temperatures = new_wall_temps[old]
        # Update wall discretization to match that of volumes
        if flow[p]:
            wall_temperatures = np.interp(
                np.cumsum(staying_volumes), np.cumsum(volumes[old]), wall_temperatures
            )
        step = -1 if reversed_[p] else 1
        c._volumes = staying_volumes[::step].copy()
        c._temperatures = staying_temperatures[::step].copy()
        c._wall_temperatures = wall_temperatures[::step].copy()

    return t_in, t_out, t_avg, np.zeros(P), delta_q
