#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only


"""
Vectorized thermal model for LagrangianPipe components.

Computes one thermal step for all lagrangian pipes of a network at once,
replicating pipe-by-pipe `LagrangianPipe._compute_temperatures`. The ragged
per-pipe parcel state is packed into flat (CSR-like) arrays for the physics
and into zero-padded 2D arrays for the parcel displacement. 
"""

import numpy as np

from pydhn.components.lagrangian_pipe import LagrangianPipe
from pydhn.fluids.dimensionless_numbers import compute_nusselt
from pydhn.fluids.dimensionless_numbers import compute_prandtl
from pydhn.utilities import safe_divide


def _pad(values, starts, counts, width, fill=0.0):
    """Scatters flat per-parcel values into a zero-padded [P, width] array."""
    P = len(counts)
    out = np.full((P, width), fill)
    cols = np.arange(len(values)) - np.repeat(starts, counts)
    out[np.repeat(np.arange(P), counts), cols] = values
    return out


def _interp_rows(x, xp, fp, x_valid, xp_valid):
    """
    Row-wise ``np.interp(x[i], xp[i], fp[i])`` on padded 2D arrays, using only
    the entries flagged valid. Replicates np.interp semantics: same slope
    rounding, left/right clamping, and duplicate grid points resolving to the
    last point with ``xp <= x``.
    """
    P, W = xp.shape
    n_xp = xp_valid.sum(axis=1)
    # Move each row's grid onto a private, strictly increasing global axis so
    # a single searchsorted brackets every query; the local values are then
    # used for the arithmetic
    offset = (np.arange(P) * 2.0 * (np.abs(xp).max() + 1.0))[:, None]
    order = np.argsort(np.where(xp_valid, xp + offset, np.inf).ravel(), kind="stable")
    xp_glob = np.where(xp_valid, xp + offset, np.inf).ravel()[order]
    xp_loc = xp.ravel()[order]
    fp_loc = fp.ravel()[order]

    row_start = np.concatenate(([0], np.cumsum(n_xp)[:-1]))[:, None]
    row_end = row_start + n_xp[:, None] - 1
    j = np.searchsorted(xp_glob, (x + offset).ravel(), side="right").reshape(P, W)
    lo = np.clip(j - 1, row_start, row_end)
    hi = np.clip(j, row_start, row_end)

    x0, x1 = xp_loc[lo], xp_loc[hi]
    y0, y1 = fp_loc[lo], fp_loc[hi]
    with np.errstate(divide="ignore", invalid="ignore"):
        res = (y1 - y0) / (x1 - x0) * (x - x0) + y0
    res = np.where(x >= x1, y1, res)  # right clamp and degenerate intervals
    res = np.where(x <= x0, y0, res)  # left clamp and exact hits
    return np.where(x_valid, res, 0.0)


def compute_lagrangian_temp_net(net, fluid, soil, ts_id=None):
    """
    Vectorized equivalent of ``LagrangianPipe._compute_temperatures`` for all
    lagrangian pipes of ``net``. Returns per-pipe ``(t_in, t_out, t_avg,
    t_out_der, delta_q)`` in mask order and updates the pipes' parcel state.
    """
    mask = net.mask(attr="component_type", value="lagrangian_pipe")
    edges = net.edges(mask=mask)
    pipes = [net[(u, v)] for u, v in edges]
    P = len(pipes)

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

    # Restore state on repeated time steps, then snapshot it (as in the
    # scalar model)
    for c in pipes:
        if c._last_ts is not None and c._last_ts == ts_id:
            c._volumes = c._last_volumes.copy()
            c._temperatures = c._last_temperatures.copy()
            c._wall_temperatures = c._last_wall_temperatures.copy()
        c._last_ts = ts_id
        c._last_volumes = c._volumes.copy()
        c._last_temperatures = c._temperatures.copy()
        c._last_wall_temperatures = c._wall_temperatures.copy()

    # Gather ragged parcel state into flat arrays; reversed pipes are flipped
    # into flow direction
    reversed_ = mdot_signed < 0
    mdot = np.abs(mdot_signed)
    counts = np.fromiter((len(c._volumes) for c in pipes), int, count=P)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    internal_volume = np.fromiter((c._internal_volume for c in pipes), float, count=P)
    section_area = np.fromiter((c._section_area for c in pipes), float, count=P)
    wall_section_area = np.fromiter(
        (c._wall_section_area for c in pipes), float, count=P
    )
    r_w = np.fromiter((c._r_w for c in pipes), float, count=P)
    r_s = np.fromiter((c._r_s for c in pipes), float, count=P)
    r_ins = np.fromiter((c._r_ins for c in pipes), float, count=P)
    r_cas = np.fromiter((c._r_cas for c in pipes), float, count=P)

    def gather(attr):
        return np.concatenate(
            [getattr(c, attr)[::-1] if r else getattr(c, attr)
             for c, r in zip(pipes, reversed_)]
        )

    volumes = gather("_volumes")
    temps = gather("_temperatures")
    wall_temps = gather("_wall_temperatures")

    # Inlet temperature from the upstream node
    nodes, t_nodes = net.nodes(data="temperature")
    t_dict = dict(zip(nodes, t_nodes))
    t_in = np.where(
        np.sign(mdot_signed) >= 0,
        [t_dict[u] for u, v in edges],
        [t_dict[v] for u, v in edges],
    )

    t_soil = np.broadcast_to(soil.get_temp(depth=depth, ts=ts_id), P)

    # ------------------- Per-parcel physics (flat arrays) ------------------ #
    rep = lambda a: np.repeat(a, counts)

    cp_fluid = fluid.get_cp(temps)
    rho_fluid = fluid.get_rho(temps)
    k_fluid = fluid.get_k(temps)
    mu_fluid = fluid.get_mu(temps)

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

    # Heat transfer coefficients (per pipe, then per parcel where needed)
    h_ins = safe_divide(2 * np.pi * k_insulation, np.log(safe_divide(r_ins, r_s)))
    h_cas = safe_divide(2 * np.pi * k_casing, np.log(safe_divide(r_cas, r_ins)))
    h_ext_pipe = h_ext * 2 * np.pi * r_cas
    denom = safe_divide(1, h_ins) + safe_divide(1, h_cas) + safe_divide(1, h_ext_pipe)
    h_ins_cas = safe_divide(1, denom)
    h_s = safe_divide(2 * np.pi * k_internal_pipe, np.log(safe_divide(r_s, r_w)))
    h_w = nu * k_fluid * np.pi
    h_w_cv = safe_divide(nu * k_fluid * np.pi, rep(diameter))
    h_b = safe_divide(1, safe_divide(1, h_w) + safe_divide(1, rep(h_s)))

    # Boundary layers
    thermal_layer_thickness = safe_divide(k_fluid, h_w_cv)
    momentum_layer_thickness = thermal_layer_thickness * (prandtl ** (1 / 3))
    momentum_layer_thickness = np.where(
        np.isnan(momentum_layer_thickness), 0, momentum_layer_thickness
    )

    # Capacities
    C_wall = rho_wall * wall_section_area * cp_wall
    core_section_area = np.pi * (rep(r_w) - momentum_layer_thickness) ** 2
    C_core = core_section_area * cp_fluid * rho_fluid
    C_b = (rep(section_area) - core_section_area) * cp_fluid * rho_fluid + rep(C_wall)

    # ODE solve for all parcels of all pipes in one batched call
    N = len(h_b)
    matrix_A = np.ones([N, 2, 2])
    matrix_A[:, 0, 0] = safe_divide(-1 * h_b, C_core)
    matrix_A[:, 0, 1] = safe_divide(h_b, C_core)
    matrix_A[:, 1, 0] = safe_divide(h_b, C_b)
    matrix_A[:, 1, 1] = safe_divide(-1 * (h_b + rep(h_ins_cas)), C_b)
    vector_b = np.array(
        [np.zeros(N), safe_divide(rep(h_ins_cas), C_b) * rep(t_soil)]
    )
    new_temps, new_wall_temps = LagrangianPipe._solve_diff_sys(
        matrix_A=matrix_A,
        vector_b=vector_b,
        x_0=temps,
        y_0=wall_temps,
        stepsize=rep(stepsize),
    )

    # Heat losses per pipe
    delta_qs = safe_divide(volumes * cp_fluid * (new_temps - temps) * rho_fluid, 3600.0)
    delta_q = np.add.reduceat(delta_qs, starts)

    # ---------------- Parcel displacement (padded 2D arrays) --------------- #
    flow = mdot != 0
    new_vol = np.where(flow, safe_divide(mdot * stepsize, fluid.get_rho(t_in)), 0.0)

    W = counts.max() + 2  # room for the inlet parcel and the split
    vols2 = np.zeros((P, W))
    temps2 = np.zeros((P, W))
    col_of = np.arange(len(volumes)) - np.repeat(starts, counts) + flow[
        np.repeat(np.arange(P), counts)
    ].astype(int)
    row_of = np.repeat(np.arange(P), counts)
    vols2[row_of, col_of] = volumes
    temps2[row_of, col_of] = new_temps
    vols2[flow, 0] = new_vol[flow]
    temps2[flow, 0] = t_in[flow]
    counts2 = counts + flow

    # First parcel (index out_idx) whose cumulative volume exceeds the pipe
    # volume gets split into a staying and a leaving part
    cumsum = np.cumsum(vols2, axis=1)
    exact_fill = flow & (new_vol == internal_volume)
    over = cumsum > internal_volume[:, None]
    out_idx = np.argmax(over, axis=1)
    split = flow & ~exact_fill
    out_part = np.where(split, cumsum[np.arange(P), out_idx] - internal_volume, 0.0)
    in_part = np.where(split, vols2[np.arange(P), out_idx] - out_part, 0.0)

    # Insert in_part at out_idx by shifting the rest right; temperatures at
    # the split are duplicated by the gather
    cols = np.arange(W)[None, :]
    shift = split[:, None] & (cols > out_idx[:, None])
    src = np.clip(cols - shift, 0, W - 1)
    vols3 = np.take_along_axis(vols2, src, axis=1)
    temps3 = np.take_along_axis(temps2, src, axis=1)
    rows_split = np.where(split)[0]
    vols3[rows_split, out_idx[rows_split]] = in_part[rows_split]
    vols3[rows_split, out_idx[rows_split] + 1] = out_part[rows_split]
    counts3 = counts2 + split

    # Number of parcels staying in the pipe
    n_stay = np.where(flow, np.where(exact_fill, 1, out_idx + 1), counts)
    staying = cols < n_stay[:, None]
    valid = cols < counts3[:, None]
    leaving = valid & ~staying

    # Outlet temperature: volume-weighted average of the leaving parcels for
    # flowing pipes, last parcel for still ones
    lv = np.where(leaving, vols3, 0.0)
    t_out = np.where(
        flow,
        safe_divide((lv * temps3).sum(axis=1), lv.sum(axis=1)),
        temps3[np.arange(P), np.maximum(counts - 1, 0)],
    )
    t_in = np.where(flow, t_in, temps3[:, 0])

    sv = np.where(staying, vols3, 0.0)
    t_avg = (sv * temps3).sum(axis=1) / sv.sum(axis=1)

    # Remap wall temperatures onto the new parcel grid of flowing pipes;
    # `volumes` is the old grid already flipped into flow direction
    wall2 = _pad(new_wall_temps, starts, counts, W)
    last_cumsum = np.cumsum(_pad(volumes, starts, counts, W), axis=1)
    stay_cumsum = np.cumsum(sv, axis=1)
    wall3 = _interp_rows(
        x=stay_cumsum,
        xp=last_cumsum,
        fp=wall2,
        x_valid=staying,
        xp_valid=cols < counts[:, None],
    )
    wall3 = np.where(flow[:, None], wall3, wall2)

    # Write the new state back, restoring the original orientation
    for p, c in enumerate(pipes):
        n = n_stay[p]
        sl = slice(n - 1, None, -1) if reversed_[p] else slice(None, n)
        c._volumes = vols3[p, sl].copy()
        c._temperatures = temps3[p, sl].copy()
        c._wall_temperatures = wall3[p, sl].copy()

    return t_in, t_out, t_avg, np.zeros(P), delta_q
