#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only


"""
Vectorized thermal model for StratifiedStorage components.

Computes one thermal step for all stratified storages of a network at once,
replicating tank-by-tank ``StratifiedStorage._compute_temperatures``. Layer
states are packed into a zero-padded [S, N] grid on which the Thomas solver
sweeps run sequentially over layers but vectorized across tanks, which makes
every result bitwise identical to the scalar model. Padded layers are given
decoupled identity rows so they cannot affect real ones. The tank objects
remain the owners of the state.
"""

import numpy as np

from pydhn.components.stratified_storage import StratifiedStorage


def _batched_thomas(lower, diag, upper, rhs):
    """
    Solves independent tridiagonal systems stored row-wise in [S, N] arrays
    with the Thomas algorithm, sweeping over layers and vectorized across
    systems. ``lower`` and ``upper`` have N - 1 columns.
    """
    S, N = diag.shape
    d, r = diag.copy(), rhs.copy()
    for i in range(1, N):
        w = lower[:, i - 1] / d[:, i - 1]
        d[:, i] -= w * upper[:, i - 1]
        r[:, i] -= w * r[:, i - 1]
    x = np.empty((S, N))
    x[:, -1] = r[:, -1] / d[:, -1]
    for i in range(N - 2, -1, -1):
        x[:, i] = (r[:, i] - upper[:, i] * x[:, i + 1]) / d[:, i]
    return x


def compute_storage_temp_net(net, fluid, soil, ts_id=None):
    """
    Vectorized equivalent of ``StratifiedStorage._compute_temperatures`` for
    all stratified storages of ``net``. Returns per-tank ``(t_in, t_out,
    t_avg, t_out_der, delta_q)`` in mask order and updates the tanks' layer
    state.
    """
    mask = net.mask(attr="component_type", value="stratified_storage")
    edges = net.edges(mask=mask)
    tanks = [net[(u, v)] for u, v in edges]
    S = len(tanks)

    keys = ("mass_flow", "stepsize", "t_ambient", "n_layers", "delta_k")
    mdot_signed, stepsize, t_amb, n_layers, delta_k = np.array(
        [[c._attrs[k] for k in keys] for c in tanks], dtype=float
    ).T
    n_layers = n_layers.astype(int)
    N = n_layers.max()
    rows = np.arange(S)
    cols = np.arange(N)[None, :]
    valid = cols < n_layers[:, None]

    # Restore state on repeated time steps, then snapshot it. Per-tank
    # scalars are computed here so that they stay bitwise equal to the
    # scalar model; in particular, fluid properties are evaluated with
    # scalar calls, since the array kernels of the power ufunc can round
    # differently on some platforms
    temps = np.zeros((S, N))
    layer_volume = np.empty(S)
    section_area = np.empty(S)
    dz = np.empty(S)
    ua = np.zeros((S, N))
    cp = np.empty(S)
    rho = np.empty(S)
    k_fluid = np.empty(S)
    for s, c in enumerate(tanks):
        if c._last_ts is not None and c._last_ts == ts_id:
            c._layer_temperatures = c._last_layer_temperatures.copy()
        c._last_ts = ts_id
        c._last_layer_temperatures = c._layer_temperatures.copy()
        n = n_layers[s]
        temps[s, :n] = c._layer_temperatures
        layer_volume[s] = c._layer_volume
        section_area[s] = c._section_area
        dz[s] = c._dz
        ua[s, :n] = c._ua_layers
        t_mean = c._layer_temperatures.mean()
        cp[s] = fluid.get_cp(t_mean)
        rho[s] = fluid.get_rho(t_mean)
        k_fluid[s] = fluid.get_k(t_mean)

    # Inlet temperature from the upstream node
    nodes, t_nodes = net.nodes(data="temperature")
    t_dict = dict(zip(nodes, t_nodes))
    t_in = np.where(
        np.sign(mdot_signed) >= 0,
        [t_dict[u] for u, v in edges],
        [t_dict[v] for u, v in edges],
    )

    # Per-tank coefficients, as in the scalar model
    mdot = np.abs(mdot_signed)
    flow = mdot_signed != 0
    charging = mdot_signed > 0
    g_cond = (k_fluid + delta_k) * section_area / dz
    capacity = rho * layer_volume * cp
    adv = mdot * cp
    n_sub = np.maximum(1, np.ceil(mdot * stepsize / (rho * layer_volume)).astype(int))
    dt = stepsize / n_sub

    # Backward-Euler tridiagonal rows; padded layers get decoupled identity
    # rows (diag 1, no coupling, rhs 0)
    coupled = cols[:, : N - 1] < (n_layers - 1)[:, None]  # real interfaces
    lower = np.where(coupled, -g_cond[:, None], 0.0)
    upper = np.where(coupled, -g_cond[:, None], 0.0)
    diag = np.where(valid, (capacity / dt)[:, None] + ua, 1.0)
    diag[:, :-1] += np.where(coupled, g_cond[:, None], 0.0)
    diag[:, 1:] += np.where(coupled, g_cond[:, None], 0.0)
    rhs_const = ua * t_amb[:, None]

    # Advection from the upstream neighbour: downwards when charging,
    # upwards when discharging
    diag += np.where(valid & flow[:, None], adv[:, None], 0.0)
    lower -= np.where(coupled & charging[:, None], adv[:, None], 0.0)
    upper -= np.where(coupled & (flow & ~charging)[:, None], adv[:, None], 0.0)
    inlet_idx = np.where(charging, 0, n_layers - 1)
    rhs_const[rows, inlet_idx] += np.where(flow, adv * t_in, 0.0)

    out_idx = np.where(mdot_signed >= 0, n_layers - 1, 0)
    t_out_sum = np.zeros(S)
    for j in range(n_sub.max()):
        active = j < n_sub
        new = _batched_thomas(
            lower, diag, upper, (capacity / dt)[:, None] * temps + rhs_const
        )
        t_out_sum += np.where(active, new[rows, out_idx], 0.0)
        for s in np.where(active)[0]:
            n = n_layers[s]
            new[s, :n] = StratifiedStorage._mix_inversions(new[s, :n])
        temps = np.where(active[:, None], new, temps)

    # Outlet temperature and energy exchanged with the network (Wh)
    t_out = np.where(flow, t_out_sum / n_sub, temps[rows, n_layers - 1])
    t_in = np.where(flow, t_in, temps[:, 0])
    delta_q = np.where(flow, mdot * cp * (t_out - t_in) * stepsize / 3600.0, 0.0)

    # Write the new state back
    t_avg = np.empty(S)
    for s, c in enumerate(tanks):
        c._layer_temperatures = temps[s, : n_layers[s]].copy()
        t_avg[s] = c._layer_temperatures.mean()

    return t_in, t_out, t_avg, np.zeros(S), delta_q
