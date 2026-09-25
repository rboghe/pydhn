#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only


"""
Thermal model of the StratifiedStorage component.

The layers of all the tanks of a network are integrated together, in arrays
with one row per tank. Tanks with fewer layers are padded with zeros.
"""

import numpy as np


def _mix_inversions(temperatures, derivatives):
    """
    Removes temperature inversions by mixing the affected layers. Going from
    the top down, each layer warmer than the one above is mixed with it, until
    temperatures never increase downwards. As all layers have the same volume,
    the mixed temperature is their mean. The derivatives of the temperatures
    with respect to the inlet temperature are mixed in the same way.
    """
    # Each pool is a group of mixed layers, stored as the sum of their
    # temperatures, the sum of their derivatives and their number
    pools = []
    for t, d in zip(temperatures, derivatives):
        pool = [t, d, 1]
        # Mix with the pools above as long as they are colder on average
        while pools and pools[-1][0] * pool[2] < pool[0] * pools[-1][2]:
            above = pools.pop()
            pool = [pool[0] + above[0], pool[1] + above[1], pool[2] + above[2]]
        pools.append(pool)
    counts = [n for _, _, n in pools]
    mixed_temperatures = np.repeat([t / n for t, _, n in pools], counts)
    mixed_derivatives = np.repeat([d / n for _, d, n in pools], counts)
    return mixed_temperatures, mixed_derivatives


def _compute_storage_temperatures(tanks, fluid, t_in, ts_id=None):
    """
    Computes one time step for a list of tanks, given their inlet temperature,
    and updates their internal state. Returns t_in, t_out, t_avg, t_out_der and
    delta_q.

    The layer temperatures are integrated with the third-order strong
    stability preserving Runge-Kutta method (SSPRK3), together with their
    derivatives with respect to the inlet temperature. Each substep is at most
    0.1 times the shortest time constant of the layers. Inversions are mixed
    before the integration and after each Runge-Kutta stage.
    """
    S = len(tanks)
    n_layers = np.array([c["n_layers"] for c in tanks])
    N = n_layers.max()
    mdot = np.array([c["mass_flow"] for c in tanks])
    stepsize = np.array([c["stepsize"] for c in tanks])
    t_in = np.array(t_in, dtype=float)

    # The balance of each layer i is written as:
    #   dT_i/dt = source_i - diag_i * T_i + lower_i * T_i-1 + upper_i * T_i+1
    # where time is divided by the step size, so that it goes from 0 to 1. The
    # last axis of state and source holds temperatures and their derivatives.
    state = np.zeros((S, N, 2))
    source = np.zeros((S, N, 2))
    diag = np.zeros((S, N))
    lower = np.zeros((S, N - 1))
    upper = np.zeros((S, N - 1))
    cp = np.zeros(S)
    for s, c in enumerate(tanks):
        # If it is a repeated step, restore previous conditions, then update
        # the internal memory
        if c._last_ts is not None and c._last_ts == ts_id:
            c._layer_temperatures = c._last_layer_temperatures.copy()
        c._last_ts = ts_id
        c._last_layer_temperatures = c._layer_temperatures.copy()
        n = n_layers[s]
        state[s, :n, 0] = c._layer_temperatures

        # Fluid properties are evaluated at the mean tank temperature
        t_mean = c._layer_temperatures.mean()
        cp[s], rho, k = fluid.get_cp(t_mean), fluid.get_rho(t_mean), fluid.get_k(t_mean)
        scale = stepsize[s] / (rho * c._layer_volume * cp[s])

        # Conductances (W/K) of conduction between layers, advection and
        # losses, scaled by the step size and the heat capacity of a layer
        g = (k + c["delta_k"]) * c._section_area / c._dz * scale
        adv = abs(mdot[s]) * cp[s] * scale
        ua = c._ua_layers * scale

        lower[s, : n - 1] = g
        upper[s, : n - 1] = g
        diag[s, :n] = ua + adv
        diag[s, : n - 1] += g
        diag[s, 1:n] += g
        source[s, :n, 0] = ua * c["t_ambient"]

        # Water enters from the top with positive mass flows and from the
        # bottom with negative ones
        if mdot[s] != 0:
            inlet = 0 if mdot[s] > 0 else n - 1
            if mdot[s] > 0:
                lower[s, : n - 1] += adv
            else:
                upper[s, : n - 1] += adv
            source[s, inlet, 0] += adv * t_in[s]
            source[s, inlet, 1] = adv

    def rate(values):
        result = source - diag[:, :, None] * values
        result[:, 1:] += lower[:, :, None] * values[:, :-1]
        result[:, :-1] += upper[:, :, None] * values[:, 1:]
        return result

    def mix(values):
        for s, n in enumerate(n_layers):
            if np.any(np.diff(values[s, :n, 0]) > 0):
                values[s, :n, 0], values[s, :n, 1] = _mix_inversions(
                    values[s, :n, 0], values[s, :n, 1]
                )
        return values

    # Find the substep of each tank. Slow tanks use a single step.
    h_max = 0.1 / np.maximum(diag.max(axis=1), 0.1)
    rows = np.arange(S)
    out_idx = np.where(mdot >= 0, n_layers - 1, 0)
    outlet = np.zeros((S, 2))

    state = mix(state)
    for j in range(int(np.ceil(1 / h_max).max())):
        # Tanks that already reached the end of the step get a null substep
        h = np.clip(1.0 - j * h_max, 0.0, h_max)[:, None, None]
        k1 = rate(state)
        stage1 = mix(state + h * k1)
        k2 = rate(stage1)
        stage2 = mix(state + (stage1 - state + h * k2) / 4)
        k3 = rate(stage2)
        new_state = mix(state + 2 / 3 * (stage2 - state + h * k3))
        # Integrate the outlet temperature with the weights of SSPRK3, so that
        # the heat exchanged matches the change of stored energy
        stages = state[rows, out_idx] + stage1[rows, out_idx]
        outlet += h[:, 0] / 6 * (stages + 4 * stage2[rows, out_idx])
        state = np.where(h > 0, new_state, state)

    # The outlet temperature is its average over the time step. If the mass
    # flow is zero, use the temperatures of the top and bottom layers.
    flowing = mdot != 0
    t_out = np.where(flowing, outlet[:, 0], state[rows, n_layers - 1, 0])
    t_in = np.where(flowing, t_in, state[:, 0, 0])
    t_out_der = np.where(flowing, outlet[:, 1], 0.0)

    # Heat given to the network (Wh), negative when charging
    delta_q = np.abs(mdot) * cp * (t_out - t_in) * stepsize / 3600.0

    # Update layer temperatures
    t_avg = np.zeros(S)
    for s, c in enumerate(tanks):
        c._layer_temperatures = state[s, : n_layers[s], 0].copy()
        t_avg[s] = c._layer_temperatures.mean()

    return t_in, t_out, t_avg, t_out_der, delta_q


def compute_storage_temp_net(net, fluid, soil, ts_id=None):
    """
    Computes one time step for all the stratified storages of the network and
    updates their internal state. Returns t_in, t_out, t_avg, t_out_der and
    delta_q, following the order of the storages in the network.
    """
    mask = net.mask(attr="component_type", value="stratified_storage")
    edges = net.edges(mask=mask)
    tanks = [net[(u, v)] for u, v in edges]
    t_in = [
        net[v if c["mass_flow"] < 0 else u]["temperature"]
        for (u, v), c in zip(edges, tanks)
    ]
    return _compute_storage_temperatures(tanks, fluid, t_in, ts_id)
