#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only


"""
Batched integration of fixed-inlet multinode storage tanks.

The scalar entry point uses the same kernel with a single tank. Layer
arithmetic is vectorized across tanks; inversion mixing is per tank.
"""

import numpy as np

from pydhn.components.stratified_storage import StratifiedStorage


def _compute_storage_temperatures(tanks, fluid, t_in, ts_id=None):
    """
    Integrate temperatures and inlet sensitivities without advancing retries.

    SSPRK3 advances the layer balances and outlet integrals together. Each
    substep is at most 0.1 times the fastest layer time constant, including
    advection, conduction and losses. Inversions are mixed at every RK stage.
    Only the final substep is shortened:
    changing the number of steps does not redistribute earlier mixing times.
    Properties are frozen at the initial mean temperature of each tank.
    """
    S = len(tanks)
    if not S:
        return tuple(np.empty(0) for _ in range(5))
    # Pad shorter tanks with zero rows that have no thermal coupling.
    n_layers = np.array([c["n_layers"] for c in tanks])
    N = n_layers.max()
    rows = np.arange(S)
    state = np.zeros((S, N, 2))  # temperature, dT/dT_in
    # The tridiagonal operator couples each layer only to its two neighbours.
    # lower[:, i] carries heat i -> i+1; upper[:, i] carries heat i+1 -> i.
    # diag stores removal rates; source supplies ambient and inlet heat.
    lower = np.zeros((S, N - 1))
    upper = np.zeros_like(lower)
    diag = np.zeros((S, N))
    source = np.zeros_like(state)
    mdot = np.array([c["mass_flow"] for c in tanks])
    stepsize = np.array([c["stepsize"] for c in tanks])
    t_in = np.array(t_in, dtype=float, copy=True)
    cp = np.empty(S)
    initial = []
    for s, c in enumerate(tanks):
        # Validate before committing any state, including other tanks' states
        for key in ("mass_flow", "stepsize", "t_ambient", "delta_k"):
            c._validate_attribute(key, c[key])
        if mdot[s] != 0 and not np.isfinite(t_in[s]):
            raise ValueError("Storage inlet temperature must be finite when flowing")
        # Retry from the same initial profile, not the preceding Newton iterate
        temps = (
            c._last_layer_temperatures
            if c._last_ts is not None and c._last_ts == ts_id
            else c._layer_temperatures
        )
        if not np.all(np.isfinite(temps)):
            raise ValueError("Storage layer temperatures must be finite")
        initial.append(temps.copy())
        n = n_layers[s]
        state[s, :n, 0] = temps
        mean = temps.mean()
        # Use the saved profile for properties too, so retries solve the same ODE
        cp[s], rho, k = fluid.get_cp(mean), fluid.get_rho(mean), fluid.get_k(mean)
        if not np.all(np.isfinite([cp[s], rho, k])) or cp[s] <= 0 or rho <= 0 or k < 0:
            raise ValueError(
                "Storage requires positive cp and rho and nonnegative conductivity"
            )
        capacity = rho * c._layer_volume * cp[s]
        # Conductances in W/K: vertical diffusion and heat carried by the flow
        g = (k + c["delta_k"]) * c._section_area / c._dz
        adv = abs(mdot[s]) * cp[s]
        # Rates are scaled to the full step, so integration time runs 0..1
        scale = stepsize[s] / capacity
        lower[s, : n - 1] = g * scale
        upper[s, : n - 1] = g * scale
        diag[s, :n] = (c._ua_layers + adv) * scale
        # End layers have one conductive interface; interior layers have two
        diag[s, : n - 1] += g * scale
        diag[s, 1:n] += g * scale
        source[s, :n, 0] = c._ua_layers * c["t_ambient"] * scale
        if mdot[s] != 0:
            # Positive flow travels down from the top; negative flow travels up
            inlet = 0 if mdot[s] > 0 else n - 1
            if mdot[s] > 0:
                lower[s, : n - 1] += adv * scale
            else:
                upper[s, : n - 1] += adv * scale
            source[s, inlet, 0] += adv * t_in[s] * scale
            # Differentiate the inlet forcing while keeping step properties fixed
            source[s, inlet, 1] = adv * scale

    def rate(values):
        # Apply the layer balance to temperatures and sensitivities together
        result = source - diag[:, :, None] * values
        result[:, 1:] += lower[:, :, None] * values[:, :-1]
        result[:, :-1] += upper[:, :, None] * values[:, 1:]
        return result

    # Resolve the fastest thermal time constant to 10%; this limits RK3 error
    # while remaining well inside its forward-Euler positivity bound.
    # The lower bound also caps h at one for slow or completely idle tanks
    h_max = 0.1 / np.maximum(diag.max(axis=1), 0.1)
    out_idx = np.where(mdot >= 0, n_layers - 1, 0)
    outlet = np.zeros((S, 2))
    real_interfaces = np.arange(N - 1) < (n_layers - 1)[:, None]

    def mix(values):
        # Ignore padding and skip profiles that already have warm water on top
        inverted = np.any(
            (np.diff(values[:, :, 0], axis=1) > 0) & real_interfaces, axis=1
        )
        for s in np.flatnonzero(inverted):
            n = n_layers[s]
            # Pool sensitivities with the same weights as their temperatures.
            values[s, :n, 0], values[s, :n, 1] = StratifiedStorage._mix_inversions(
                values[s, :n, 0], values[s, :n, 1]
            )
        return values

    for j in range(int(np.ceil(1 / h_max).max())):
        # Finished tanks get a zero step while the rest continue integrating
        h = np.minimum(h_max, np.maximum(0.0, 1.0 - j * h_max))
        step = h[:, None, None]
        # Stage 1: forward Euler, then let buoyancy remove unstable layers
        k1 = rate(state)
        stage1 = mix(state + step * k1)
        # Stage 2: combine the initial state with a second Euler prediction
        k2 = rate(stage1)
        stage2 = mix(state + (stage1 - state + step * k2) / 4)
        # Stage 3: the SSPRK3 weighted combination completes the substep
        k3 = rate(stage2)
        new = mix(state + 2 / 3 * (stage2 - state + step * k3))
        # Integrate the outlet with the same stage weights as the layer balance.
        # Mixing preserves total energy, so this quadrature still closes it
        outlet += (
            h[:, None]
            / 6
            * (state[rows, out_idx] + stage1[rows, out_idx] + 4 * stage2[rows, out_idx])
        )
        state = np.where(step > 0, new, state)

    # At zero flow, report the final port temperatures instead of time averages
    flowing = mdot != 0
    t_out = np.where(flowing, outlet[:, 0], state[rows, n_layers - 1, 0])
    t_in = np.where(flowing, t_in, state[:, 0, 0])
    t_out_der = np.where(flowing, outlet[:, 1], 0.0)
    # Network heat exchange in Wh: charging is negative, discharging positive
    delta_q = abs(mdot) * cp * (t_out - t_in) * stepsize / 3600.0
    t_avg = np.empty(S)
    # Commit only after the whole batch succeeds, retaining the retry snapshot
    for s, c in enumerate(tanks):
        c._last_ts = ts_id
        c._last_layer_temperatures = initial[s]
        c._layer_temperatures = state[s, : n_layers[s], 0].copy()
        t_avg[s] = c._layer_temperatures.mean()
    return t_in, t_out, t_avg, t_out_der, delta_q


def compute_storage_temp_net(net, fluid, soil, ts_id=None):
    """
    Compute one thermal step for every stratified storage in a network.

    Parameters
    ----------
    net : Network
        Network containing the tanks, their mass flows and upstream node
        temperatures. Each tank supplies its own ``stepsize`` in seconds.
    fluid : Fluid
        Working fluid. Heat capacity, density and conductivity are evaluated
        at each tank's initial mean temperature and held fixed for the step.
    soil : Soil
        Accepted for compatibility with the network thermal interface. Tank
        losses use each tank's ``t_ambient`` attribute instead of soil data.
    ts_id : int, optional
        Physical timestep identifier. Repeating a non-None ID recomputes the
        step from its saved initial state. The default is None, which advances
        one step per call. The network solver supplies IDs during iteration.

    Returns
    -------
    t_in : numpy.ndarray
        Inlet temperatures (°C), shape (n_storages,). At zero flow, these are
        the final top-layer temperatures.
    t_out : numpy.ndarray
        Time-averaged outlet temperatures (°C), shape (n_storages,). At zero
        flow, these are the final bottom-layer temperatures.
    t_avg : numpy.ndarray
        Mean layer temperatures at the end of the step (°C), shape
        (n_storages,). This spatial mean is not a time average and cannot
        directly supply an integrated ambient-loss balance.
    t_out_der : numpy.ndarray
        Dimensionless derivatives dT_out/dT_in, including inversion mixing,
        shape (n_storages,). Zero for idle tanks. At a mixing boundary, the
        derivative follows the selected mixing branch.
    delta_q : numpy.ndarray
        Heat transferred to the network during the step (Wh), shape
        (n_storages,). Negative when charging, positive when discharging,
        and zero at zero flow. Ambient losses are not reported separately.

    Raises
    ------
    ValueError
        If a tank's thermal inputs or layer temperatures are invalid, a
        flowing tank has a non-finite inlet temperature, or fluid properties
        are non-finite or physically invalid.

    Notes
    -----
    All arrays follow the network's stratified-storage mask order. A network
    without tanks returns five empty arrays. Successful calls update tank
    layer states and timestep snapshots in place, but do not write the
    returned values to the network's edge attributes.

    With temperature-dependent fluids, each step conserves energy using its
    frozen heat capacity and density. Re-evaluating these properties on the
    next step does not guarantee conservation of a global enthalpy inventory.
    ConstantWater avoids this approximation. Inversion mixing also assumes
    equal layer heat capacities. During a network thermal solve, idle edges
    temporarily receive signed epsilon mass flows. The tank then follows its
    normal flowing conventions, including a time-averaged outlet and negligible
    artificial advection at the solver's default epsilon magnitude.
    """
    mask = net.mask(attr="component_type", value="stratified_storage")
    edges = net.edges(mask=mask)
    tanks = [net[(u, v)] for u, v in edges]
    t_in = [
        net[u if c["mass_flow"] >= 0 else v]["temperature"]
        for (u, v), c in zip(edges, tanks)
    ]
    return _compute_storage_temperatures(tanks, fluid, t_in, ts_id)
