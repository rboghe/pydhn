#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only


"""
Implementation of the one-dimensional multi-node stratified storage tank
model ("multinode" model, TRNSYS Type 4 lineage) described in:

    Kleinbach, E. M., W. A. Beckman, and S. A. Klein.
    "Performance study of one-dimensional models for stratified thermal
    storage tanks."
    Solar Energy 50.2 (1993): 155-166.
"""


import numpy as np

from pydhn.components import Component
from pydhn.default_values import SETPOINT_TYPE_HYD_STORAGE
from pydhn.default_values import SETPOINT_VALUE_HYD_STORAGE
from pydhn.default_values import STEPSIZE
from pydhn.default_values import STORAGE_DELTA_K
from pydhn.default_values import STORAGE_HEIGHT
from pydhn.default_values import STORAGE_N_LAYERS
from pydhn.default_values import STORAGE_U_VALUE
from pydhn.default_values import STORAGE_VOLUME
from pydhn.default_values import T_AMBIENT
from pydhn.default_values import TEMPERATURE
from pydhn.utilities import docstring_parameters


def _solve_tridiagonal(lower, diag, upper, rhs):
    """
    Solves a tridiagonal system with the Thomas algorithm. ``lower`` and
    ``upper`` have one element less than ``diag``.
    """
    n = len(diag)
    d, r = diag.copy(), rhs.copy()
    for i in range(1, n):
        w = lower[i - 1] / d[i - 1]
        d[i] -= w * upper[i - 1]
        r[i] -= w * r[i - 1]
    x = np.empty(n)
    x[-1] = r[-1] / d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (r[i] - upper[i] * x[i + 1]) / d[i]
    return x


class StratifiedStorage(Component):
    """
    Class implementing the one-dimensional multi-node stratified storage
    tank ("multinode" model) studied in:

        Kleinbach, E. M., W. A. Beckman, and S. A. Klein.
        "Performance study of one-dimensional models for stratified thermal
        storage tanks."
        Solar Energy 50.2 (1993): 155-166.

    As an ideal leaf component: the start node connects to the top of the
    tank and the end node to its bottom. A positive mass flow setpoint
    charges the tank (hot water enters the top), a negative one discharges
    it. The tank is always full and each layer exchanges heat with its
    neighbours by advection and conduction, and with the ambient through the
    envelope. Temperature inversions are removed by mixing the affected
    layers. Fluid properties are evaluated once per step at the mean tank
    temperature.
    """

    @docstring_parameters(
        STORAGE_VOLUME,
        STORAGE_HEIGHT,
        STORAGE_N_LAYERS,
        STORAGE_U_VALUE,
        STORAGE_DELTA_K,
        T_AMBIENT,
        SETPOINT_TYPE_HYD_STORAGE,
        SETPOINT_VALUE_HYD_STORAGE,
        STEPSIZE,
    )
    def __init__(
        self,
        volume=STORAGE_VOLUME,
        height=STORAGE_HEIGHT,
        n_layers=STORAGE_N_LAYERS,
        u_value=STORAGE_U_VALUE,
        delta_k=STORAGE_DELTA_K,
        t_ambient=T_AMBIENT,
        setpoint_type_hyd=SETPOINT_TYPE_HYD_STORAGE,
        setpoint_value_hyd=SETPOINT_VALUE_HYD_STORAGE,
        stepsize=STEPSIZE,
        **kwargs
    ):
        """
        Constructs all the necessary attributes for the object.

        Parameters
        ----------
        volume : float, optional
            Water volume of the tank (m³). The default is {}.
        height : float, optional
            Height of the tank (m). The default is {}.
        n_layers : int, optional
            Number of vertical layers. The default is {}.
        u_value : float, optional
            Overall heat loss coefficient of the tank envelope (W/(m²·K)).
            The default is {}.
        delta_k : float, optional
            Additional vertical conductivity added to the fluid one to
            account for destratification, e.g. through the tank wall
            (W/(m·K)). The default is {}.
        t_ambient : float, optional
            Temperature of the ambient around the tank (°C). The default
            is {}.
        setpoint_type_hyd : str, optional
            Hydraulic setpoint type. Currently, the only supported option is
            "mass_flow". The default is "{}".
        setpoint_value_hyd : float, optional
            Imposed mass flow (kg/s): positive charges the tank from the
            start node, negative discharges it. The default is {}.
        stepsize : float, optional
            Size of a time-step (s). The default is {}.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        None.

        """

        super(StratifiedStorage, self).__init__()

        # Component class and type
        self._class = "leaf_component"
        self._type = "stratified_storage"
        self._is_ideal = True

        # Add new inputs
        input_dict = {
            "volume": volume,
            "height": height,
            "n_layers": n_layers,
            "u_value": u_value,
            "delta_k": delta_k,
            "t_ambient": t_ambient,
            "setpoint_type_hyd": setpoint_type_hyd,
            "setpoint_value_hyd": setpoint_value_hyd,
            "stepsize": stepsize,
        }

        self._attrs.update(input_dict)
        self._attrs.update(kwargs)

        # Compute useful characteristics
        self._layer_volume = volume / n_layers
        self._dz = height / n_layers
        self._section_area = volume / height
        diameter = np.sqrt(4 * self._section_area / np.pi)
        # Envelope UA per layer; top and bottom layers also lose through the
        # tank ends
        ua = np.full(n_layers, u_value * np.pi * diameter * self._dz)
        ua[0] += u_value * self._section_area
        ua[-1] += u_value * self._section_area
        self._ua_layers = ua

        # Tank internal status: layer 0 is the top (start-node side)
        self._layer_temperatures = np.full(n_layers, TEMPERATURE, dtype=float)

        # Tank internal status at the former time step
        self._last_layer_temperatures = self._layer_temperatures.copy()

        # Keep track of the last time step ID
        self._last_ts = None

    @staticmethod
    def _mix_inversions(temperatures):
        """
        Removes temperature inversions by mixing the affected layers: going
        from the top down, every layer warmer than the (mixed) layers above
        it is pooled with them at their energy-conserving mean, so that the
        returned profile is non-increasing.
        """
        sums, counts = [], []
        for t in temperatures:
            s, c = t, 1
            while sums and sums[-1] * c < s * counts[-1]:  # mean above < mean
                s += sums.pop()
                c += counts.pop()
            sums.append(s)
            counts.append(c)
        means = [s / c for s, c in zip(sums, counts)]
        return np.repeat(means, counts)

    # ------------------------------ Hydraulics ----------------------------- #

    def _compute_delta_p(self, fluid, compute_hydrostatic=False, ts_id=None):
        return 0.0, 0

    # ------------------------------- Thermal ------------------------------- #

    def _compute_temperatures(self, fluid, soil, t_in, ts_id=None):
        # If it is a repeated step, restore previous conditions
        if self._last_ts is not None:
            if self._last_ts == ts_id:
                self._layer_temperatures = self._last_layer_temperatures.copy()

        # Update internal memory
        self._last_ts = ts_id
        self._last_layer_temperatures = self._layer_temperatures.copy()

        # Get attributes
        stepsize = self._attrs["stepsize"]
        mdot = self._attrs["mass_flow"]
        t_amb = self._attrs["t_ambient"]
        n = self._attrs["n_layers"]

        temps = self._layer_temperatures

        # Fluid properties at the mean tank temperature (constant within the
        # step, so that the advection bookkeeping conserves energy exactly)
        t_mean = temps.mean()
        cp = fluid.get_cp(t_mean)
        rho = fluid.get_rho(t_mean)
        k_eff = fluid.get_k(t_mean) + self._attrs["delta_k"]

        # Coupling terms (W/K): conduction between adjacent layers, envelope
        # losses and advection
        g_cond = k_eff * self._section_area / self._dz
        capacity = rho * self._layer_volume * cp
        adv = np.abs(mdot) * cp

        # Sub-steps so that at most one layer volume is flushed per solve,
        # keeping the outlet temperature sequence physical
        n_sub = max(1, int(np.ceil(np.abs(mdot) * stepsize / (rho * self._layer_volume))))
        dt = stepsize / n_sub

        # Backward-Euler tridiagonal system: (C/dt + G) t_new = C/dt t_old + b
        # Advection enters each layer from its upstream neighbour; layer 0 is
        # the top, so charging (mdot > 0) flows downwards
        lower = np.full(n - 1, -g_cond)
        upper = np.full(n - 1, -g_cond)
        diag = capacity / dt + self._ua_layers
        diag[:-1] += g_cond  # one interface for the end layers, two for the
        diag[1:] += g_cond  # interior ones
        rhs_const = self._ua_layers * t_amb
        if mdot > 0:
            diag += adv
            lower -= adv
            rhs_const[0] += adv * t_in
        elif mdot < 0:
            diag += adv
            upper -= adv
            rhs_const[-1] += adv * t_in

        out_idx = n - 1 if mdot >= 0 else 0
        t_out_sum = 0.0
        for _ in range(n_sub):
            temps = _solve_tridiagonal(
                lower, diag, upper, capacity / dt * temps + rhs_const
            )
            # Outlet is sampled before buoyant mixing so that the reported
            # delta_q matches the tank internal energy change exactly
            t_out_sum += temps[out_idx]
            temps = self._mix_inversions(temps)

        # Outlet temperature and energy exchanged with the network (Wh)
        if mdot != 0:
            t_out = t_out_sum / n_sub
            delta_q = np.abs(mdot) * cp * (t_out - t_in) * stepsize / 3600.0
        else:
            t_out = temps[-1]
            t_in = temps[0]
            delta_q = 0.0

        t_avg = temps.mean()

        # Update internal memory
        self._layer_temperatures = temps

        return t_in, t_out, t_avg, 0.0, delta_q
