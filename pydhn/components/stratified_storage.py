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


from numbers import Real

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
    layers at each Runge-Kutta stage. Fluid properties are evaluated once per
    step at the initial mean tank temperature. With variable properties this
    conserves the frozen-property balance of each step, not a global enthalpy
    inventory; use ConstantWater for a consistent constant-property balance.
    This is a fixed-inlet, two-port adaptation, without a
    separate plume-entrainment model. Integration uses third-order SSP
    Runge-Kutta steps limited by advection, conduction and envelope losses.
    The outlet is averaged over the step; its derivative includes mixing.
    The reported average tank temperature is the final spatial mean, not a
    time average, and cannot be used directly to integrate ambient losses.
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
        temperature=TEMPERATURE,
        initial_layer_temperatures=None,
        **kwargs,
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
            Positive number of equal-volume vertical layers. Cannot be
            changed after construction. The default is {}.
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
        temperature : float, optional
            Uniform initial fluid temperature (°C). The default is 50.
        initial_layer_temperatures : array-like, optional
            Initial temperatures from top to bottom, with one value per
            layer, in °C. Must have shape (n_layers,) and finite values.
            The default is None, which uses ``temperature`` for all layers.
            A supplied profile is copied and overrides the uniform value.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        None
            Initializes the component and its internal layer temperatures.

        Raises
        ------
        ValueError
            If a physical parameter or initial temperature profile is invalid,
            or the hydraulic setpoint type is not "mass_flow".

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
        # Extra attributes include the mass flow supplied by the network solver
        self._attrs.update(kwargs)
        self._attrs["temperature"] = temperature
        for key, value in self._attrs.items():
            self._validate_attribute(key, value)

        self._update_geometry()
        # Layer 0 is the top; copy supplied profiles so the tank owns its state
        if initial_layer_temperatures is None:
            temps = np.full(n_layers, temperature, dtype=float)
        else:
            temps = np.array(initial_layer_temperatures, dtype=float, copy=True)
            if temps.shape != (n_layers,) or not np.all(np.isfinite(temps)):
                raise ValueError(
                    "initial_layer_temperatures must contain one finite value per layer"
                )
        self._layer_temperatures = temps
        self._attrs["temperature"] = temps.mean()
        # Keep the start-of-step profile for repeated network iterations
        self._last_layer_temperatures = temps.copy()
        self._last_ts = None

    @staticmethod
    def _validate_attribute(key, value):
        if key == "setpoint_type_hyd":
            if value != "mass_flow":
                raise ValueError('Storage only supports setpoint_type_hyd="mass_flow"')
        elif key == "n_layers":
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < 1
            ):
                raise ValueError("n_layers must be a positive integer")
        elif key in {
            "volume",
            "height",
            "stepsize",
            "u_value",
            "delta_k",
            "temperature",
            "t_ambient",
            "mass_flow",
            "setpoint_value_hyd",
        }:
            if not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f"{key} must be a finite number")
            if key in {"volume", "height", "stepsize"} and value <= 0:
                raise ValueError(f"{key} must be positive")
            if key in {"u_value", "delta_k"} and value < 0:
                raise ValueError(f"{key} must be nonnegative")

    def set(self, key, value):
        """
        Update attributes; layer count is fixed after construction.

        Parameters
        ----------
        key : str
            Name of the component attribute to update.
        value : Any
            New attribute value. Physical parameters must satisfy the same
            constraints as during construction.

        Returns
        -------
        None
            Updates the component in place.

        Raises
        ------
        ValueError
            If the value is invalid or changes ``n_layers`` after construction.

        Notes
        -----
        Changes to ``volume``, ``height`` or ``u_value`` rebuild the cached
        geometry and loss conductances while retaining layer temperatures.
        Setting ``temperature`` updates reporting metadata only; it does not
        reset the internal temperature profile.
        """
        # Reject invalid updates before changing either attributes or cached state
        self._validate_attribute(key, value)
        if key == "n_layers" and hasattr(self, "_layer_temperatures"):
            if value != self._attrs[key]:
                raise ValueError("n_layers cannot be changed; construct a new tank")
        super().set(key, value)
        if key in {"volume", "height", "u_value"} and hasattr(self, "_ua_layers"):
            self._update_geometry()

    def _update_geometry(self):
        volume, height = self._attrs["volume"], self._attrs["height"]
        n_layers, u_value = self._attrs["n_layers"], self._attrs["u_value"]

        # Equal-volume slices of a vertical cylinder share a cross-section
        self._layer_volume = volume / n_layers
        self._dz = height / n_layers
        self._section_area = volume / height
        diameter = np.sqrt(4 * self._section_area / np.pi)
        # Envelope UA per layer; top and bottom layers also lose through the
        # tank ends
        ua = np.full(n_layers, u_value * np.pi * diameter * self._dz)
        ua[0] += u_value * self._section_area
        # For a single layer, both end caps belong to that same layer
        ua[-1] += u_value * self._section_area
        self._ua_layers = ua

    @staticmethod
    def _mix_inversions(temperatures, sensitivity=None):
        """
        Removes temperature inversions by mixing the affected layers: going
        from the top down, every layer warmer than the (mixed) layers above
        it is pooled with them at their energy-conserving mean, so that the
        returned profile is non-increasing.
        """
        # Each stack entry represents a contiguous pool of equal-volume layers
        sums, counts, derivatives = [], [], []
        for i, t in enumerate(temperatures):
            # Start with a single layer, then absorb any colder pools above it
            s, c = t, 1
            if sensitivity is not None:
                derivative = sensitivity[i]
            # Merge upwards until the pool is no warmer than the one above
            while sums and sums[-1] * c < s * counts[-1]:  # mean above < mean
                s += sums.pop()
                c += counts.pop()
                if sensitivity is not None:
                    derivative += derivatives.pop()
            sums.append(s)
            counts.append(c)
            if sensitivity is not None:
                derivatives.append(derivative)
        # Expand the pooled means back to one temperature per layer
        means = [s / c for s, c in zip(sums, counts)]
        mixed = np.repeat(means, counts)
        if sensitivity is None:
            return mixed
        # Differentiate the same pools; at a mixing boundary this is the
        # derivative of the selected piecewise-linear branch
        return mixed, np.repeat([d / c for d, c in zip(derivatives, counts)], counts)

    # ------------------------------ Hydraulics ----------------------------- #

    def _compute_delta_p(self, fluid, compute_hydrostatic=False, ts_id=None):
        # The imposed-flow ideal branch contributes no pressure-loss equation
        return 0.0, 0

    # ------------------------------- Thermal ------------------------------- #

    def _compute_temperatures(self, fluid, soil, t_in, ts_id=None):
        # A direct call without an ID advances one step. Network solves
        # supply a shared ID for all iterations of the same physical step
        from pydhn.components.stratified_storage_thermal import (
            _compute_storage_temperatures,
        )

        # Reuse the batch equations and unwrap their one-tank output arrays
        outputs = _compute_storage_temperatures([self], fluid, [t_in], ts_id)
        return tuple(values[0] for values in outputs)
