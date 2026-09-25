#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only


"""
Implementation of the one-dimensional multi-node stratified storage tank
studied in:

    Kleinbach, E. M., W. A. Beckman, and S. A. Klein.
    "Performance study of one-dimensional models for stratified thermal
    storage tanks."
    Solar Energy 50.2 (1993): 155-166.
"""


import numpy as np

from pydhn.components import Component
from pydhn.components.stratified_storage_thermal import _compute_storage_temperatures
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
    Class implementing the one-dimensional multi-node stratified storage tank
    studied in:

        Kleinbach, E. M., W. A. Beckman, and S. A. Klein.
        "Performance study of one-dimensional models for stratified thermal
        storage tanks."
        Solar Energy 50.2 (1993): 155-166.

    As a leaf component. The start node is connected to the top of the tank
    and the end node to its bottom. A positive mass flow charges the tank (hot
    water enters from the top), while a negative one discharges it.

    The tank is divided in layers of equal volume, which exchange heat with
    their neighbours by advection and conduction, and with the ambient through
    the envelope. Layers warmer than the ones above are mixed with them. Fluid
    properties are evaluated once per time step, at the mean tank temperature.
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
        TEMPERATURE,
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
            Number of layers. It cannot be changed after construction. The
            default is {}.
        u_value : float, optional
            U-value of the tank envelope (W/(m²·K)). The default is {}.
        delta_k : float, optional
            Thermal conductivity added to that of the fluid between layers, for
            example to account for conduction through the tank wall (W/(m·K)).
            The default is {}.
        t_ambient : float, optional
            Temperature around the tank (°C). The default is {}.
        setpoint_type_hyd : str, optional
            Hydraulic setpoint type. Only "mass_flow" is supported. The default
            is "{}".
        setpoint_value_hyd : float, optional
            Imposed mass flow (kg/s), positive to charge the tank and negative
            to discharge it. The default is {}.
        stepsize : float, optional
            Size of a time-step (s). The default is {}.
        temperature : float, optional
            Initial temperature of all layers (°C). The default is {}.
        initial_layer_temperatures : array-like, optional
            Initial temperature of each layer from the top to the bottom (°C).
            If given, it replaces temperature. The default is None.
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
        for key, value in input_dict.items():
            self._validate(key, value)

        self._attrs.update(input_dict)
        self._attrs.update(kwargs)

        # Compute useful characteristics
        self._update_geometry()

        # Tank internal status: temperature of each layer from the top
        if initial_layer_temperatures is None:
            self._layer_temperatures = np.full(n_layers, float(temperature))
        else:
            self._layer_temperatures = np.array(initial_layer_temperatures, float)
        self._attrs["temperature"] = self._layer_temperatures.mean()

        # Tank internal status at the former time step
        self._last_layer_temperatures = self._layer_temperatures.copy()

        # Keep track of the last time step ID
        self._last_ts = None

    @staticmethod
    def _validate(key, value):
        if key == "setpoint_type_hyd" and value != "mass_flow":
            raise ValueError('Storages only support setpoint_type_hyd="mass_flow"')
        if key in ("volume", "height", "n_layers", "stepsize") and not value > 0:
            raise ValueError(f"{key} must be positive")
        if key in ("u_value", "delta_k") and not value >= 0:
            raise ValueError(f"{key} cannot be negative")

    def set(self, key, value):
        self._validate(key, value)
        # The number of layers is fixed by the size of the internal status
        if key == "n_layers" and value != self._attrs["n_layers"]:
            raise ValueError("n_layers cannot be changed after construction")
        super().set(key, value)
        if key in ("volume", "height", "u_value"):
            self._update_geometry()

    def _update_geometry(self):
        """
        Computes the volume, height and cross-sectional area of the layers,
        as well as the UA-value of their envelope (W/K). The tank is a
        vertical cylinder.
        """
        volume, height = self._attrs["volume"], self._attrs["height"]
        n_layers, u_value = self._attrs["n_layers"], self._attrs["u_value"]
        self._layer_volume = volume / n_layers
        self._dz = height / n_layers
        self._section_area = volume / height
        diameter = np.sqrt(4 * self._section_area / np.pi)
        # The top and bottom layers also lose heat through the ends of the tank
        self._ua_layers = np.full(n_layers, u_value * np.pi * diameter * self._dz)
        self._ua_layers[0] += u_value * self._section_area
        self._ua_layers[-1] += u_value * self._section_area

    # ------------------------------ Hydraulics ----------------------------- #

    def _compute_delta_p(self, fluid, compute_hydrostatic=False, ts_id=None):
        return 0.0, 0

    # ------------------------------- Thermal ------------------------------- #

    def _compute_temperatures(self, fluid, soil, t_in, ts_id=None):
        outs = _compute_storage_temperatures([self], fluid, [t_in], ts_id)
        return tuple(values[0] for values in outs)
