#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only


"""Class for base valve"""


from pydhn.components import Component
from pydhn.components.base_components_hydraulics import compute_dp_valve
from pydhn.default_values import KV
from pydhn.utilities import docstring_parameters


class BranchValve(Component):
    r"""
    Class for base BranchValve component. It implements a valve controlled by
    the flow coefficient :math:`K_v`:

    .. math::

        \Delta p = \frac{1.296 \cdot 10^9}{\rho K_v^2} \| \dot m \| \dot m

    The model assumes no temperature changes in the fluid across the valve.

    """

    @docstring_parameters(KV=KV)
    def __init__(
        self, kv: float = KV, dz: float = 0.0, line: str = None, **kwargs
    ) -> None:
        """
        Init BranchValve

        Parameters
        ----------
        kv : float, optional
            Flow coefficient :math:`K_v`. The default is {KV}.
        dz : float, optional
            Altitude difference (m) between the two ends of the component. The
            default is 0.0.
        line : str, optional
            Network line in which the component is placed. The default is None.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        None

        """
        super(BranchValve, self).__init__()

        # Component class and type
        self._class = "branch_component"
        self._type = "base_branch_valve"
        self._is_ideal = False

        # Add new inputs
        input_dict = {"kv": kv, "dz": dz, "line": line}

        self._attrs.update(input_dict)
        self._attrs.update(kwargs)

    # ------------------------------ Hydraulics ----------------------------- #

    def _compute_delta_p(
        self,
        fluid,
        compute_hydrostatic=False,
        compute_der=True,
        set_values=False,
        ts_id=None,
    ):
        # Compute the pressure losses
        rho_fluid = fluid.get_rho(self._attrs["temperature"])
        dp, dp_der = compute_dp_valve(
            mdot=self._attrs["mass_flow"],
            kv=self._attrs["kv"],
            dz=self._attrs["dz"],
            rho_fluid=rho_fluid,
            compute_hydrostatic=compute_hydrostatic,
            compute_der=compute_der,
            ts_id=ts_id,
        )
        if set_values:
            self.set("delta_p", dp)

        return dp, dp_der

    # ------------------------------- Thermal ------------------------------- #

    def _compute_temperatures(self, fluid, soil, t_in, ts_id=None):
        return t_in, t_in, t_in, 0.0, 0.0
