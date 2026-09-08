#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only


"""
Functions to compute the temperature in the different components of the
Network.
"""

import numpy as np

from pydhn.components.vector_functions import COMPONENT_FUNCTIONS_DICT


def compute_edge_temperatures(
    net, fluid, soil, set_values=False, mask=None, ts_id=None
):
    """
    Computes the inlet, outlet and average temperature in each component, as
    well as the derivative dT_out/dT_in.

    Parameters
    ----------
    net : Network
        Network with component mass flows and node temperatures already set.
    fluid : Fluid
        Working fluid used to evaluate component thermophysical properties.
    soil : Soil
        Soil model used by components that exchange heat with the ground.
    set_values : bool, optional
        Whether to write inlet, outlet and average temperatures and
        ``delta_q`` to the selected edge attributes. The default is False.
        Dynamic components update their internal states in either case.
    mask : numpy.ndarray of int, optional
        Unique edge indices to evaluate. The default is None, which selects
        all edges. Unselected components and their attributes are unchanged.
    ts_id : int, optional
        Timestep identifier forwarded to component models. Dynamic storage
        components restore their initial state when a non-None ID is repeated.
        The default is None; this function does not generate timestep IDs.

    Returns
    -------
    t_in : numpy.ndarray
        Upstream node temperatures (°C), shape (net.n_edges,). Values are
        provided for every edge, including edges outside ``mask``.
    t_out : numpy.ndarray
        Component outlet temperatures (°C), shape (net.n_edges,). Dynamic
        models may return averages over the timestep.
    t_avg : numpy.ndarray
        Component average temperatures (°C), shape (net.n_edges,). For storage
        tanks, these are spatial averages at the end of the timestep.
    t_out_der : numpy.ndarray
        Dimensionless outlet derivatives dT_out/dT_in, shape (net.n_edges,).
    delta_q : numpy.ndarray
        Component-reported heat exchange, shape (net.n_edges,). Storage tanks,
        Lagrangian pipes and heat exchangers report Wh per step; steady-state
        pipes report W. The physical meaning follows the component model.

    Notes
    -----
    Returned arrays use the full network edge order. Entries outside ``mask``
    are zero in all outputs except ``t_in``. Evaluating temperatures can
    advance dynamic component states even when ``set_values`` is False.
    """
    # If a mask is not specified, all edges are considered
    if mask is None:
        mask = np.arange(net.n_edges)
    # Initialize output arrays as full of zeros
    t_in = np.zeros(net.n_edges)
    t_out = np.zeros(net.n_edges)
    t_avg = np.zeros(net.n_edges)
    t_out_der = np.zeros(net.n_edges)
    delta_q = np.zeros(net.n_edges)

    # Get original index of edges
    edges_orig, mass_flow_orig = net.edges("mass_flow")

    # Iterate over unique component types
    component_types = net.get_edges_attribute_array("component_type")
    component_types = np.unique(component_types[mask])
    for component in component_types:
        component_mask = net.mask(
            attr="component_type", value=component, condition="equality"
        )
        component_count = len(component_mask)
        component_mask = np.intersect1d(mask, component_mask, assume_unique=True)
        # If a vector function is specified for the component, use it
        has_vector = False
        if COMPONENT_FUNCTIONS_DICT[component] is not None:
            if "temperatures" in COMPONENT_FUNCTIONS_DICT[component].keys():
                has_vector = True
        # Whole-network vector functions cannot safely update a partial mask.
        if has_vector and len(component_mask) == component_count:
            foo = COMPONENT_FUNCTIONS_DICT[component]["temperatures"]
            outs = foo(net=net, fluid=fluid, soil=soil, ts_id=ts_id)
            t_in[component_mask] = outs[0]
            t_out[component_mask] = outs[1]
            t_avg[component_mask] = outs[2]
            t_out_der[component_mask] = outs[3]
            delta_q[component_mask] = outs[4]
        # Otherwise, compute the value for each component of that type
        # separately
        else:
            for i, (u, v) in enumerate(edges_orig[component_mask]):
                mdot = net[(u, v)]["mass_flow"]
                if np.sign(mdot) >= 0.0:
                    t_in_n = net[u]["temperature"]
                else:
                    t_in_n = net[v]["temperature"]
                outs = net[(u, v)]._compute_temperatures(
                    fluid, soil, t_in=t_in_n, ts_id=ts_id
                )
                idx = component_mask[i]
                t_in[idx] = outs[0]
                t_out[idx] = outs[1]
                t_avg[idx] = outs[2]
                t_out_der[idx] = outs[3]
                delta_q[idx] = outs[4]
    # Get temperature at startnode and endnode of HX
    nodes, t_nodes = net.nodes(data="temperature")
    u, v = edges_orig.T
    nodes_t_dict = dict(zip(nodes, t_nodes))
    t_0 = np.fromiter((nodes_t_dict[n] for n in u), dtype=float)
    t_1 = np.fromiter((nodes_t_dict[n] for n in v), dtype=float)

    t_in = np.where(mass_flow_orig >= 0, t_0, t_1)

    if set_values:
        net.set_edge_attributes(t_in[mask], "inlet_temperature", mask=mask)
        net.set_edge_attributes(t_out[mask], "outlet_temperature", mask=mask)
        net.set_edge_attributes(t_avg[mask], "temperature", mask=mask)
        net.set_edge_attributes(delta_q[mask], "delta_q", mask=mask)

    return t_in, t_out, t_avg, t_out_der, delta_q
