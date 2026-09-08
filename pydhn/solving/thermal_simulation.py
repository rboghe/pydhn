#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only


"""Generic functions needed for the thermal simulations."""

# Avoid circular import for type hints
from typing import TYPE_CHECKING
from warnings import warn

import networkx as nx
import numpy as np

from pydhn.classes import Results
from pydhn.solving.temperature import compute_edge_temperatures
from pydhn.utilities import np_cache

if TYPE_CHECKING:
    from pydhn import Fluid
    from pydhn import Network
    from pydhn import Soil


def _fill_zero_mass_flow(net, edges, nodes, mass_flow, mass_flow_min=1e-16):
    """Replace zero flows with signed epsilon flows for `solve_thermal`.

    The thermal solver excludes edges whose mass flow is exactly zero. Each
    zero is therefore replaced by `+/- mass_flow_min`. The signs orient the
    zero-flow subgraph along Eulerian walks: whenever the walk enters a node on
    a real zero-flow edge, it also leaves it on another one. This avoids
    creating an artificial source or sink inside an idle subgraph.

    A zero-flow component with only even-degree nodes has an Eulerian circuit
    directly. For odd-degree nodes, a temporary `eulerian_node` is joined
    to every odd node, making every degree even and allowing a circuit to be
    found.  Removing the temporary edges turns that circuit into open paths
    whose endpoints are the original odd-degree nodes. Consequently, the
    epsilon perturbation is exactly mass-conserving only for closed zero-flow
    components; open paths require compatible live flow at their endpoints.


    Parameters
    ----------
    net : Network
        Network object.
    edges : Array
        Array of edges in the network. Since it is needed in solve_thermal(),
        not recomputing it saves some time.
    nodes : Array
        Array of nodes in the network. It is currently unused here but is kept
        for the data flow shared with solve_thermal().
    mass_flow : Array
        Array of mass flow values. Since it is needed in solve_thermal(),
        not recomputing it saves some time.
    mass_flow_min : float, optional
        Mass flow that is assigned instead of 0 so that the edge is not ignored
        in solve_thermal(). The default is 1e-16.

    Returns
    -------
    mass_flow : Array
        Array where every zero is replaced by mass_flow_min with a sign chosen
        from an Eulerian walk. Closed zero-flow components remain balanced;
        odd-degree components become open epsilon paths.

    """
    G = net._graph

    # Only exact zeros may be changed. Work on their undirected topology: the
    # stored edge direction is considered only after the Eulerian walk is made.
    zero_mdot_list = list(map(tuple, edges[np.where(mass_flow == 0.0)[0]]))
    S = G.edge_subgraph(zero_mdot_list).copy().to_undirected()

    # An Eulerian circuit exists when every node has even degree. Such a walk
    # pairs every arrival at a node with a departure, which gives the desired
    # signs for the real zero-flow edges.
    degrees = np.array(S.degree())
    indices = np.where(degrees[:, 1].astype(float) % 2 != 0)[0]
    if len(indices) != 0:
        # The number of odd-degree nodes is always even. Connecting a temporary
        # nonphysical Eulerian node to each one adds one edge at that node,
        # making all degrees even. It also joins otherwise disconnected open
        # components through the temporary node, so one circuit visits them all.
        new_node = "eulerian_node"
        new_edges = [("eulerian_node", n) for n in degrees[indices, 0]]
        S.add_edges_from(new_edges)
    else:
        # The real zero-flow graph is already Eulerian; any node is a valid
        # starting point for a closed circuit.
        new_node = next(iter(S.nodes()))

    # Orient every real zero edge in the direction in which the Eulerian walk
    # traverses it. Temporary edges only close open paths and are discarded.
    for u, v in nx.eulerian_circuit(S, source=new_node):
        if "eulerian_node" in [u, v]:
            continue

        # The network stores an edge in exactly one direction. A traversal that
        # matches it gets positive flow; a reverse traversal gets negative flow.
        pos_arr = np.where((edges == (u, v)).all(axis=1))[0]
        neg_arr = np.where((edges == (v, u)).all(axis=1))[0]
        assert len(pos_arr) + len(neg_arr) == 1
        if len(pos_arr) == 1:
            mass_flow[pos_arr[0]] = mass_flow_min
        elif len(neg_arr) == 1:
            mass_flow[neg_arr[0]] = mass_flow_min * -1
        else:
            raise ValueError("Repeated edge found.")

    # solve_thermal can now include every edge in its thermal balance.
    assert np.isin(0, mass_flow) == False
    return mass_flow


@np_cache(maxsize=48)
def _prepare_arrays(E, mass_flow):
    E = E * np.sign(mass_flow)
    node_mask = np.where(np.abs(E).sum(axis=1) != 0)[0]
    edge_mask = np.where(np.abs(E).sum(axis=0) != 0)[0]
    E = E[np.ix_(node_mask, edge_mask)]

    E_in = (E + np.abs(E)) / 2
    rows = np.argmax(E, axis=0)
    columns = np.argmin(E, axis=0)

    # Compute total inlet mass flow of each node
    mass_flow_in = E_in @ np.abs(mass_flow[edge_mask])

    return E, E_in, edge_mask, node_mask, rows, columns, mass_flow_in


def solve_thermal(
    net: "Network",
    fluid: "Fluid",
    soil: "Soil",
    error_threshold: float = 1e-6,
    max_iters: int = 100,
    damping_factor: float = 1,
    decreasing: bool = False,
    adaptive: bool = False,
    verbose: int = 1,
    mass_flow_min: float = 1e-16,
    ts_id: int = None,
    **kwargs,
) -> dict:
    """
    Runs a thermal simulation of the Network object. The method is based on
    solving the heat balance at each node of the network using the
    Newton-Raphson method.

    The function returns a dictionary with the results of the simulation and
    details on the convergence of each Newton step.
    Temperatures and heat losses of the input Network are also modified in
    place with the results of the simulation.

    Parameters
    ----------
    net : Network
        Network to be simulated.
    fluid : Fluid
        Working fluid to be used.
    soil : Soil
        Soil object to be used in the simulation.
    error_threshold : float, optional
        Maximum absolute node residual, measured as mass flow times
        temperature (kg·K/s). The default is 1e-6.
    max_iters : int, optional
        Maximum number of iterations for the solver. The default is 100.
    damping_factor : float, optional
        Damping factor for the Newton iterations. The default is 1.
    decreasing : bool, optional
        Whether to reduce the damping factor at each Newton iteration. The
        default is False.
    adaptive : bool, optional
        Whether to reduce the damping factor on plateau. The default is False.
    verbose : int, optional
        Controls the verbosity of the simulation. The default is 1.
    mass_flow_min : float, optional
        Mass flow (kg/s) used to approximate 0. The default is 1e-16.
        Signed replacements are temporarily exposed as edge attributes during
        component thermal evaluation. Original flows are restored afterward,
        including on exceptions. Components use their normal flowing behaviour,
        with negligible artificial advection at the default magnitude.
    ts_id : int, optional
        Specifies the ID of the current time-step. When omitted, advances
        to the next ID on this network (starting at zero). All Newton
        iterations share that ID so dynamic components advance only once.
        Omitting the ID emits a warning: each call starts a new timestep,
        even when retrying a failed solve. Pass the same explicit ID for
        repeated solves within one timestep, and align IDs with any
        time-dependent inputs.
    **kwargs : dict
        Additional simulation arguments, accepted for compatibility with
        simulation loops. They are not used by this solver.

    Returns
    -------
    results : Results
        Dictionary-like simulation results. ``history`` contains the
        convergence flag, final iteration index and node residual history.
        ``nodes`` contains temperature arrays of shape (1, net.n_nodes).
        ``edges`` contains temperature, inlet and outlet temperature,
        ``delta_t`` and ``delta_q`` arrays of shape (1, net.n_edges).
        Temperatures and temperature differences are in °C and K,
        respectively; ``delta_q`` uses each component's heat-exchange units.
        Node and edge labels are stored under their respective ``columns``
        keys. The network attributes and dynamic states are updated in place.

    """
    # Explicit IDs also anchor subsequent automatically numbered steps.
    if ts_id is None:
        # A new call may be a retry; automatic numbering cannot distinguish it.
        warn(
            "No ts_id supplied: this call is treated as a new timestep. "
            "Pass an explicit timestep ID when retrying a solve or performing "
            "multiple solves within one timestep, and align it with any "
            "time-dependent inputs.",
            stacklevel=2,
        )
        ts_id = getattr(net, "_thermal_ts", -1) + 1
    net._thermal_ts = ts_id

    # Get mass flow and temperatures
    edges, mass_flow = net.edges("mass_flow")
    nodes, t_nodes = net.nodes("temperature")
    zero_flow_mask = np.flatnonzero(mass_flow == 0)
    original_zero_flows = mass_flow[zero_flow_mask].copy()

    # Find direction of zero mass flow elements
    if len(zero_flow_mask):
        mass_flow = _fill_zero_mass_flow(
            net=net,
            edges=edges,
            nodes=nodes,
            mass_flow=mass_flow,
            mass_flow_min=mass_flow_min,
        )

    # Initialize matrices of the network graph oriented as the mass flow:
    #   - E -> Incidence matrix , without 0 mass flow elements
    #   - E_in -> Incidence matrix of inlet edges only built from E
    E = net.incidence_matrix

    arrays = _prepare_arrays(E, mass_flow)
    E, E_in, edge_mask, node_mask, rows, columns, mass_flow_in = arrays

    dim = len(node_mask)

    # Initialize damp and converged
    damp = damping_factor
    converged = False
    errors_list = []

    for k in range(max_iters):
        # Set new temperatures
        net.set_node_attributes(t_nodes, "temperature")

        # Components must see the same flow direction as the node balance.
        # Expose epsilon flows only during thermal evaluation; finally restores
        # hydraulic zeros before convergence checks, even on errors or Ctrl+C.
        try:
            if len(zero_flow_mask):
                net.set_edge_attributes(
                    mass_flow[zero_flow_mask], "mass_flow", mask=zero_flow_mask
                )
            t_in, t_out, t_avg, t_out_der, delta_q = compute_edge_temperatures(
                net, fluid, soil, set_values=True, ts_id=ts_id
            )
        finally:
            if len(zero_flow_mask):
                net.set_edge_attributes(
                    original_zero_flows, "mass_flow", mask=zero_flow_mask
                )

        jac = np.zeros((dim, dim))
        for n, (i, j) in enumerate(zip(rows, columns)):
            jac[i][j] = t_out_der[edge_mask][n] * np.abs(mass_flow)[edge_mask][n]

        jac -= np.diag(mass_flow_in)

        errors = np.zeros((dim, dim))
        for n, (i, j) in enumerate(zip(rows, columns)):
            errors[i][j] = t_out[edge_mask][n] * np.abs(mass_flow)[edge_mask][n]

        errors -= np.diag(mass_flow_in * t_nodes[node_mask])
        errors = errors.sum(axis=1)

        error = np.max(np.abs(errors))
        errors_list.append(error)

        # Check if the simulation is converged
        if verbose > 1:
            print(f"Error at iteration {k}: {error}")

        if error <= error_threshold:
            converged = True
            break

        delta_t = np.linalg.solve(jac, -errors)
        t_nodes[node_mask] += damp * delta_t

        # The damping factor is lowered at each iteration
        if decreasing:
            damp = damping_factor - damping_factor * (k / max_iters)

        # Reduce the damping factor if the error is not decreasing
        if adaptive:
            if k > 2 and k % 2 != 0:
                if error > errors_list[k - 1]:
                    damp -= damp * (k / max_iters)
                else:
                    damp = damping_factor

    if verbose > 0:
        if converged:
            msg = f"Thermal simulation converged after {k} iterations with "
            msg += f"an error of {error} °C"
            print(msg)
        else:
            msg = "Thermal simulation not converged with an error of "
            msg += f"{error} °C!"
            warn(msg)

    results = Results(
        {
            "history": {
                "thermal converged": converged,
                "thermal iterations": k,
                "thermal errors": errors_list,
            }
        }
    )

    delta_t = (t_out - t_in) * np.sign(mass_flow)

    net.set_edge_attributes(delta_t, "delta_t")

    # Add column entry to results
    _, names = net.edges(data="name")
    results["edges"] = {}
    results["edges"]["columns"] = names
    names, _ = net.nodes()
    results["nodes"] = {}
    results["nodes"]["columns"] = names

    # Store node and edge results:
    # Nodes
    data = ["temperature"]
    arrays = tuple(net.nodes(data=data))
    d = {"nodes": {k: v[None] for k, v in zip(data, arrays[1:])}}
    # Edges
    data = [
        "temperature",
        "inlet_temperature",
        "outlet_temperature",
        "delta_t",
        "delta_q",
    ]
    arrays = tuple(net.edges(data=data))
    d.update({"edges": {k: v[None] for k, v in zip(data, arrays[1:])}})

    # Store results
    results.append(d)

    return results
