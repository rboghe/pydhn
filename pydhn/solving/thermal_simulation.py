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
from scipy import sparse
from scipy.sparse.linalg import spsolve

from pydhn.classes import Results
from pydhn.solving.temperature import compute_edge_temperatures
from pydhn.utilities import np_cache

if TYPE_CHECKING:
    from pydhn import Fluid
    from pydhn import Network
    from pydhn import Soil

# Components with an internal state that evolves at each time step
DYNAMIC_COMPONENTS = ["lagrangian_pipe", "stratified_storage"]


def _fill_zero_mass_flow(net, edges, mass_flow, mass_flow_min=1e-16):
    """
    The function solve_thermal() ignores all edges where the mass flow is zero.
    In order to avoid this, these values need to be temporarily replaced with a
    very low mass flow. The sign of the new mass flow must be such that, if the
    edges of the network graph are assigned the direction in which the mass
    flow is positive, no nodes have only incoming or outgoing edges.

    The signs follow an Eulerian circuit of the zero flow edges: every time
    the circuit enters a node, it also leaves it. Nodes with an odd number of
    zero flow edges are first connected to a temporary node, so that a
    circuit always exists. They become the ends of open paths.

    Parameters
    ----------
    net : Network
        Network object.
    edges : Array
        Array of edges in the network. Since it is needed in solve_thermal(),
        not recomputing it saves some time.
    mass_flow : Array
        Array of mass flow values. Since it is needed in solve_thermal(),
        not recomputing it saves some time.
    mass_flow_min : float, optional
        Mass flow that is assigned instead of 0 so that the edge is not ignored
        in solve_thermal(). The default is 1e-16.

    Returns
    -------
    mass_flow : Array
        Array of mass flow values where all the 0s are converted to a value
        equal to mass_flow_min times either 1 or -1, in a way that no
        converging or diverging nodes are created.

    """
    # Find direction of zero mass flow elements such that
    G = net._graph

    zero_mdot_list = list(map(tuple, edges[np.where(mass_flow == 0.0)[0]]))
    S = G.edge_subgraph(zero_mdot_list).copy().to_undirected()
    degrees = np.array(S.degree())
    indices = np.where(degrees[:, 1].astype(float) % 2 != 0)[0]
    if len(indices) != 0:
        new_node = "eulerian_node"
        new_edges = [("eulerian_node", n) for n in degrees[indices, 0]]
        S.add_edges_from(new_edges)
    else:
        # All degrees are even: the circuit can start from any node
        new_node = next(iter(S.nodes()))
    for u, v in nx.eulerian_circuit(S, source=new_node):
        if "eulerian_node" in [u, v]:
            continue
        pos_arr = np.where((edges == (u, v)).all(axis=1))[0]
        neg_arr = np.where((edges == (v, u)).all(axis=1))[0]
        assert len(pos_arr) + len(neg_arr) == 1
        if len(pos_arr) == 1:
            mass_flow[pos_arr[0]] = mass_flow_min
        elif len(neg_arr) == 1:
            mass_flow[neg_arr[0]] = mass_flow_min * -1
        else:
            raise ValueError("Repeated edge found.")
    assert not np.isin(0, mass_flow)
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
        Error threshold for the solver, as the maximum imbalance of mass flow
        times temperature in nodes (kg·K/s). The default is 1e-6.
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
    ts_id : int, optional
        Specifies the ID of the current time-step. Dynamic components restore
        their initial state when the same ID is repeated. If None and the
        network has dynamic components, the ID following the last one used is
        taken and a warning is raised. The default is None.
    **kwargs
        Arbitrary keyword arguments.

    Returns
    -------
    dict
        Dictionary with the simulation results.

    """
    # Get mass flow and temperatures
    edges, mass_flow = net.edges("mass_flow")
    nodes, t_nodes = net.nodes("temperature")

    # All the iterations of a time step must share the same ID, otherwise
    # dynamic components would advance at each iteration
    types = net.get_edges_attribute_array("component_type")
    dynamic = np.isin(types, DYNAMIC_COMPONENTS)
    if dynamic.any():
        if ts_id is None:
            msg = "No ts_id given: the thermal simulation is treated as a new "
            msg += "time step. Pass the same ts_id to repeat a time step."
            warn(msg, stacklevel=2)
            ts_id = getattr(net, "_last_ts_id", -1) + 1
        net._last_ts_id = ts_id

    # Dynamic components with zero mass flow compute their outlet temperature
    # at the end given by the sign of the new values, so they temporarily
    # receive them as well
    idle_dynamic = np.flatnonzero((mass_flow == 0) & dynamic)

    # Find direction of zero mass flow elements
    if np.any(mass_flow == 0):
        mass_flow = _fill_zero_mass_flow(
            net=net,
            edges=edges,
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

        # Compute temperature in edges. The zero mass flows of dynamic
        # components are restored even if the computation fails.
        idle_mass_flow = mass_flow[idle_dynamic]
        net.set_edge_attributes(idle_mass_flow, "mass_flow", mask=idle_dynamic)
        try:
            t_in, t_out, t_avg, t_out_der, delta_q = compute_edge_temperatures(
                net, fluid, soil, set_values=True, ts_id=ts_id
            )
        finally:
            zeros = np.zeros_like(idle_mass_flow)
            net.set_edge_attributes(zeros, "mass_flow", mask=idle_dynamic)

        # Compute the error of the heat balance in each node
        flows = np.abs(mass_flow[edge_mask])
        errors = np.bincount(rows, t_out[edge_mask] * flows, minlength=dim)
        errors -= mass_flow_in * t_nodes[node_mask]

        error = np.max(np.abs(errors))
        errors_list.append(error)

        # Check if the simulation is converged
        if verbose > 1:
            print(f"Error at iteration {k}: {error}")

        if error <= error_threshold:
            converged = True
            break

        # Solve the Newton step. Sparse matrices are faster above about 100
        # nodes, while dense ones are faster for smaller networks.
        jac_values = t_out_der[edge_mask] * flows
        if dim < 100:
            jac = np.zeros((dim, dim))
            jac[rows, columns] = jac_values
            delta_t = np.linalg.solve(jac - np.diag(mass_flow_in), -errors)
        else:
            jac = sparse.csc_matrix((jac_values, (rows, columns)), shape=(dim, dim))
            delta_t = spsolve((jac - sparse.diags(mass_flow_in)).tocsc(), -errors)
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
            msg += f"an error of {error} kg·K/s"
            print(msg)
        else:
            msg = "Thermal simulation not converged with an error of "
            msg += f"{error} kg·K/s!"
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
