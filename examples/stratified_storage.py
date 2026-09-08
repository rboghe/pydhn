#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
Charge, idle and discharge a stratified tank connected to a small DHN.

The tank is a full cylinder divided into ten equal-volume, well-mixed layers.
Layer 0 is at the top and layer 9 at the bottom. Keeping separate temperatures
allows hot water to remain above colder water instead of instantly mixing the
whole tank. A moving transition between hot and cold layers appears during use.

The two tank ports connect the supply and return lines of the toy network:

    Charging (+mass flow)                  Discharging (-mass flow)
   
    Supply S2                              Supply S2
        │                                      ^
        v                                      │
    +---------+                            +---------+
    │ 0: top  │  hot water enters          │ 0: top  │  hot water leaves
    │    v    │                            │    ^    │
    │ layers  │                            │ layers  │
    │    v    │                            │    ^    │
    │9: bottom│  bottom water leaves       │9: bottom│  return water enters
    +---------+                            +---------+
        │                                      ^
        v                                      │
    Return R2                              Return R2

The same mass flow crosses both ports, so the tank never fills or empties.
The sign selects the flow direction; the inlet temperature determines whether
the water actually adds or removes heat. 

In the following example, the supply is heated to 80 degC and consumers lower 
its temperature by 30 K before returning it to the network.

At zero flow, no water enters or leaves. Heat still conducts between layers
and escapes through the envelope to the 20 degC ambient. If a lower layer
becomes warmer than the one above, buoyancy mixes the unstable layers.

Run this file to see the imposed flow schedule and three layer temperatures.
The tank starts uniformly at 30 degC: its top should warm before its bottom.
During discharge, stored hot water feeds the supply junction while cooler
return water enters from below. The idle interval shows the slower standby
evolution. This is an imposed schedule, rather than a tank controller.

All network pipes use the Lagrangian model: water parcels carry their thermal
history along the pipe, while the pipe wall also stores heat. Temperature
changes therefore reach downstream nodes after a transport delay. We use
10-second timesteps to resolve the short pipes in this toy network.
A shared warm-up establishes the initial pipe temperatures before the tank
is added; both peak-shaving cases start from copies of that same pipe state.
"""

import matplotlib.pyplot as plt
import numpy as np

from pydhn.fluids import ConstantWater
from pydhn.networks.load_networks import star_network
from pydhn.soils import Soil
from pydhn.solving import SimpleStep

# Use the same toy network as multistep_simulation.py.
net = star_network()
fluid = ConstantWater()
# Constant properties make the tank's energy accounting consistent across steps.
soil = Soil(k=0.8, temp=5)
stepsize = 30.0  # Seconds, shared by pipes, storage and heat exchangers.

# Replace the toy network's steady-state pipes, retaining their geometry.
# A 10 m pipe with 20 mm diameter holds about 3.1 kg of water: at 0.1 kg/s,
# its nominal transit time is 31 s.
for u, v in net.edges(mask=net.pipes_mask):
    pipe = net[(u, v)]
    net.add_lagrangian_pipe(
        name=pipe["name"],
        start_node=u,
        end_node=v,
        length=pipe["length"],
        diameter=pipe["diameter"],
        roughness=pipe["roughness"],
        line=pipe["line"],
        stepsize=stepsize,
    )

# Configure hydraulics first: each consumer draws 0.1 kg/s independently of
# temperature. The existing producer supplies the remaining network flow.
net.set_edge_attribute("mass_flow", "control_type", mask=net.consumers_mask)
net.set_edge_attribute("mass_flow", "setpoint_type_hyd", mask=net.consumers_mask)
net.set_edge_attribute(0.1, "setpoint_value_hyd", mask=net.consumers_mask)
# Configure heat exchange: consumers cool their water; the producer reheats it.
net.set_edge_attribute("delta_t", "setpoint_type_hx", mask=net.consumers_mask)
net.set_edge_attribute(-30.0, "setpoint_value_hx", mask=net.consumers_mask)
net.set_edge_attribute("t_out", "setpoint_type_hx", mask=net.producers_mask)
net.set_edge_attribute(80.0, "setpoint_value_hx", mask=net.producers_mask)

loop = SimpleStep(
    hydraulic_sim_kwargs={"error_threshold": 1e-6, "verbose": 0},
    thermal_sim_kwargs={"error_threshold": 1e-6, "verbose": 0},
    with_thermal=True,
)
net.set_edge_attribute(stepsize, "stepsize")
# Warm the initially 50 degC pipe water and walls at the comparison's low load.
# This shared two-hour pre-run is outside the plotted five-hour schedules;
# the pipe walls take longer to settle than the water's transit time.
net.set_edge_attribute(0.05, "setpoint_value_hyd", mask=net.consumers_mask)
warmup_steps = int(2 * 3600 / stepsize)
for ts_id in range(-warmup_steps, 0):
    loop.execute(net=net, fluid=fluid, soil=soil, ts_id=ts_id)
net.set_edge_attribute(0.1, "setpoint_value_hyd", mask=net.consumers_mask)

# Keep the warmed network without a tank for the peak-shaving comparison below.
without_storage = net.copy()

# Add the tank as a branch between an existing supply/return node pair.
# The start node always connects to the top, even when the flow is reversed.
net.add_stratified_storage(
    name="tank",
    start_node="S2",
    end_node="R2",
    volume=1.0,
    height=2.0,
    n_layers=10,
    temperature=30.0,  # Uniform initial state; a layer profile can also be supplied.
    u_value=0.5,  # Envelope heat transfer coefficient, W/(m2 K).
    t_ambient=20.0,
)
# All components must represent the same physical interval.
net.set_edge_attribute(stepsize, "stepsize")
tank = net[("S2", "R2")]
# Start that comparison from the same cold tank, before this first simulation.
with_storage = net.copy()

# Two hours charging, one hour idle, then two hours discharging.
# Positive flow enters the top; negative flow enters from the return line.
time_hours = np.arange(0.0, 5 * 3600.0, stepsize) / 3600.0
mass_flows = np.select(
    [time_hours < 2.0, time_hours < 3.0], [0.1, 0.0], default=-0.1
)

# Copy layer states after each step to retain the evolving temperature profile.
profiles = [tank._layer_temperatures.copy()]
for ts_id, mass_flow in enumerate(mass_flows):
    # Change the hydraulic target; SimpleStep solves flows before temperatures.
    tank.set("setpoint_value_hyd", mass_flow)
    # A new ID advances time. Internal Newton retries reuse it, so they do not
    # charge/discharge the tank multiple times within this physical interval.
    results = loop.execute(net=net, fluid=fluid, soil=soil, ts_id=ts_id)
    # This private array is read here only to illustrate the internal profile.
    profiles.append(tank._layer_temperatures.copy())

profiles = np.array(profiles)
hours = np.arange(len(profiles)) * stepsize / 3600.0

# Compare the imposed flow with the response at three heights in the tank.
# Profiles include the initial state at t=0 and each subsequent end-of-step state.
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
axes[0].step(hours, np.r_[mass_flows, mass_flows[-1]], where="post")
axes[0].axhline(0.0, color="grey", linewidth=0.8)
axes[0].set_ylabel("Mass flow (kg/s)")
axes[0].set_title("Stratified storage: charge, idle, discharge")
for layer, label in ((0, "Top"), (5, "Middle"), (9, "Bottom")):
    axes[1].plot(hours, profiles[:, layer], label=label)
axes[1].set_xlabel("Time (h)")
axes[1].set_ylabel("Temperature (°C)")
axes[1].legend()
fig.tight_layout()

# Peak shaving: compare the same demand with and without the storage branch.
# Each consumer draws 0.05 kg/s normally and 0.15 kg/s during a one-hour peak.
# Equal flows, a 30 K drop and ConstantWater give identical consumer heat loads
# in both runs. Producer power also reflects pipe heat losses and changes in
# the energy stored in the pipe water and walls, with delayed return temperatures.
consumer_flows = np.where((time_hours >= 2.0) & (time_hours < 3.0), 0.15, 0.05)
storage_flows = np.select(
    [time_hours < 2.0, time_hours < 3.0], [0.1, -0.1], default=0.0
)
producer_power = {}
for label, comparison in (
    ("Without storage", without_storage),
    ("With storage", with_storage),
):
    comparison.set_edge_attribute(stepsize, "stepsize")
    power = []
    for ts_id, consumer_flow in enumerate(consumer_flows):
        comparison.set_edge_attribute(
            consumer_flow, "setpoint_value_hyd", mask=comparison.consumers_mask
        )
        if label == "With storage":
            # Charge during low demand, discharge at the peak, then idle.
            comparison[("S2", "R2")].set(
                "setpoint_value_hyd", storage_flows[ts_id]
            )
        result = loop.execute(net=comparison, fluid=fluid, soil=soil, ts_id=ts_id)
        # Producer delta_q is energy per step in Wh: convert to average kW.
        heat = result["edges"]["delta_q"][0, comparison.producers_mask].sum()
        power.append(heat * 3600.0 / stepsize / 1000.0)
    producer_power[label] = np.array(power)

# Compare maxima over the WHOLE schedule, including the extra charging load.
# The tank shifts heat production to earlier hours; this is not an energy-saving
# or repeated-cycle calculation, since the final tank state differs from its start.
peak_without = producer_power["Without storage"].max()
peak_with = producer_power["With storage"].max()
reduction = 100.0 * (peak_without - peak_with) / peak_without
print(
    f"Producer peak: {peak_without:.1f} kW without storage, "
    f"{peak_with:.1f} kW with storage ({reduction:.1f}% reduction)."
)

fig_peak, ax_peak = plt.subplots(figsize=(8, 4))
for label, power in producer_power.items():
    ax_peak.step(hours, np.r_[power, power[-1]], where="post", label=label)
ax_peak.axvspan(2.0, 3.0, color="grey", alpha=0.15, label="Demand peak")
ax_peak.set_xlabel("Time (h)")
ax_peak.set_ylabel("Producer thermal power (kW)")
ax_peak.set_title(f"Peak shaving: {reduction:.1f}% lower producer peak")
ax_peak.set_ylim(bottom=0)
ax_peak.legend()
fig_peak.tight_layout()

if __name__ == "__main__":
    plt.show()
