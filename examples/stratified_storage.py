#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
Example of a stratified storage tank in a small meshed network with Lagrangian
pipes. The tank is connected between the supply and the return line: its top
to the supply and its bottom to the return. It is first charged, then left
idle and finally discharged. In a second simulation, the tank is used to
reduce the peak power of the producer.
"""

import matplotlib.pyplot as plt
import numpy as np

from pydhn.fluids import ConstantWater
from pydhn.networks.load_networks import star_network
from pydhn.soils import Soil
from pydhn.solving import SimpleStep

# Load the star toy network
net = star_network()

# The working fluid is water with constant properties
fluid = ConstantWater()

# Initialize soil with thermal conductivity of 0.8 W/mK and temperature of 5°C
soil = Soil(k=0.8, temp=5)

# Short time steps are needed to follow the water in the short pipes (s)
stepsize = 30.0

# Replace the pipes with Lagrangian pipes of the same geometry
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

# Consumers have an imposed mass flow and cool the water by 30 K, while the
# producer heats it to 80°C
net.set_edge_attribute("mass_flow", "control_type", mask=net.consumers_mask)
net.set_edge_attribute("mass_flow", "setpoint_type_hyd", mask=net.consumers_mask)
net.set_edge_attribute("delta_t", "setpoint_type_hx", mask=net.consumers_mask)
net.set_edge_attribute(-30.0, "setpoint_value_hx", mask=net.consumers_mask)
net.set_edge_attribute("t_out", "setpoint_type_hx", mask=net.producers_mask)
net.set_edge_attribute(80.0, "setpoint_value_hx", mask=net.producers_mask)
net.set_edge_attribute(stepsize, "stepsize")

loop = SimpleStep(
    hydraulic_sim_kwargs={"error_threshold": 1e-6, "verbose": 0},
    thermal_sim_kwargs={"error_threshold": 1e-6, "verbose": 0},
    with_thermal=True,
)

# Warm up the pipes for two hours with a low demand
net.set_edge_attribute(0.05, "setpoint_value_hyd", mask=net.consumers_mask)
warmup_steps = int(2 * 3600 / stepsize)
for ts_id in range(-warmup_steps, 0):
    loop.execute(net=net, fluid=fluid, soil=soil, ts_id=ts_id)
without_storage = net.copy()

# Add a 1 m³ tank between S2 and R2, initially at 30°C
net.add_stratified_storage(
    name="tank",
    start_node="S2",
    end_node="R2",
    volume=1.0,
    height=2.0,
    n_layers=10,
    temperature=30.0,
    u_value=0.5,
    t_ambient=20.0,
    stepsize=stepsize,
)
tank = net[("S2", "R2")]
with_storage = net.copy()

# Charge for two hours, leave idle for one hour and discharge for two hours
net.set_edge_attribute(0.1, "setpoint_value_hyd", mask=net.consumers_mask)
time_hours = np.arange(0.0, 5 * 3600.0, stepsize) / 3600.0
mass_flows = np.select([time_hours < 2.0, time_hours < 3.0], [0.1, 0.0], -0.1)

# Run the simulation and store the layer temperatures after each step. Each
# step has its own ts_id, so that the tank advances once per step.
profiles = [tank._layer_temperatures]
for ts_id, mass_flow in enumerate(mass_flows):
    tank.set("setpoint_value_hyd", mass_flow)
    loop.execute(net=net, fluid=fluid, soil=soil, ts_id=ts_id)
    profiles.append(tank._layer_temperatures)
profiles = np.array(profiles)
hours = np.arange(len(profiles)) * stepsize / 3600.0

# Plot the mass flow and the temperature at three heights of the tank
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

# Peak shaving: consumers draw 0.05 kg/s, and 0.15 kg/s during a one hour
# peak. The tank is charged before the peak and discharged during it.
consumer_flows = np.where((time_hours >= 2.0) & (time_hours < 3.0), 0.15, 0.05)
storage_flows = np.select([time_hours < 2.0, time_hours < 3.0], [0.1, -0.1], 0.0)
producer_power = {}
for label, network in (
    ("Without storage", without_storage),
    ("With storage", with_storage),
):
    power = []
    for ts_id, consumer_flow in enumerate(consumer_flows):
        network.set_edge_attribute(
            consumer_flow, "setpoint_value_hyd", mask=network.consumers_mask
        )
        if label == "With storage":
            network[("S2", "R2")].set("setpoint_value_hyd", storage_flows[ts_id])
        results = loop.execute(net=network, fluid=fluid, soil=soil, ts_id=ts_id)
        # Convert the heat of the producer from Wh per step to kW
        heat = results["edges"]["delta_q"][0, network.producers_mask].sum()
        power.append(heat * 3600.0 / stepsize / 1000.0)
    producer_power[label] = np.array(power)

peak_without = producer_power["Without storage"].max()
peak_with = producer_power["With storage"].max()
reduction = 100.0 * (peak_without - peak_with) / peak_without
print(
    f"Producer peak: {peak_without:.1f} kW without storage, "
    f"{peak_with:.1f} kW with storage ({reduction:.1f}% reduction)."
)

# Plot the power of the producer
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
