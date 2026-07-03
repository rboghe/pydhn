#!/usr/bin/env python
# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute, EPFL
#
# SPDX-FileContributor: Roberto Boghetti <roberto.boghetti@idiap.ch>
#
# SPDX-License-Identifier: AGPL-3.0-only

"""
Integration test on the OpenDHN benchmark: a meshed, two-producer,
150-substation district heating network with sensor measurements.

It tests the full steady-state pipeline on a real, meshed, multi-source network.

Benchmark data (tests/data/opendhn-data.zip), CC-BY-4.0, from:
    Boghetti, R., & Kämpf, J. H. (2024). OpenDHN data (1.0.1) [Data set].
    Zenodo. https://doi.org/10.5281/zenodo.10793816
Methodology: https://gitlab.idiap.ch/eguzki/opendhn
"""

import io
import os
import unittest
import zipfile

import numpy as np
import pandas as pd
from scipy.stats.mstats import trimmed_std

from pydhn.classes import Network
from pydhn.fluids import Water
from pydhn.soils import Soil
from pydhn.solving import SimpleStep
from pydhn.utilities.loading import add_nodes_from_dataframe

DIR = os.path.dirname(__file__)
ZIP = os.path.join(DIR, "data", "opendhn-data.zip")
CASE = "CASE1"


def _read(z, name, **kwargs):
    return pd.read_csv(io.BytesIO(z.read(name)), **kwargs)


def build_network():
    """Build the OpenDHN CASE1 network from the vendored benchmark data."""
    z = zipfile.ZipFile(ZIP)
    nodes = _read(z, "network/nodes.csv")
    pipes = _read(z, "network/pipes.csv")
    substations = _read(z, "network/substations.csv")
    hstations = _read(z, "network/heating_stations.csv")
    mass_flow = _read(z, "data/mass_flow.csv", index_col=0)
    supply_t = _read(z, "data/supply_temperature.csv", index_col=0)
    power = _read(z, "data/power.csv", index_col=0)

    net = Network()
    add_nodes_from_dataframe(
        net=net, df=nodes, name_col="node_id", x_col="x", y_col="y", z_col="z"
    )
    for i in pipes.index:
        p = pipes.loc[i]
        net.add_pipe(
            name=p["pipe_id"],
            start_node=p["inlet_node"],
            end_node=p["outlet_node"],
            length=p["length"],
            diameter=p["d_int"],
            internal_pipe_thickness=p["t_int"],
            insulation_thickness=p["t_ins"],
            casing_thickness=p["t_ext"],
            k_insulation=p["lambda_ins"],
            roughness=p["roughness"],
            depth=0.8,
            line="supply" if p["is_supply"] else "return",
            discretization=1,
        )
    # Substations impose the measured mass flow (hydraulic) and the measured
    # power as an extracted heat load (thermal).
    for i in substations.index:
        s = substations.loc[i]
        net.add_consumer(
            name=s["sub_id"],
            start_node=s["inlet_node"],
            end_node=s["outlet_node"],
            setpoint_type_hyd="mass_flow",
            setpoint_value_hyd=mass_flow.loc[CASE, s["sub_id"]],
            setpoint_type_hx="delta_q",
            setpoint_value_hx=-power.loc[CASE, s["sub_id"]],
            control_type="mass_flow",
            mass_flow_min=0,
        )
    # First heating station is the pressure slack; the other imposes its
    # measured mass flow. Both impose their measured supply temperature.
    for i in hstations.index:
        h = hstations.loc[i]
        if i == 0:
            hyd_type, hyd_value = "pressure", -100000
        else:
            hyd_type, hyd_value = "mass_flow", mass_flow.loc[CASE, h["hs_id"]]
        net.add_producer(
            name=h["hs_id"],
            start_node=h["inlet_node"],
            end_node=h["outlet_node"],
            setpoint_type_hyd=hyd_type,
            setpoint_value_hyd=hyd_value,
            setpoint_type_hx="t_out",
            setpoint_value_hx=supply_t.loc[CASE, h["hs_id"]],
        )
    return net


class OpenDHNTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        z = zipfile.ZipFile(ZIP)
        cls.supply_t = _read(z, "data/supply_temperature.csv", index_col=0)
        cls.return_t = _read(z, "data/return_temperature.csv", index_col=0)
        cls.net = build_network()
        results = SimpleStep(
            hydraulic_sim_kwargs={"error_threshold": 50},
            thermal_sim_kwargs={"error_threshold": 1e-12},
            with_thermal=True,
        ).execute(net=cls.net, fluid=Water(), soil=Soil(temp=-2.5, k=2.4))
        cls.results = results
        columns = results["edges"]["columns"]
        cls.inlet_t = pd.DataFrame(
            results["edges"]["inlet_temperature"], columns=columns, index=[CASE]
        )

        # Summary statistics against the measurements, following the benchmark
        # paper's Table 2: the two heating-station return-temperature errors and
        # the median / 5%-trimmed std / 5th & 95th percentiles of the substation
        # supply-temperature error.
        subs = [f"S{i}" for i in range(150)]
        err = cls.inlet_t.loc[CASE, subs].astype(float) - cls.supply_t.loc[
            CASE, subs
        ].astype(float)
        cls.stats = {
            "HS0": float(cls.inlet_t.loc[CASE, "HS0"] - cls.return_t.loc[CASE, "HS0"]),
            "HS1": float(cls.inlet_t.loc[CASE, "HS1"] - cls.return_t.loc[CASE, "HS1"]),
            "median": float(np.median(err)),
            "trimmed_std": float(trimmed_std(err.values, limits=[0.05, 0.05])),
            "pct5": float(err.quantile(0.05)),
            "pct95": float(err.quantile(0.95)),
        }
        s = cls.stats
        print(
            "\nOpenDHN CASE1 - PyDHN vs benchmark paper (Table 2):"
            f"\n  HS0 error              {s['HS0']:+6.2f} degC  (paper  +0.37)"
            f"\n  HS1 error              {s['HS1']:+6.2f} degC  (paper  +0.24)"
            f"\n  substation median      {s['median']:+6.2f} degC  (paper  0.07)"
            f"\n  substation trimmed std {s['trimmed_std']:6.2f} degC  (paper  1.07)"
            f"\n  substation 5th pct     {s['pct5']:+6.2f} degC  (paper -2.61)"
            f"\n  substation 95th pct    {s['pct95']:+6.2f} degC  (paper  +2.72)"
        )

    def test_converged(self):
        """Both the hydraulic and thermal solves reach their thresholds."""
        history = self.results["history"]
        self.assertTrue(history["hydraulics converged"])
        self.assertTrue(history["thermal converged"])

    def test_mass_conservation(self):
        """Net mass flow into every node is zero (incidence @ mass_flow = 0)."""
        _, mass_flow = self.net.edges("mass_flow")
        residual = self.net.incidence_matrix @ mass_flow
        np.testing.assert_allclose(residual, 0.0, atol=1e-6)

    def test_heating_station_accuracy(self):
        """Heating-station return-temperature errors stay near the paper values."""
        self.assertLess(abs(self.stats["HS0"]), 0.4)  # paper 0.37
        self.assertLess(abs(self.stats["HS1"]), 0.3)  # paper 0.24

    def test_substation_accuracy(self):
        """
        Substation supply-temperature error statistics match the benchmark
        paper (Table 2), with a margin over the reported values.
        """
        self.assertLess(abs(self.stats["median"]), 0.1)  # paper  0.07
        self.assertLess(self.stats["trimmed_std"], 1.2)  # paper  1.07
        self.assertGreater(self.stats["pct5"], -3.)  # paper -2.61
        self.assertLess(self.stats["pct95"], 3.)  # paper  2.72


if __name__ == "__main__":
    unittest.main()
