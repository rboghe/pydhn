Change Log
==========

0.2.0
-----

Unreleased

Added
~~~~~

* **Stratified storage:** Added the ``StratifiedStorage`` component, a dynamic thermal storage tank based on the one-dimensional multi-node model studied by `Kleinbach et al. <https://doi.org/10.1016/0038-092X(93)90087-5>`_, together with ``Network.add_stratified_storage`` and the example ``examples/stratified_storage.py``.
* **Vectorized Lagrangian pipe:** Added a vectorized thermal model of ``LagrangianPipe``, which network simulations use automatically. It gives the same results as computing each pipe separately.
* Components can list the attributes handled by their control logic in ``_controlled_keys``, so that looking up other attributes does not call it.
* Added a new test (``tests/test_dynamic_thermal_balance.py``) to verify thermal energy balance in dynamic simulations.
* Added a new example (``examples/dynamic_simulation.py``) for dynamic simulations with ``LagrangianPipe``.
* Added tests for dimensionless numbers, node pressures, zero mass flows, network matrices and thermal simulations, as well as the OpenDHN benchmark.
* Added the change log to the documentation.
* **Continuous integration:** Tests now run on Python 3.10 to 3.13, with the lowest and highest supported versions of the dependencies, and weekly with their latest versions.
* **Developer Guidelines:** Added ``CONTRIBUTING.md``.
* **Line Ending Standardization:** Added ``.gitattributes`` to enforce consistent line endings (LF) across text files.

Changed
~~~~~~~

* **Dependencies:** Added support for NumPy 2 and NetworkX 3, and for Python 3.10 to 3.13. Python 3.9 is no longer supported.
* **Time steps:** If ``solve_thermal`` runs without a ``ts_id`` on a network with dynamic components, it now uses the ID following the last one used and raises a warning. All the iterations of a time step share the same ID, so that dynamic components advance only once. This replaces the warning raised by ``SimpleStep`` when no ``ts_id`` is given.
* **Zero mass flows:** During thermal simulations, dynamic components with zero mass flow temporarily receive the small mass flows used by the solver, so that their outlet temperature is computed at the right end.
* **Performance:** The thermal solver uses sparse matrices for networks with 100 nodes or more, and networks are built faster. On the OpenDHN benchmark (1352 nodes), thermal simulations are about three times faster.
* ``compute_edge_temperatures`` only stores the values of the edges in ``mask`` when ``set_values`` is True.
* **Packaging:** The package metadata now uses the standard ``[project]`` table of ``pyproject.toml``, and declares the license as ``AGPL-3.0-only`` instead of an unrecognised one.
* **Hydraulic pressure calculation refactor:** Solved an issue where Reynolds number and friction factor were not consistently set for all pipe types, particularly ``LagrangianPipe``. The ``compute_dp_pipe_net`` and ``compute_dp_valve_net`` functions now accept a ``mask`` parameter, processing and returning results only for the specified components. Consequently, their ``dp`` and ``dp_der`` outputs now have a dimensionality corresponding to the masked subset, rather than the entire network. The ``compute_dp`` function has been updated to correctly pass component masks to these vector functions and integrate their masked outputs into the overall pressure arrays. (See #6)
* **Heat exchanger model:** Refined the heat exchanger model's behavior (``compute_hx_temp`` and related functions) for consistent energy interpretation in dynamic simulations. The ``delta_q`` parameter and return values are now uniformly treated as Watt-hours (Wh). A new ``stepsize`` parameter (in seconds), defaulting to ``3600.0`` s for backward compatibility, was added for internal energy-to-power conversions. Users with non-hourly ``stepsize`` in dynamic simulations should now explicitly pass their ``stepsize`` to consumers and producers. (See #5)
* Changed the sign of ``LagrangianPipe``'s ``delta_q`` to be negative when heat is lost.
* Updated documentation reflecting changes from #5.
* Updated default ``stepsize`` to ``3600.0``.

Fixed
~~~~~

* ``set_edge_attributes`` with a ``mask`` could assign the values to the wrong edges, depending on the order of the nodes. This affected, for example, the Reynolds number and friction factor of pipes in networks with different types of pipes.
* Dynamic components no longer advance at each iteration of thermal simulations run without a ``ts_id``.
* Fixed thermal simulations with zero mass flow edges forming closed loops.
* Fixed the wall temperatures of ``LagrangianPipe`` with negative mass flows, which were matched to the wrong volumes.
* ``LagrangianPipe`` no longer computes complex temperatures with NumPy 2.
* Fixed the hydrostatic pressure difference in pipes, which raised an error in network simulations and was returned as an array for single pipes.
* Fixed the Nusselt number in the transition regime, which was always equal to the laminar value.
* Fixed the ``isconstant`` flag of ``ConstantWater``, ``Water`` and ``KusudaSoil``.
* ``valves_mask`` and ``pumps_mask`` now return branch valves and pumps instead of consumers and producers, and ``imposed_valves_mask`` and ``imposed_pumps_mask`` only return the edges with an imposed value.
* Fixed the computation of node pressures when a source node is given.
* The error printed by ``solve_thermal`` now has the right unit (kg·K/s).
* Fixed a bug in ``pipe_test`` related to local data reading.

Removed
~~~~~~~

* Removed ``setup.py``: the package is built from ``pyproject.toml``.


0.1.3
-----

Released on October 30, 2024

Added
~~~~~

* Added documentation files in /docs
* Added a GitHub action to deploy the documentation on Pages

Changed
~~~~~~~

* Improved docstrings and type hints
* docstring_parameters now ignores curly braces outside the Parameter section of a docstring
* Modified .pre-commit-config.yaml so that isort ignores init files to avoid circular import issues
* Added dependencies needed for docs to pyproject.toml


0.1.2
-----

Released on March 19, 2024

Changed
~~~~~~~

* Fixed a bug where the wrong vector function was called when using LagrangianPipe


0.1.1
-----

Released on January 19, 2024

Added
~~~~~

* Added dynamic pipe from `Boghetti et al. <https://doi.org/10.1016/j.energy.2023.130169>`_ with:

  - Verification on experimental data from `Schweiger et al. <https://doi.org/10.1016/j.energy.2018.08.193>`_
  - Tests

* Added new default values
* Added CHANGELOG.md
* Added decorator for dynamic default values in docstring

Changed
~~~~~~~

* Updated Pipe's docstring


0.1.0
-----

Released on January 12, 2024

* Initial release
