# Change Log

## Unreleased

### Added

### Changed

### Removed


## 0.1.4

### Added

* **Stratified thermal storage:** Added the new `StratifiedStorage` component, a dynamic storage tank based on the one-dimensional multi-node model of [Kleinbach et al.](https://doi.org/10.1016/0038-092X(93)90087-5), the same family of models as TRNSYS Type 4. It comes with the `Network.add_stratified_storage()` method, default values, a vectorized thermal model, documentation and tests.
* **Vectorized LagrangianPipe thermal model:** Network simulations now advance all Lagrangian pipes at once using the new vectorized function in `pydhn.components.lagrangian_pipe_thermal`, instead of looping over components, giving the same results with a much faster runtime.
* **Faster control logic dispatch:** Components can now declare which keys their control logic handles in the class attribute `_controlled_keys`, so that lookups of all other attributes skip the call to `_run_control_logic` entirely. Custom components that override `_run_control_logic` without declaring `_controlled_keys` keep the previous behavior.
* Added a warning when a thermal simulation runs without a specified `ts_id`.
* Added a new test (`tests/test_dynamic_thermal_balance.py`) to verify thermal energy balance in dynamic simulations.
* Added a new example (`examples/dynamic_simulation.py`) for dynamic simulations with `LagrangainPipe`.
* **Developer Guidelines:** Added `CONTRIBUTING.md`.
* **Line Ending Standardization:** Added `.gitattributes` to enforce consistent line endings (LF) across text files.
* Added new tests covering dimensionless numbers, nodal pressure assignment, zero-mass-flow filling, incidence and cycle matrices (including multi-producer networks), control logic, the Lagrangian pipe (component and vectorized), the stratified storage (component and vectorized), and a benchmark against OpenDHN results.
* Added new CI workflows: a canary run on the latest Python version and a matrix testing the extremes of the supported dependency ranges with `uv`, including an installation check.

### Changed

* **Hydraulic pressure calculation refactor:** Solved an issue where Reynolds number and friction factor were not consistently set for all pipe types, particularly `LagrangianPipe`. The `compute_dp_pipe_net` and `compute_dp_valve_net` functions now accept a `mask` parameter, processing and returning results only for the specified components. Consequently, their `dp` and `dp_der` outputs now have a dimensionality corresponding to the masked subset, rather than the entire network. The `compute_dp` function has been updated to correctly pass component masks to these vector functions and integrate their masked outputs into the overall pressure arrays. (See #6)
* Fixed a bug in `pipe_test` related to local data reading.
* **Heat exchanger model:** Refined the heat exchanger model's behavior (`compute_hx_temp` and related functions) for consistent energy interpretation in dynamic simulations. The `delta_q` parameter and return values are now uniformly treated as Watt-hours (Wh). A new `stepsize` parameter (in seconds), defaulting to `3600.0`s for backward compatibility, was added for internal energy-to-power conversions. Users with non-hourly `stepsize` in dynamic simulations should now explicitly pass their `stepsize` to consumers and producers. (See #5)
* Changed the sign of `LagrangianPipe`'s `delta_q` to be negative when heat is lost.
* Updated documentation reflecting changes from #5.
* Updated default `stepsize` to `3600.0`.
* **Extended dependency ranges:** Added support for NumPy 2 (replacing the removed `np.in1d` with `np.isin`), NetworkX 3 and Python 3.13. Python 3.9, which reached end of life, is no longer supported.
* **Faster nodal pressure assignment:** `_assign_node_pressure` was rewritten to batch node updates and avoid repeated edge lookups, and now also works when the pressure of a source node is imposed.
* The `valves_mask` and `pumps_mask` properties of `Network` now correctly return the indices of branch valves and branch pumps instead of consumers and producers. The `imposed_valves_mask` and `imposed_pumps_mask` properties now correctly ignore unset (`None` or NaN) values.
* Fixed a bug in `compute_nusselt` where the Nusselt number in the transitional regime collapsed to the laminar value.
* Fixed a bug in `compute_dp_pipe` where a trailing comma broke the computation of the hydrostatic pressure.
* Fixed the `isconstant` flag of `ConstantWater`, `Water` and `KusudaSoil`, which was not correctly passed to the parent class.
* Fixed the choice of the starting node for the Eulerian circuit used to fill zero-mass-flow edges in thermal simulations.
* Fixed a bug in `LagrangianPipe` where, in case of flow reversal, wall temperatures were not remapped on the flow-direction grid.

### Removed


## 0.1.3

Released on October 30, 2024

### Added

* Added documentation files in /docs
* Added a GitHub action to deploy the documentation on Pages

### Changed

* Improved docstrings and type hints
* docstring_parameters now ignores curly braces outside the Parameter section of a docstring
* Modified .pre-commit-config.yaml so that isort ignores init files to avoid circular import issues
* Added dependencies needed for docs to pyproject.toml

### Removed


## 0.1.2

Released on March 19, 2024

### Added

### Changed

* Fixed a bug where the wrong vector function was called when using LagrangianPipe

### Removed


## 0.1.1

Released on January 19, 2024

### Added

* Added dynamic pipe from [Boghetti et al.](https://doi.org/10.1016/j.energy.2023.130169) with:
	- Verification on experimental data from [Schweiger et al.](https://doi.org/10.1016/j.energy.2018.08.193)
	- Tests
* Added new default values
* Added CHANGELOG.md
* Added decorator for dynamic default values in docstring

### Changed

* Updated Pipe's docstring

### Removed


## 0.1.0

Released on January 12, 2024

* Initial release
