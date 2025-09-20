# custom_cantera_solver.py
import cantera as ct
import numpy as np

def run_combustion_simulation(pore_diameter_mm, porosity_val):
    """
    Runs a porous burner simulation for given pore diameter and porosity.
    Compatible with Cantera 2.5.0
    """
    try:
        # --- 1. Gas and Flame Setup ---
        gas = ct.Solution('gri30.yaml')

        T_inlet = 300.0
        P = ct.one_atm
        phi = 0.65

        gas.set_equivalence_ratio(phi, 'CH4', {'O2':1.0, 'N2':3.76})
        gas.TP = T_inlet, P
        inlet_mass_fractions = gas.Y
        rho_inlet = gas.density
        mdot = 0.45 * rho_inlet  # inlet velocity u0 = 0.45 m/s

        gas.equilibrate('HP')
        adiabatic_flame_temp = gas.T

        # --- 2. Counterflow Premixed Flame ---
        width = 0.0605
        flame = ct.CounterflowPremixedFlame(gas, width=width)

        # Cantera 2.5.0 does not have flame.inlet
        # Use old API to set inlet conditions
        flame.mdot = mdot
        flame.T[0] = T_inlet
        flame.Y[0,:] = inlet_mass_fractions

        flame.set_refine_criteria(ratio=3.0, slope=0.1, curve=0.15)

        # --- 3. Porous Burner Profiles ---
        solid_temp = np.full(flame.grid.shape, T_inlet)
        pore_diameter = np.zeros_like(flame.grid)
        porosity = np.zeros_like(flame.grid)

        stage1_pore_diameter = 0.00029
        stage1_porosity = 0.835
        stage2_pore_diameter = pore_diameter_mm / 1000.0

        interface_location = 0.035
        transition_width = 0.004

        for i, z in enumerate(flame.grid):
            weight = 0.5 * (1 + np.tanh((z - interface_location) / (transition_width / 4)))
            pore_diameter[i] = stage1_pore_diameter * (1 - weight) + stage2_pore_diameter * weight
            porosity[i] = stage1_porosity * (1 - weight) + porosity_val * weight

        flame.energy_enabled = True
        flame.solve(loglevel=0, refine_grid=True)

        # --- 4. Simplified iterative solid-gas coupling ---
        for _ in range(25):
            gas_temp = flame.T
            Re = flame.density * np.abs(flame.velocity) * pore_diameter / (flame.viscosity + 1e-12)

            C = -400 * pore_diameter + 0.687
            m = 443.7 * pore_diameter + 0.361
            Nu_v = C * (Re**m)
            h_v = flame.thermal_conductivity * Nu_v / (pore_diameter**2 + 1e-9)

            source_term = h_v * (gas_temp - solid_temp)
            solid_temp += 0.05 * source_term

            # Set gas source term directly
            flame.set_source('T', h_v * (solid_temp - gas_temp) / (porosity + 1e-9))

            flame.solve(loglevel=0, refine_grid=False)

        flame.solve(loglevel=0, refine_grid=True)

        exit_solid_temp = solid_temp[-1]
        efficiency = (exit_solid_temp / adiabatic_flame_temp)**4
        return efficiency

    except Exception as e:
        print(f"    Simulation failed: {e}")
        return 0.0
