import cantera as ct
import numpy as np

# --- 1. Gas and Inlet Conditions ---
gas = ct.Solution('drm19.yaml')
inlet_temp = 300.0
pressure = ct.one_atm
equivalence_ratio = 0.65
inlet_velocity = 0.45

gas.TP = inlet_temp, pressure
gas.set_equivalence_ratio(phi=equivalence_ratio, fuel='CH4', oxidizer={'O2': 0.21, 'N2': 0.78, 'AR': 0.01})

print("Successfully loaded 'drm19.yaml'.")
print("Initial gas state defined and ready for simulation.")
print(gas.report())

# --- 2. Simulation Domain ---
width = 0.0605  # meters
flame = ct.FreeFlame(gas, width=width)
flame.inlet.T = inlet_temp
flame.inlet.X = gas.X
flame.inlet.mdot = gas.density * inlet_velocity

print("\n1D Flame object created.")
