import cantera as ct

print("--- Cantera Installation Test ---")
print(f"Using Cantera version: {ct.__version__}")
print(f"Cantera library path: {ct.__file__}")

# Basic smoke tests for Python API availability
try:
    gas = ct.Solution("gri30.yaml")
    ok_solution = True
except Exception as e:
    ok_solution = False
    print(f"Failed to create Solution: {e}")

# Check presence of public 1D classes instead of Empty1D
have_1d = all(hasattr(ct, name) for name in [
    "FreeFlow", "Inlet1D", "Outlet1D", "Sim1D"
])

if ok_solution and have_1d:
    print("\nSUCCESS: Cantera Python installation looks correct (Solution + 1D classes available).")
else:
    print("\nISSUE: Missing key Python classes or Solution creation failed.")

# Optional: build a tiny 1D stack to be extra sure
try:
    gas = ct.Solution("gri30.yaml")
    flow = ct.FreeFlow(gas)
    left = ct.Inlet1D(gas)
    right = ct.Outlet1D(gas)
    sim = ct.Sim1D([left, flow, right])
    print("SUCCESS: Created a minimal Sim1D stack.")
except Exception as e:
    print(f"1D stack creation failed: {e}")
