import numpy as np
import matplotlib.pyplot as plt
from simulation import get_flame, ThreeStageSolver

def plot_symposia_results(z, Tg, Ts, filename='temperature_profile.png'):
    # Plot full domain to see exit behavior
    mask = z <= 0.03
    z = z[mask]
    Tg = Tg[mask]
    Ts = Ts[mask]
    
    plt.figure(figsize=(10, 7))
    
    # Plot lines
    plt.plot(z, Tg, 'b-', linewidth=2.5, label='Gas Temp ($T_{Gas}$)')
    plt.plot(z, Ts, 'r-', linewidth=2.5, label='Solid Temp ($T_{Solid}$)')
    
    # Experimental Dots
    exp_x = [0.01, 0.018, 0.022, 0.03]
    exp_y = [380, 600, 1260, 1180]
    plt.plot(exp_x, exp_y, 'ko', markersize=8, label='$T_{Experimental}$')

    # Interface Line
    plt.axvline(x=0.022, color='k', linestyle=':', linewidth=1.5, label='Interface')

    plt.xlabel('Axial Position (m)', fontsize=14, fontweight='bold')
    plt.ylabel('Temperature (K)', fontsize=14, fontweight='bold')
    
    plt.title(r"80% LPG, 20% $H_2$" + "\n" + r"1.8 kW, $\phi=0.752$", fontsize=14)
    
    plt.legend(loc='upper left', frameon=True, edgecolor='black', fancybox=False, fontsize=12)
    
    # Full domain view
    plt.xlim(0.004, 0.03) 
    plt.ylim(200, 1800)   
    
    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray', alpha=0.3)
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha=0.1)
    
    ax = plt.gca()
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)

    plt.tight_layout()
    print(f"Saving reproduction plot to {filename}...")
    plt.savefig(filename, dpi=300)

def run_simulation():
    flame_ref = get_flame()
    # Design: [dp1, eps1, dp2, eps2, dp3, eps3]
    design = [1.3, 0.4, 0.6, 0.4, 5.0, 0.9]
    
    # Note: Passing None for z_grid and Tg_ref as solver generates its own
    solver = ThreeStageSolver(None, design, flame_ref, None)
    eff, Ts, Tg = solver.solve()
    
    print(f"Simulation Complete.")
    print(f"Peak Gas Temp: {np.max(Tg):.0f} K")
    print(f"Efficiency: {eff:.1%}")
    
    plot_symposia_results(solver.z, Tg, Ts)

if __name__ == "__main__":
    run_simulation()
