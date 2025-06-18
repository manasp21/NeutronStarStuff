#!/usr/bin/env python3
"""
Simple mass-radius plotter for neutron star EoS data
"""

import numpy as np
import matplotlib.pyplot as plt

def read_eos_data(filepath):
    """Read EoS data from .dat file"""
    data = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # Split line by whitespace and convert to floats
            values = line.strip().split()
            if len(values) == 4:
                try:
                    epsilon = float(values[0])
                    pressure = float(values[1])
                    data.append((epsilon, pressure))
                except ValueError:
                    continue
    return data

# Physical constants in geometric units (G=c=1)
M_sun_km = 1.476  # Solar mass in km
MeV_fm3_to_km2 = 1.32e-15  # Conversion factor: 1 MeV/fm³ = 1.32e-15 km⁻²

def solve_tov(epsilon_c, eos_data, max_radius=20, dr=0.01):
    """
    Solve TOV equations for given central energy density
    Returns: (mass, radius) in geometric units
    """
    # Convert central energy density to geometric units
    epsilon_c_geom = epsilon_c * MeV_fm3_to_km2
    
    # Find pressure at central energy density using interpolation
    eps_vals, p_vals = zip(*eos_data)
    p_c_geom = np.interp(epsilon_c, eps_vals, p_vals) * MeV_fm3_to_km2
    
    # Initialize variables
    r = dr  # Start at dr to avoid singularity
    m = (4/3) * np.pi * r**3 * epsilon_c_geom  # Initial mass
    p = p_c_geom
    
    # Store integration history
    radii = [r]
    masses = [m]
    pressures = [p]
    
    # Integrate TOV equations
    while p > 1e-15 * p_c_geom and r < max_radius:
        # Get energy density at current pressure
        eps_geom = np.interp(p, [pr * MeV_fm3_to_km2 for pr in p_vals], eps_vals) * MeV_fm3_to_km2
        
        # TOV equations
        if r > 0 and r > 2*m:  # Avoid division by zero
            dp_dr = - (eps_geom + p) * (m + 4 * np.pi * r**3 * p) / (r * (r - 2 * m))
            dm_dr = 4 * np.pi * r**2 * eps_geom
        else:
            dp_dr = 0
            dm_dr = 0
        
        # Update values
        p += dp_dr * dr
        m += dm_dr * dr
        r += dr
        
        # Store values
        pressures.append(p)
        masses.append(m)
        radii.append(r)
    
    # Return surface values
    return (m, r)

def main():
    eos_file = 'EoS_DD2MEV_p15_CSS1_1.4_ncrit_0.335561_ecrit_334.285_pcrit_42.1651.dat'
    eos_data = read_eos_data(eos_file)
    
    # Get mass-radius points for different central densities (in physical units)
    central_densities = np.logspace(np.log10(1), np.log10(1000), 50)  # MeV/fm³
    mr_points = []
    
    for cd in central_densities:
        m_geom, r_geom = solve_tov(cd, eos_data)
        if m_geom > 0 and r_geom > 0:
            # Convert to physical units
            m_phys = m_geom / M_sun_km  # Solar masses
            r_phys = r_geom  # km
            mr_points.append((r_phys, m_phys))
    
    # Plot results
    radii, masses = zip(*mr_points)
    plt.figure(figsize=(10, 6))
    plt.plot(radii, masses, 'bo-', linewidth=2)
    plt.xlabel('Radius (km)')
    plt.ylabel('Mass (solar masses)')
    plt.title('Mass-Radius Relation for Neutron Star')
    plt.grid(True, alpha=0.3)
    plt.savefig('mass_radius_plot.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Mass-radius plot saved as 'mass_radius_plot.png'")

if __name__ == "__main__":
    main()