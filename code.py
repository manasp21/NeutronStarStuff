import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- Physical Constants and Conversion Factors ---
CONV_FACTOR = 1.32534e-6  # MeV/fm^3 to m^-2
SOLAR_MASS_IN_M = 1477.0  # meters

def load_eos(filepath):
    """
    Loads and pre-processes the Equation of State data from a CSV file.
    """
    try:
        # Load the data from the CSV file using the header.
        eos_data = pd.read_csv(filepath, comment='#')
        eos_data.columns = [col.strip() for col in eos_data.columns]

        # --- FIX: Pre-process the EoS to improve numerical stability ---
        # The EoS contains extremely low pressures for the crust/atmosphere, which can
        # make the solver unstable. We'll filter these out before interpolating.
        # A pressure of 1e-8 MeV/fm^3 is still a very low pressure but provides a
        # more stable floor for the integration.
        stable_eos_data = eos_data[eos_data['p'] > 1e-8].copy()
        
        if stable_eos_data.empty:
            print("ERROR: Filtering removed all EoS data. Check the pressure values in the file.")
            return None, None, None, None, None

        pressure_data = stable_eos_data['p']
        energy_density_data = stable_eos_data['epsilon']

        pressure_geom = pressure_data * CONV_FACTOR
        energy_density_geom = energy_density_data * CONV_FACTOR
        
        # The new termination condition is the minimum pressure from our STABLE EoS.
        min_P_geom = pressure_geom.min()

        # Create log-space interpolation functions from the filtered data.
        P_from_eps = interp1d(np.log(energy_density_geom), np.log(pressure_geom), kind='linear',
                              bounds_error=False, fill_value=-np.inf) # Return -inf log(P) if out of bounds
        eps_from_P = interp1d(np.log(pressure_geom), np.log(energy_density_geom), kind='linear',
                              bounds_error=False, fill_value=-np.inf)

        # Wrap the log-interpolators to handle the exp/log conversions safely.
        def P_from_eps_wrapper(eps):
            return np.exp(P_from_eps(np.log(np.maximum(eps, 1e-40))))

        def eps_from_P_wrapper(P):
            return np.exp(eps_from_P(np.log(np.maximum(P, 1e-40))))

        print("EoS loaded and pre-processed successfully.")
        # We still return the original min/max for scanning the full physical range.
        min_eps_orig = eos_data['epsilon'].min()
        max_eps_orig = eos_data['epsilon'].max()

        return P_from_eps_wrapper, eps_from_P_wrapper, min_eps_orig, max_eps_orig, min_P_geom

    except Exception as e:
        print(f"An error occurred while loading the EoS file: {e}")
        return None, None, None, None, None


def tov_equations(r, y, eps_from_P):
    """
    Defines the Tolman-Oppenheimer-Volkoff (TOV) equations.
    """
    m, P = y
    
    if r < 1e-6: return [0, 0]
    if P <= 0: return [0, 0]

    eps = eps_from_P(P)
    if eps <= 0: return [0, 0]

    dmdr = 4 * np.pi * r**2 * eps

    if 2 * m >= r: return [dmdr, -np.inf]

    numerator = (eps + P) * (m + 4 * np.pi * r**3 * P)
    denominator = r * (r - 2 * m)
    dPdr = -numerator / denominator

    return [dmdr, dPdr]

def solve_tov(central_eps, P_from_eps, eps_from_P, min_P_geom):
    """
    Solves the TOV equations for a given central energy density.
    """
    central_eps_geom = central_eps * CONV_FACTOR
    central_P = P_from_eps(central_eps_geom)

    r_init = 1e-6
    m_init = (4. / 3.) * np.pi * r_init**3 * central_eps_geom
    P_init = central_P

    # Event function terminates when pressure drops to the minimum value
    # from our filtered EoS table.
    def surface_condition(r, y):
        return y[1] - min_P_geom
    surface_condition.terminal = True
    surface_condition.direction = -1

    max_radius_m = 30000.0  # 30 km
    
    # --- FIX: Use a solver designed for "stiff" problems like this one. ---
    solution = solve_ivp(
        lambda r, y: tov_equations(r, y, eps_from_P),
        (r_init, max_radius_m),
        [m_init, P_init],
        method='LSODA',  # LSODA is robust for stiff and non-stiff problems.
        events=surface_condition,
        dense_output=True,
        atol=1e-8, # Loosen tolerance slightly for stiff solver
        rtol=1e-8
    )

    if solution.status == 1:
        final_radius_m = solution.t[-1]
        final_mass_m = solution.y[0][-1]
        final_radius_km = final_radius_m / 1000.0
        final_mass_solar = final_mass_m / SOLAR_MASS_IN_M
        return final_mass_solar, final_radius_km
    else:
        return None, None


def generate_mr_curve(eos_filepath):
    """
    Generates and plots the mass-radius curve for a given EoS.
    """
    P_from_eps, eps_from_P, min_eps, max_eps, min_P_geom = load_eos(eos_filepath)
    if not P_from_eps:
        return

    min_core_eps = 50.0
    start_eps = max(min_eps, min_core_eps)

    if start_eps >= max_eps:
        print("\nERROR: EoS data does not contain high densities needed for neutron star cores.")
        return

    central_densities = np.logspace(np.log10(start_eps), np.log10(max_eps), num=250) # Increased points

    results = []
    print("\nStarting TOV integration for a range of central densities...")
    for i, eps_c in enumerate(central_densities):
        try:
            mass, radius = solve_tov(eps_c, P_from_eps, eps_from_P, min_P_geom)
            if mass is not None and radius is not None and radius > 5 and mass > 0.1:
                results.append((radius, mass))
                print(f"  {i+1}/{len(central_densities)}: eps_c = {eps_c:.2e} MeV/fm^3 -> M = {mass:.3f} M_sun, R = {radius:.2f} km")
        except Exception as e:
            # This allows the loop to continue even if one point fails.
            print(f"  {i+1}/{len(central_densities)}: Failed for eps_c = {eps_c:.2e} MeV/fm^3. Error: {e}")

    if not results:
        print("\nCould not generate any stable star configurations. The solver failed for all central densities.")
        return

    radii, masses = zip(*results)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(radii, masses, marker='.', linestyle='-', markersize=4, color='royalblue', label='M-R Curve')
    
    if masses:
        max_mass = max(masses)
        max_mass_idx = masses.index(max_mass)
        max_mass_radius = radii[max_mass_idx]
        ax.plot(max_mass_radius, max_mass, 'o', color='crimson', markersize=8,
                label=f'Max Mass: {max_mass:.2f} $M_\\odot$')
        ax.text(max_mass_radius + 0.2, max_mass, f'{max_mass:.2f} $M_\\odot$',
                verticalalignment='center', color='crimson', weight='bold')

    ax.set_xlabel('Radius (km)', fontsize=14)
    ax.set_ylabel('Mass ($M/M_\\odot$)', fontsize=14)
    ax.set_title('Mass-Radius Relation for EoS', fontsize=16, weight='bold')
    ax.set_xlim(left=9) 
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()


# --- Main Execution ---
if __name__ == "__main__":
    eos_file = 'EoS_DD2MEV_p15_CSS1_1.4_ncrit_0.335561_ecrit_334.285_pcrit_42.1651.csv'
    generate_mr_curve(eos_file)
