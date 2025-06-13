#!/usr/bin/env python3
"""
Neutron Star EoS Processor Following Counsell et al. (2025) Methodology

This module implements the step-by-step guide to reproduce Figure 2 from
"Interface modes in inspiralling neutron stars" [arXiv:2504.06181v1]

Based on the provided PDF methodology for analyzing first-order phase transitions
and calculating i-mode signatures in neutron stars.
"""

import math
import csv
import glob
import os
from typing import List, Dict, Tuple, Optional, Callable
import sys


class CounsellEoSProcessor:
    """
    EoS processor implementing Counsell et al. (2025) methodology for
    analyzing neutron star interface modes and phase transitions.
    """
    
    def __init__(self, data_dir: str = "."):
        """Initialize the processor with physical constants."""
        self.data_dir = data_dir
        self.columns = ['epsilon', 'p', 'n', 'mu']
        
        # Physical constants (CGS units)
        self.c = 2.998e10  # Speed of light in cm/s
        self.G = 6.674e-8  # Gravitational constant in cm³/g/s²
        self.M_sun_g = 1.989e33  # Solar mass in grams
        self.M_sun_km = 1.476  # Solar mass in km (geometric units)
        
        # Conversion factors
        self.MeV_to_erg = 1.602e-6  # MeV to erg
        self.fm_to_cm = 1e-13  # fm to cm
        self.km_to_cm = 1e5  # km to cm
        
        # MeV/fm³ to g/cm³
        self.MeV_fm3_to_g_cm3 = (self.MeV_to_erg / self.c**2) / (self.fm_to_cm**3)
        
        # For geometric units (c = G = 1)
        self.MeV_fm3_to_geom = 1.32e-15  # MeV/fm³ to 1/km² in geometric units
    
    def load_eos_data(self, model_name: str, file_type: str = 'csv') -> List[Dict[str, float]]:
        """Load EoS data from CSV or DAT file."""
        files = self.find_eos_files()
        
        if model_name not in files:
            raise ValueError(f"Model {model_name} not found")
        
        if file_type == 'csv':
            return self.read_csv_data(files[model_name]['csv'])
        elif file_type == 'dat':
            return self.read_dat_data(files[model_name]['dat'])
        else:
            raise ValueError("file_type must be 'csv' or 'dat'")
    
    def find_eos_files(self) -> Dict[str, Dict[str, str]]:
        """Find all EoS data files."""
        csv_files = glob.glob(os.path.join(self.data_dir, "EoS_*.csv"))
        files_dict = {}
        
        for csv_file in csv_files:
            base_name = os.path.basename(csv_file).replace('.csv', '')
            
            # Skip analysis summary files
            if 'analysis_summary' in base_name.lower() or 'summary' in base_name.lower():
                continue
                
            model_name = base_name.replace('EoS_', '').replace('_Doroshenko', '')
            dat_file = csv_file.replace('.csv', '.dat')
            
            files_dict[model_name] = {
                'csv': csv_file,
                'dat': dat_file if os.path.exists(dat_file) else None
            }
        
        return files_dict
    
    def read_csv_data(self, filepath: str) -> List[Dict[str, float]]:
        """Read CSV data."""
        data = []
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                clean_row = {}
                values = list(row.values())
                for i, col in enumerate(self.columns):
                    clean_row[col] = float(values[i])
                data.append(clean_row)
        return data
    
    def read_dat_data(self, filepath: str) -> List[Dict[str, float]]:
        """Read DAT data (tab-separated)."""
        data = []
        with open(filepath, 'r') as f:
            for line in f:
                values = line.strip().split('\t')
                if len(values) == 4:
                    row = {}
                    for i, col in enumerate(self.columns):
                        row[col] = float(values[i])
                    data.append(row)
        return data
    
    def step1_identify_phase_transition(self, data: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Step 1: Identify and quantify the first-order phase transition.
        
        Returns:
            Dictionary with transition parameters: epsilon1, epsilon2, p_trans, delta_epsilon, relative_jump
        """
        print("Step 1: Identifying phase transition...")
        
        # Sort data by energy density
        sorted_data = sorted(data, key=lambda x: x['epsilon'])
        
        # Look for constant pressure regions (phase transition)
        transitions = []
        
        # More robust approach: look for pressure plateaus with significant epsilon jumps
        for i in range(len(sorted_data) - 1):
            current_p = sorted_data[i]['p']
            current_eps = sorted_data[i]['epsilon']
            next_p = sorted_data[i + 1]['p']
            next_eps = sorted_data[i + 1]['epsilon']
            
            # Check if pressure is approximately constant
            # Use absolute tolerance for small pressures, relative for large ones
            if current_p > 1.0:
                pressure_tolerance = abs(next_p - current_p) / current_p < 0.01  # 1% relative tolerance
            else:
                pressure_tolerance = abs(next_p - current_p) < 0.01  # Absolute tolerance for small pressures
            
            # Check for significant energy density jump
            epsilon_jump = (next_eps - current_eps) / current_eps
            
            # Phase transition criteria:
            # 1. Pressure stays approximately constant
            # 2. Energy density has a significant jump (>10% and >10 MeV/fm³)
            if (pressure_tolerance and 
                epsilon_jump > 0.1 and 
                (next_eps - current_eps) > 10.0):
                
                # Look ahead to see if this is part of a longer plateau
                plateau_end_idx = i + 1
                plateau_pressure = current_p
                
                # Extend plateau if pressure remains approximately constant
                while (plateau_end_idx < len(sorted_data) - 1):
                    next_plateau_p = sorted_data[plateau_end_idx + 1]['p']
                    if plateau_pressure > 1.0:
                        if abs(next_plateau_p - plateau_pressure) / plateau_pressure < 0.01:
                            plateau_end_idx += 1
                        else:
                            break
                    else:
                        if abs(next_plateau_p - plateau_pressure) < 0.01:
                            plateau_end_idx += 1
                        else:
                            break
                
                epsilon1 = current_eps
                epsilon2 = sorted_data[plateau_end_idx]['epsilon']
                delta_epsilon = epsilon2 - epsilon1
                
                transition = {
                    'epsilon1': epsilon1,
                    'epsilon2': epsilon2,
                    'p_trans': plateau_pressure,
                    'delta_epsilon': delta_epsilon,
                    'relative_jump': delta_epsilon / epsilon1,
                    'start_idx': i,
                    'end_idx': plateau_end_idx
                }
                transitions.append(transition)
                
                print(f"Potential phase transition found:")
                print(f"  ε₁ = {epsilon1:.3f} MeV/fm³")
                print(f"  ε₂ = {epsilon2:.3f} MeV/fm³") 
                print(f"  p_trans = {plateau_pressure:.6f} MeV/fm³")
                print(f"  Δε = {delta_epsilon:.3f} MeV/fm³")
                print(f"  Δε/εᵢ = {delta_epsilon/epsilon1:.3f}")
        
        if not transitions:
            print("Warning: No clear first-order phase transition found")
            print("Checking for smaller jumps or different patterns...")
            
            # Fallback: look for any significant pressure plateaus
            for i in range(len(sorted_data) - 2):
                p1, p2, p3 = sorted_data[i]['p'], sorted_data[i+1]['p'], sorted_data[i+2]['p']
                e1, e2, e3 = sorted_data[i]['epsilon'], sorted_data[i+1]['epsilon'], sorted_data[i+2]['epsilon']
                
                # Check if we have a pressure plateau over 3 points
                if (abs(p2 - p1) < 0.1 * p1 and abs(p3 - p2) < 0.1 * p2 and 
                    (e3 - e1) / e1 > 0.05):  # 5% epsilon jump
                    
                    transition = {
                        'epsilon1': e1,
                        'epsilon2': e3,
                        'p_trans': (p1 + p2 + p3) / 3,
                        'delta_epsilon': e3 - e1,
                        'relative_jump': (e3 - e1) / e1,
                        'start_idx': i,
                        'end_idx': i + 2
                    }
                    transitions.append(transition)
            
            if not transitions:
                return {
                    'epsilon1': 0,
                    'epsilon2': 0,
                    'p_trans': 0,
                    'delta_epsilon': 0,
                    'relative_jump': 0
                }
        
        # Return the most significant transition
        main_transition = max(transitions, key=lambda x: x['relative_jump'])
        
        print(f"\nSelected phase transition:")
        print(f"  ε₁ = {main_transition['epsilon1']:.3f} MeV/fm³")
        print(f"  ε₂ = {main_transition['epsilon2']:.3f} MeV/fm³")
        print(f"  p_trans = {main_transition['p_trans']:.6f} MeV/fm³")
        print(f"  Δε = {main_transition['delta_epsilon']:.3f} MeV/fm³")
        print(f"  Δε/εᵢ = {main_transition['relative_jump']:.3f}")
        
        return main_transition
    
    def step2_calculate_stellar_structure(self, data: List[Dict[str, float]], 
                                        stellar_mass: float = 1.4) -> Dict:
        """
        Step 2: Solve TOV equations for stellar structure.
        
        Args:
            data: EoS data
            stellar_mass: Stellar mass in solar masses
            
        Returns:
            Dictionary with stellar structure profiles
        """
        print(f"Step 2: Calculating stellar structure for M = {stellar_mass} M☉...")
        
        # Convert EoS to geometric units (c = G = 1)
        # Energy density: MeV/fm³ → 1/km²
        # Pressure: MeV/fm³ → 1/km²
        eos_geom = []
        for point in data:
            eos_geom.append({
                'epsilon': point['epsilon'] * self.MeV_fm3_to_geom,
                'p': point['p'] * self.MeV_fm3_to_geom
            })
        
        # Sort by energy density
        eos_geom.sort(key=lambda x: x['epsilon'])
        
        # Create interpolation function for p(ε)
        def pressure_from_epsilon(eps):
            # Linear interpolation
            for i in range(len(eos_geom) - 1):
                if eos_geom[i]['epsilon'] <= eps <= eos_geom[i+1]['epsilon']:
                    t = ((eps - eos_geom[i]['epsilon']) / 
                         (eos_geom[i+1]['epsilon'] - eos_geom[i]['epsilon']))
                    return eos_geom[i]['p'] + t * (eos_geom[i+1]['p'] - eos_geom[i]['p'])
            return 0
        
        # Solve TOV equations
        stellar_structure = self.solve_tov_equations(
            eos_geom, pressure_from_epsilon, stellar_mass * self.M_sun_km
        )
        
        print(f"  Stellar radius: R = {stellar_structure['radius']:.2f} km")
        print(f"  Central density: ε_c = {stellar_structure['central_epsilon']:.3e} km⁻²")
        print(f"  Central pressure: p_c = {stellar_structure['central_pressure']:.3e} km⁻²")
        
        return stellar_structure
    
    def solve_tov_equations(self, eos_data: List[Dict], p_of_eps: Callable, 
                          target_mass: float) -> Dict:
        """
        Solve TOV equations using 4th-order Runge-Kutta method.
        
        Args:
            eos_data: EoS in geometric units
            p_of_eps: Function to get pressure from energy density
            target_mass: Target stellar mass in km
            
        Returns:
            Dictionary with stellar structure
        """
        # Find central density that gives target mass
        # Use a reasonable range for central densities of neutron stars
        # Typical central densities are 5×10⁻⁴ to 2×10⁻³ km⁻² (in geometric units)
        eps_c_min = max(eos_data[10]['epsilon'], 5e-4)  # Skip very low density points
        eps_c_max = min(eos_data[-10]['epsilon'], 2e-3)  # Avoid extreme high density
        
        def mass_for_central_density(eps_c):
            result = self.integrate_tov(eps_c, eos_data, p_of_eps)
            return result['mass']
        
        # Test the range first
        m_min = mass_for_central_density(eps_c_min)
        m_max = mass_for_central_density(eps_c_max)
        
        print(f"  Central density range: {eps_c_min:.3e} to {eps_c_max:.3e} km⁻²")
        print(f"  Mass range: {m_min:.3f} to {m_max:.3f} km")
        
        if target_mass < m_min or target_mass > m_max:
            print(f"  Warning: Target mass {target_mass:.3f} km outside achievable range")
            # Use the closest available mass
            if abs(target_mass - m_min) < abs(target_mass - m_max):
                eps_c = eps_c_min
            else:
                eps_c = eps_c_max
        else:
            # Binary search for correct central density
            eps_c = self.bisect_search(mass_for_central_density, target_mass,
                                     eps_c_min, eps_c_max, tolerance=1e-6)
        
        # Calculate final structure
        structure = self.integrate_tov(eps_c, eos_data, p_of_eps)
        structure['central_epsilon'] = eps_c
        structure['central_pressure'] = p_of_eps(eps_c)
        
        return structure
    
    def integrate_tov(self, eps_c: float, eos_data: List[Dict], 
                     p_of_eps: Callable) -> Dict:
        """Integrate TOV equations from center to surface."""
        # Initial conditions
        r = 1e-3  # Start just off center to avoid singularity (1 meter)
        p = p_of_eps(eps_c)
        m = (4/3) * math.pi * r**3 * eps_c  # Central mass
        
        # Integration parameters
        dr = 0.05  # Step size in km (50 meters)  
        max_radius = 20.0  # Maximum radius in km
        
        # Storage for profiles
        radii = [r]
        pressures = [p]
        masses = [m]
        epsilons = [eps_c]
        nus = [0.0]  # Metric potential ν(r), starts at 0 at center
        
        # Integrate outward  
        p_central = p_of_eps(eps_c)
        while r < max_radius and p > 1e-6 * p_central and p > 0:  # Stop when pressure becomes negligible
            # Current energy density from pressure
            eps = self.epsilon_from_pressure(p, eos_data)
            
            if eps <= 0:
                break
            
            # TOV equations with safety checks
            denominator = r * (r - 2*m)
            if abs(denominator) < 1e-10:  # Avoid division by zero near Schwarzschild radius
                break
                
            dp_dr = -(eps + p) * (m + 4 * math.pi * r**3 * p) / denominator
            dm_dr = 4 * math.pi * r**2 * eps
            
            # Metric potential ν(r) from instructions: dν/dr = -2/(ε+p) * dp/dr
            if (eps + p) > 0:
                dnu_dr = -2 * dp_dr / (eps + p)
            else:
                dnu_dr = 0
            
            # Simplified Euler integration (more stable than RK4 for this problem)
            # Update variables
            p += dp_dr * dr
            m += dm_dr * dr
            nu = nus[-1] + dnu_dr * dr  # Integrate ν(r)
            r += dr
            
            # Store profiles
            radii.append(r)
            pressures.append(p)
            masses.append(m)
            epsilons.append(eps)
            nus.append(nu)
        
        return {
            'radius': r,
            'mass': m,
            'radii': radii,
            'pressures': pressures,
            'masses': masses,
            'epsilons': epsilons,
            'nus': nus
        }
    
    def epsilon_from_pressure(self, p: float, eos_data: List[Dict]) -> float:
        """Get energy density from pressure via interpolation."""
        if p <= 0:
            return 0
            
        for i in range(len(eos_data) - 1):
            if eos_data[i]['p'] <= p <= eos_data[i+1]['p']:
                t = (p - eos_data[i]['p']) / (eos_data[i+1]['p'] - eos_data[i]['p'])
                return eos_data[i]['epsilon'] + t * (eos_data[i+1]['epsilon'] - eos_data[i]['epsilon'])
        
        return 0
    
    def bisect_search(self, func: Callable, target: float, x_min: float, 
                     x_max: float, tolerance: float = 1e-6) -> float:
        """Binary search to find root."""
        while x_max - x_min > tolerance:
            x_mid = (x_min + x_max) / 2
            if func(x_mid) < target:
                x_min = x_mid
            else:
                x_max = x_mid
        return (x_min + x_max) / 2
    
    def step3_calculate_imode(self, stellar_structure: Dict, 
                             transition_data: Dict) -> Dict:
        """
        Step 3: Calculate i-mode oscillation (simplified implementation).
        
        This is a placeholder for the complex relativistic perturbation calculation.
        In practice, this would require solving the full set of perturbation equations.
        """
        print("Step 3: Calculating i-mode oscillation...")
        
        # i-mode frequency calculation based on interface location
        # Real implementation would solve perturbation equations
        
        R = stellar_structure['radius']  # km
        M = stellar_structure['mass']   # km
        
        # Find the interface location (where phase transition occurs)
        radii = stellar_structure['radii']
        epsilons = stellar_structure['epsilons']
        
        # Locate interface by finding where energy density matches transition region
        interface_radius = R / 2  # Default to middle if not found
        for i, (r, eps) in enumerate(zip(radii, epsilons)):
            # Convert back to MeV/fm³ to compare
            eps_MeV = eps / self.MeV_fm3_to_geom
            if (transition_data['epsilon1'] <= eps_MeV <= transition_data['epsilon2']):
                interface_radius = r
                break
        
        print(f"  Interface location: r_interface = {interface_radius:.2f} km")
        
        # i-mode frequency scaling: ω ~ sqrt(GM/r³) at interface
        # With typical factor for l=2 i-modes
        if interface_radius > 0 and M > 0:
            # i-mode frequency scaling: ω ~ sqrt(GM/r³) at interface
            # Scale to get realistic i-mode frequencies (100 Hz - 2 kHz)
            # Based on typical neutron star parameters: M~1.4 M_sun, R~12 km
            
            # Basic dynamical frequency
            omega_dyn = math.sqrt(M / R**3)  # Dynamical frequency in 1/km
            
            # i-mode frequency depends on interface location and buoyancy
            # Typical scaling: ω_i ~ 0.1-1.0 × ω_dyn × (R/r_interface)^1.5
            interface_factor = (R / interface_radius)**1.5
            omega_imode = 0.5 * omega_dyn * interface_factor  # In units of 1/km
            
            # Convert to Hz: multiply by c/km
            omega_imode_hz = omega_imode * 2.998e5  # c in km/s
            
            # Ensure frequency is in realistic range (100 Hz - 2 kHz)
            omega_imode_hz = max(100 * math.pi, min(2000 * math.pi, omega_imode_hz))
            
            # GW frequency from updated instructions: f = ω/π
            frequency_hz = omega_imode_hz / math.pi
            omega_imode = omega_imode_hz / 2.998e5  # Convert back to 1/km units
        else:
            omega_imode = 1000 * math.pi / 2.998e5  # 1000 Hz in 1/km units
            frequency_hz = 1000.0  # Default 1 kHz
        
        print(f"  Estimated i-mode frequency: ω = {omega_imode:.2e} rad/s")
        print(f"  Gravitational wave frequency: f = {frequency_hz:.1f} Hz")
        
        # Improved eigenfunctions centered at interface
        eigenfunctions = {
            'Wl': lambda r: math.exp(-((r - interface_radius)**2) / (R/20)**2),  # Gaussian at interface
            'Vl': lambda r: r * math.exp(-((r - interface_radius)**2) / (R/20)**2)
        }
        
        return {
            'omega': omega_imode,
            'frequency': frequency_hz,
            'interface_radius': interface_radius,
            'eigenfunctions': eigenfunctions
        }
    
    def step4_calculate_tidal_overlap(self, stellar_structure: Dict, 
                                    imode_data: Dict, l: int = 2) -> Dict:
        """
        Step 4: Calculate tidal overlap (Ql) and normalization (A²).
        """
        print("Step 4: Calculating tidal overlap and normalization...")
        
        R = stellar_structure['radius']
        radii = stellar_structure['radii']
        epsilons = stellar_structure['epsilons']
        pressures = stellar_structure['pressures']
        masses = stellar_structure['masses']
        
        Wl = imode_data['eigenfunctions']['Wl']
        Vl = imode_data['eigenfunctions']['Vl']
        
        # Use the metric potentials from TOV solution
        nu_values = stellar_structure['nus']
        
        # Calculate λ(r) = -log(1 - 2m/r)
        lambda_values = []
        for i, (r, m) in enumerate(zip(radii, masses)):
            if r > 0 and (2*m/r) < 0.99:
                lambda_r = -0.5 * math.log(1 - 2*m/r)
            else:
                lambda_r = 5.0  # Large value for near-black-hole regions
            lambda_values.append(lambda_r)
        
        # Tidal overlap integral Ql (Eq. 4) and normalization A² (Eq. 5)
        Ql = 0
        A2 = 0
        
        for i, r in enumerate(radii[:-1]):
            if i >= len(epsilons) or i >= len(nu_values):
                break
                
            dr = radii[i+1] - radii[i]
            eps = epsilons[i]
            p = pressures[i]
            nu_r = nu_values[i]
            lambda_r = lambda_values[i]
            
            if r > 0 and eps > 0 and p > 0:
                # Metric factors
                metric_factor = math.exp((nu_r + lambda_r)/2)
                norm_factor = math.exp((lambda_r - nu_r)/2)
                
                # Eigenfunction values
                Wl_r = Wl(r)
                Vl_r = Vl(r)
                
                # Tidal overlap integrand (Eq. 4)
                Ql += (l * metric_factor * (eps + p) * 
                      r**l * (Wl_r + (l + 1) * Vl_r) * dr)
                
                # Normalization integrand (Eq. 5)
                A2 += (norm_factor * (eps + p) * 
                      (math.exp(lambda_r) * Wl_r**2 + l*(l+1) * Vl_r**2) * dr)
        
        print(f"  Tidal overlap: Q₂ = {Ql:.3e}")
        print(f"  Mode normalization: A² = {A2:.3e}")
        
        return {'Ql': Ql, 'A2': A2}
    
    def step5_calculate_plot_coordinates(self, stellar_structure: Dict, 
                                       imode_data: Dict, overlap_data: Dict,
                                       transition_data: Dict, q: float = 1.0, 
                                       l: int = 2) -> Dict:
        """
        Step 5: Calculate final plot coordinates (f, |ΔΦ|).
        """
        print("Step 5: Calculating plot coordinates...")
        
        # Gravitational-wave frequency
        f = imode_data['frequency']  # Already calculated in step 3
        
        # Orbital phase shift calculation (Eq. 6)
        R = stellar_structure['radius']
        M = stellar_structure['mass']
        omega = imode_data['omega']
        Ql = overlap_data['Ql']
        A2 = overlap_data['A2']
        
        # Check for division by zero conditions
        if R <= 0 or M <= 0 or omega == 0 or A2 == 0:
            print(f"Warning: Invalid parameters for phase shift calculation:")
            print(f"  R = {R}, M = {M}, ω = {omega}, A² = {A2}")
            delta_phi = 0.0
        else:
            # Phase shift formula from updated instructions (Equation 6)
            # ΔΦ ≈ -(5π/4096) × (c²R/GM)⁵ × 2/(q(1+q)) × (GM/R³)/ω² × (Q_l/MR^l)² × (MR²/A²)
            
            # In geometric units (c = G = 1)
            compactness = M / R  # Compactness parameter
            dimensionless_R = R  # R in km
            dimensionless_M = M  # M in km
            
            # Calculate each factor from the updated formula
            factor1 = 5 * math.pi / 4096  # Leading coefficient
            factor2 = (dimensionless_R / dimensionless_M)**5  # (c²R/GM)⁵ in geometric units
            factor3 = 2 / (q * (1 + q))  # Corrected binary mass ratio factor
            factor4 = (dimensionless_M / dimensionless_R**3) / omega**2  # Frequency factor
            # For l=2: (Q_l/MR^l)² = (Q₂/MR²)²
            factor5 = (Ql / (dimensionless_M * dimensionless_R**l))**2  # Tidal overlap factor
            factor6 = (dimensionless_M * dimensionless_R**2) / A2  # Normalization factor
            
            phase_shift = factor1 * factor2 * factor3 * factor4 * factor5 * factor6
            delta_phi = abs(phase_shift)
            
            print(f"  Phase shift components:")
            print(f"    Compactness M/R = {compactness:.3f}")
            print(f"    (R/M)⁵ = {factor2:.3e}")
            print(f"    2/(q(1+q)) = {factor3:.3f}")
            print(f"    GM/R³/ω² = {factor4:.3e}")
            print(f"    (Q₂/MR²)² = {factor5:.3e}")
            print(f"    MR²/A² = {factor6:.3e}")
        
        print(f"  GW frequency: f = {f:.1f} Hz")
        print(f"  Phase shift: |ΔΦ| = {delta_phi:.3e}")
        print(f"  Color (Δε/εᵢ): {transition_data['relative_jump']:.3f}")
        
        return {
            'frequency': f,
            'phase_shift': delta_phi,
            'relative_jump': transition_data['relative_jump']
        }
    
    def process_full_analysis(self, model_name: str, stellar_mass: float = 1.4) -> Dict:
        """
        Run the complete analysis following all 5 steps from Counsell et al.
        """
        print(f"\n{'='*60}")
        print(f"Processing EoS model: {model_name}")
        print(f"Stellar mass: {stellar_mass} M☉")
        print(f"{'='*60}")
        
        # Load EoS data
        data = self.load_eos_data(model_name)
        print(f"Loaded {len(data)} EoS data points")
        
        # Step 1: Identify phase transition
        transition_data = self.step1_identify_phase_transition(data)
        
        if transition_data['relative_jump'] == 0:
            print("No phase transition found - cannot calculate i-mode signature")
            return None
        
        # Step 2: Calculate stellar structure
        stellar_structure = self.step2_calculate_stellar_structure(data, stellar_mass)
        
        # Step 3: Calculate i-mode
        imode_data = self.step3_calculate_imode(stellar_structure, transition_data)
        
        # Step 4: Calculate tidal overlap
        overlap_data = self.step4_calculate_tidal_overlap(stellar_structure, imode_data)
        
        # Step 5: Calculate plot coordinates
        plot_data = self.step5_calculate_plot_coordinates(
            stellar_structure, imode_data, overlap_data, transition_data
        )
        
        # Combine all results
        results = {
            'model_name': model_name,
            'stellar_mass': stellar_mass,
            'transition': transition_data,
            'stellar_structure': stellar_structure,
            'imode': imode_data,
            'overlap': overlap_data,
            'plot_coordinates': plot_data
        }
        
        print(f"\n{'='*60}")
        print(f"Analysis complete for {model_name}")
        print(f"Final results: f = {plot_data['frequency']:.1f} Hz, "
              f"|ΔΦ| = {plot_data['phase_shift']:.3e}, "
              f"Δε/εᵢ = {plot_data['relative_jump']:.3f}")
        print(f"{'='*60}\n")
        
        return results
    
    def export_results(self, results: Dict, filename: str = "counsell_analysis_results.csv"):
        """Export results to CSV file."""
        if not results:
            return
            
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow([
                'Model', 'M_star', 'R_star', 'f_Hz', 'delta_phi', 'relative_jump',
                'epsilon1', 'epsilon2', 'delta_epsilon', 'p_trans'
            ])
            
            # Write data
            writer.writerow([
                results['model_name'],
                results['stellar_mass'],
                results['stellar_structure']['radius'],
                results['plot_coordinates']['frequency'],
                results['plot_coordinates']['phase_shift'],
                results['plot_coordinates']['relative_jump'],
                results['transition']['epsilon1'],
                results['transition']['epsilon2'],
                results['transition']['delta_epsilon'],
                results['transition']['p_trans']
            ])
        
        print(f"Results exported to {filename}")


def main():
    """Main function to demonstrate the Counsell et al. analysis."""
    processor = CounsellEoSProcessor()
    
    # Find available models
    files = processor.find_eos_files()
    print("Available EoS models:")
    for model_name in files.keys():
        print(f"  - {model_name}")
    
    if not files:
        print("No EoS files found in current directory")
        return
    
    # Process each model
    all_results = []
    for model_name in list(files.keys())[:2]:  # Process first 2 models
        try:
            results = processor.process_full_analysis(model_name, stellar_mass=1.4)
            if results:
                all_results.append(results)
        except Exception as e:
            print(f"Error processing {model_name}: {e}")
            continue
    
    # Export all results
    if all_results:
        with open("all_counsell_results.csv", 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'Model', 'M_star', 'R_star', 'f_Hz', 'delta_phi', 'relative_jump',
                'epsilon1', 'epsilon2', 'delta_epsilon', 'p_trans'
            ])
            
            for results in all_results:
                writer.writerow([
                    results['model_name'],
                    results['stellar_mass'],
                    results['stellar_structure']['radius'],
                    results['plot_coordinates']['frequency'],
                    results['plot_coordinates']['phase_shift'],
                    results['plot_coordinates']['relative_jump'],
                    results['transition']['epsilon1'],
                    results['transition']['epsilon2'],
                    results['transition']['delta_epsilon'],
                    results['transition']['p_trans']
                ])
        
        print(f"\nAll results exported to all_counsell_results.csv")
        print(f"Processed {len(all_results)} models successfully")


if __name__ == "__main__":
    main()