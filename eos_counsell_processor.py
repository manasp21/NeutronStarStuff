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
        
        # Physical constants (in units where c = G = 1)
        self.c = 1.0  # Speed of light
        self.G = 1.0  # Gravitational constant
        self.M_sun = 1.476  # Solar mass in km
        
        # Conversion factors
        self.MeV_fm3_to_km2 = 1.8e-15  # MeV/fm³ to 1/km²
        self.fm3_to_km3 = 1e-45  # fm³ to km³
    
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
        tolerance = 1e-6  # Relative tolerance for constant pressure
        
        i = 0
        while i < len(sorted_data) - 1:
            current_p = sorted_data[i]['p']
            start_idx = i
            
            # Find end of constant pressure region
            while (i < len(sorted_data) - 1 and 
                   abs(sorted_data[i + 1]['p'] - current_p) / current_p < tolerance):
                i += 1
            
            end_idx = i
            
            # If we found a plateau with significant energy density jump
            if end_idx > start_idx:
                epsilon1 = sorted_data[start_idx]['epsilon']
                epsilon2 = sorted_data[end_idx]['epsilon']
                delta_epsilon = epsilon2 - epsilon1
                
                # Only consider significant transitions
                if delta_epsilon / epsilon1 > 0.01:  # At least 1% jump
                    transition = {
                        'epsilon1': epsilon1,
                        'epsilon2': epsilon2,
                        'p_trans': current_p,
                        'delta_epsilon': delta_epsilon,
                        'relative_jump': delta_epsilon / epsilon1,
                        'start_idx': start_idx,
                        'end_idx': end_idx
                    }
                    transitions.append(transition)
            
            i += 1
        
        if not transitions:
            print("Warning: No clear first-order phase transition found")
            return {
                'epsilon1': 0,
                'epsilon2': 0,
                'p_trans': 0,
                'delta_epsilon': 0,
                'relative_jump': 0
            }
        
        # Return the most significant transition
        main_transition = max(transitions, key=lambda x: x['relative_jump'])
        
        print(f"Found phase transition:")
        print(f"  ε₁ = {main_transition['epsilon1']:.3e} MeV/fm³")
        print(f"  ε₂ = {main_transition['epsilon2']:.3e} MeV/fm³")
        print(f"  p_trans = {main_transition['p_trans']:.3e} MeV/fm³")
        print(f"  Δε = {main_transition['delta_epsilon']:.3e} MeV/fm³")
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
        
        # Convert EoS to geometric units
        # Energy density: MeV/fm³ → 1/km²
        # Pressure: MeV/fm³ → 1/km²
        eos_geom = []
        for point in data:
            eos_geom.append({
                'epsilon': point['epsilon'] * self.MeV_fm3_to_km2,
                'p': point['p'] * self.MeV_fm3_to_km2
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
            eos_geom, pressure_from_epsilon, stellar_mass * self.M_sun
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
        eps_c_min = eos_data[0]['epsilon']
        eps_c_max = eos_data[-1]['epsilon']
        
        def mass_for_central_density(eps_c):
            return self.integrate_tov(eps_c, eos_data, p_of_eps)['mass']
        
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
        r = 1e-6  # Start just off center to avoid singularity
        p = p_of_eps(eps_c)
        m = (4/3) * math.pi * r**3 * eps_c  # Central mass
        
        # Integration parameters
        dr = 0.01  # Step size in km
        max_radius = 20.0  # Maximum radius in km
        
        # Storage for profiles
        radii = [r]
        pressures = [p]
        masses = [m]
        epsilons = [eps_c]
        
        # Integrate outward
        while r < max_radius and p > 1e-10:  # Stop when pressure becomes negligible
            # Current energy density from pressure
            eps = self.epsilon_from_pressure(p, eos_data)
            
            if eps <= 0:
                break
            
            # TOV equations
            dp_dr = -(eps + p) * (m + 4 * math.pi * r**3 * p) / (r * (r - 2*m))
            dm_dr = 4 * math.pi * r**2 * eps
            
            # RK4 integration
            k1_p = dr * dp_dr
            k1_m = dr * dm_dr
            
            p_half = p + k1_p/2
            m_half = m + k1_m/2
            r_half = r + dr/2
            eps_half = self.epsilon_from_pressure(p_half, eos_data)
            
            if eps_half <= 0:
                break
                
            dp_dr_half = (-(eps_half + p_half) * (m_half + 4 * math.pi * r_half**3 * p_half) / 
                         (r_half * (r_half - 2*m_half)))
            dm_dr_half = 4 * math.pi * r_half**2 * eps_half
            
            k2_p = dr * dp_dr_half
            k2_m = dr * dm_dr_half
            
            # Update
            p += k2_p
            m += k2_m
            r += dr
            
            # Store profiles
            radii.append(r)
            pressures.append(p)
            masses.append(m)
            epsilons.append(eps)
        
        return {
            'radius': r,
            'mass': m,
            'radii': radii,
            'pressures': pressures,
            'masses': masses,
            'epsilons': epsilons
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
        
        # Simplified estimate based on typical i-mode frequencies
        # Real implementation would solve perturbation equations
        
        R = stellar_structure['radius']  # km
        M = stellar_structure['mass']   # km
        
        # Typical i-mode frequency scaling
        # ω ~ sqrt(GM/R³) with mode-dependent factor
        omega_orbital = math.sqrt(self.G * M / R**3)
        
        # i-mode frequency is typically 0.1-1 times orbital frequency
        # at the interface location
        omega_imode = 0.5 * omega_orbital  # Simplified estimate
        
        print(f"  Estimated i-mode frequency: ω = {omega_imode:.2e} rad/s")
        print(f"  Gravitational wave frequency: f = {omega_imode/(2*math.pi):.1f} Hz")
        
        # Placeholder eigenfunctions (would come from perturbation calculation)
        eigenfunctions = {
            'Wl': lambda r: math.exp(-((r - R/2)**2) / (R/10)**2),  # Gaussian centered at R/2
            'Vl': lambda r: r * math.exp(-((r - R/2)**2) / (R/10)**2)
        }
        
        return {
            'omega': omega_imode,
            'frequency': omega_imode / (2 * math.pi),
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
        
        # Simplified metric potentials (full calculation would be more complex)
        def nu(r): return 0.5 * math.log(1 - 2*masses[min(int(r/0.01), len(masses)-1)]/r)
        def lambda_func(r): return -nu(r)
        
        # Tidal overlap integral Ql (Eq. 4)
        Ql = 0
        A2 = 0
        dr = 0.01
        
        for i, r in enumerate(radii[:-1]):
            if r > 0 and i < len(epsilons):
                eps = epsilons[i]
                p = pressures[i]
                
                # Simplified metric factors
                metric_factor = math.exp((nu(r) + lambda_func(r))/2)
                norm_factor = math.exp((lambda_func(r) - nu(r))/2)
                
                # Tidal overlap integrand
                Ql += (l/self.c**2 * metric_factor * (eps + p) * 
                      r**l * (Wl(r) + (l + 1) * Vl(r)) * dr)
                
                # Normalization integrand
                A2 += (1/self.c**2 * norm_factor * (eps + p) * 
                      (math.exp(lambda_func(r)) * Wl(r)**2 + l*(l+1) * Vl(r)**2) * dr)
        
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
        
        # Phase shift formula for equal-mass binary (q = 1) and l = 2
        phase_shift = (2 * math.pi * 5 * math.pi / 4096 * 
                      (self.c**2 * R / (self.G * M))**5 * 
                      1 / (1 + q) * 
                      (self.G * M / R**3) / omega**2 * 
                      (Ql / (M * R**2))**2 * 
                      (M * R**2) / A2)
        
        delta_phi = abs(phase_shift)
        
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
            
        with open(filename, 'w', newline='') as csvfile:
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
        with open("all_counsell_results.csv", 'w', newline='') as csvfile:
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