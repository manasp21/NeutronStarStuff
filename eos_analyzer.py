#!/usr/bin/env python3
"""
Simple EoS Data Analyzer

Provides basic analysis and visualization of neutron star equation of state data
without external dependencies.
"""

import csv
import glob
import os
import math
from typing import List, Dict, Tuple


class SimpleEoSAnalyzer:
    """Simple analyzer for EoS data using only standard library."""
    
    def __init__(self, data_dir: str = "."):
        self.data_dir = data_dir
        self.columns = ['epsilon', 'p', 'n', 'mu']
    
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
    
    def load_eos_data(self, model_name: str, file_type: str = 'csv') -> List[Dict[str, float]]:
        """Load EoS data from file."""
        files = self.find_eos_files()
        
        if model_name not in files:
            raise ValueError(f"Model {model_name} not found")
        
        if file_type == 'csv':
            return self.read_csv_data(files[model_name]['csv'])
        elif file_type == 'dat':
            return self.read_dat_data(files[model_name]['dat'])
    
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
        """Read DAT data."""
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
    
    def analyze_phase_transitions(self, data: List[Dict[str, float]]) -> Dict:
        """Analyze for potential phase transitions."""
        print("Analyzing for phase transitions...")
        
        # Sort by energy density
        sorted_data = sorted(data, key=lambda x: x['epsilon'])
        
        # Calculate pressure derivative dp/dε (speed of sound squared)
        cs2_values = []
        for i in range(1, len(sorted_data)):
            deps = sorted_data[i]['epsilon'] - sorted_data[i-1]['epsilon']
            dp = sorted_data[i]['p'] - sorted_data[i-1]['p']
            if deps > 0:
                cs2 = dp / deps
                cs2_values.append(cs2)
            else:
                cs2_values.append(0)
        
        # Look for pressure plateaus
        plateaus = []
        tolerance = 1e-4  # Relative tolerance
        
        i = 0
        while i < len(sorted_data) - 5:  # Need at least 5 points for a plateau
            current_p = sorted_data[i]['p']
            plateau_start = i
            
            # Count consecutive points with similar pressure
            while (i < len(sorted_data) - 1 and
                   abs(sorted_data[i + 1]['p'] - current_p) / max(current_p, 1e-10) < tolerance):
                i += 1
            
            plateau_end = i
            plateau_length = plateau_end - plateau_start
            
            if plateau_length >= 5:  # Significant plateau
                epsilon_start = sorted_data[plateau_start]['epsilon']
                epsilon_end = sorted_data[plateau_end]['epsilon']
                delta_epsilon = epsilon_end - epsilon_start
                
                if delta_epsilon / epsilon_start > 0.001:  # At least 0.1% jump
                    plateaus.append({
                        'start_idx': plateau_start,
                        'end_idx': plateau_end,
                        'length': plateau_length,
                        'pressure': current_p,
                        'epsilon_start': epsilon_start,
                        'epsilon_end': epsilon_end,
                        'delta_epsilon': delta_epsilon,
                        'relative_jump': delta_epsilon / epsilon_start
                    })
            
            i += 1
        
        # Analyze speed of sound
        cs2_analysis = {
            'min': min(cs2_values) if cs2_values else 0,
            'max': max(cs2_values) if cs2_values else 0,
            'mean': sum(cs2_values) / len(cs2_values) if cs2_values else 0,
            'causal_violations': len([cs2 for cs2 in cs2_values if cs2 > 1])
        }
        
        return {
            'plateaus': plateaus,
            'cs2_analysis': cs2_analysis,
            'total_points': len(data)
        }
    
    def print_detailed_analysis(self, model_name: str, data: List[Dict[str, float]]):
        """Print detailed analysis of EoS data."""
        print(f"\n{'='*60}")
        print(f"Detailed Analysis: {model_name}")
        print(f"{'='*60}")
        
        print(f"Total data points: {len(data)}")
        
        # Basic statistics
        for col in self.columns:
            values = [row[col] for row in data]
            print(f"\n{col}:")
            print(f"  Range: {min(values):.3e} - {max(values):.3e}")
            print(f"  Mean:  {sum(values)/len(values):.3e}")
        
        # Phase transition analysis
        transition_analysis = self.analyze_phase_transitions(data)
        
        print(f"\nPhase Transition Analysis:")
        print(f"  Pressure plateaus found: {len(transition_analysis['plateaus'])}")
        
        for i, plateau in enumerate(transition_analysis['plateaus']):
            print(f"  Plateau {i+1}:")
            print(f"    Pressure: {plateau['pressure']:.3e} MeV/fm³")
            print(f"    ε range: {plateau['epsilon_start']:.3e} - {plateau['epsilon_end']:.3e}")
            print(f"    Δε: {plateau['delta_epsilon']:.3e} ({plateau['relative_jump']:.1%})")
            print(f"    Length: {plateau['length']} points")
        
        print(f"\nSpeed of Sound Analysis:")
        cs2 = transition_analysis['cs2_analysis']
        print(f"  c_s² range: {cs2['min']:.3f} - {cs2['max']:.3f}")
        print(f"  c_s² mean: {cs2['mean']:.3f}")
        print(f"  Causal violations (c_s² > 1): {cs2['causal_violations']}")
        
        # Look for discontinuities in derivatives
        print(f"\nDerivative Analysis:")
        sorted_data = sorted(data, key=lambda x: x['epsilon'])
        
        # Calculate d²p/dε² to find inflection points
        d2p_de2_values = []
        for i in range(2, len(sorted_data)):
            eps = [sorted_data[j]['epsilon'] for j in range(i-2, i+1)]
            p_vals = [sorted_data[j]['p'] for j in range(i-2, i+1)]
            
            # Second derivative approximation
            if eps[2] - eps[0] > 0:
                d2p_de2 = 2 * (p_vals[2] - 2*p_vals[1] + p_vals[0]) / (eps[2] - eps[0])**2
                d2p_de2_values.append(abs(d2p_de2))
        
        if d2p_de2_values:
            inflection_threshold = sum(d2p_de2_values) / len(d2p_de2_values) + 3 * (
                sum([(x - sum(d2p_de2_values)/len(d2p_de2_values))**2 
                     for x in d2p_de2_values]) / len(d2p_de2_values))**0.5
            
            inflection_points = [i for i, val in enumerate(d2p_de2_values) 
                               if val > inflection_threshold]
            print(f"  Potential inflection points: {len(inflection_points)}")
        
        print(f"{'='*60}\n")
    
    def compare_models(self, model_names: List[str]):
        """Compare multiple EoS models."""
        print(f"\n{'='*60}")
        print(f"Model Comparison")
        print(f"{'='*60}")
        
        model_data = {}
        for model_name in model_names:
            try:
                data = self.load_eos_data(model_name)
                model_data[model_name] = data
                print(f"Loaded {model_name}: {len(data)} points")
            except Exception as e:
                print(f"Failed to load {model_name}: {e}")
        
        # Compare pressure ranges at similar densities
        print(f"\nPressure comparison at key densities:")
        test_densities = [1e-3, 1e-2, 1e-1, 0.5]  # 1/fm³
        
        for test_n in test_densities:
            print(f"\nAt n = {test_n} fm⁻³:")
            for model_name, data in model_data.items():
                # Find closest density point
                closest_point = min(data, key=lambda x: abs(x['n'] - test_n))
                if abs(closest_point['n'] - test_n) / test_n < 0.1:  # Within 10%
                    print(f"  {model_name}: p = {closest_point['p']:.3e} MeV/fm³")
    
    def export_analysis_summary(self, filename: str = "eos_analysis_summary.csv"):
        """Export analysis summary for all models."""
        files = self.find_eos_files()
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'Model', 'DataPoints', 'MinEpsilon', 'MaxEpsilon', 'MinPressure', 'MaxPressure',
                'PlateauCount', 'MaxRelativeJump', 'CausalViolations'
            ])
            
            for model_name in files.keys():
                try:
                    data = self.load_eos_data(model_name)
                    transition_analysis = self.analyze_phase_transitions(data)
                    
                    epsilon_vals = [row['epsilon'] for row in data]
                    p_vals = [row['p'] for row in data]
                    
                    max_jump = (max([p['relative_jump'] for p in transition_analysis['plateaus']], default=0))
                    
                    writer.writerow([
                        model_name,
                        len(data),
                        min(epsilon_vals),
                        max(epsilon_vals),
                        min(p_vals),
                        max(p_vals),
                        len(transition_analysis['plateaus']),
                        max_jump,
                        transition_analysis['cs2_analysis']['causal_violations']
                    ])
                    
                except Exception as e:
                    print(f"Error analyzing {model_name}: {e}")
        
        print(f"Analysis summary exported to {filename}")


def main():
    """Main function."""
    analyzer = SimpleEoSAnalyzer()
    
    # Find available models
    files = analyzer.find_eos_files()
    print("Available EoS models:")
    for model_name in files.keys():
        print(f"  - {model_name}")
    
    if not files:
        print("No EoS files found")
        return
    
    # Analyze each model in detail
    for model_name in files.keys():
        try:
            data = analyzer.load_eos_data(model_name)
            analyzer.print_detailed_analysis(model_name, data)
        except Exception as e:
            print(f"Error analyzing {model_name}: {e}")
    
    # Compare models
    model_names = list(files.keys())
    if len(model_names) > 1:
        analyzer.compare_models(model_names)
    
    # Export summary
    analyzer.export_analysis_summary()


if __name__ == "__main__":
    main()