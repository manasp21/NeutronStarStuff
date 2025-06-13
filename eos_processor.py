#!/usr/bin/env python3
"""
Neutron Star Equation of State (EoS) Data Processor

This module provides functionality to read and process neutron star EoS data
containing energy density (epsilon), pressure (p), number density (n), and 
chemical potential (mu) values.
"""

import pandas as pd
import numpy as np
import glob
import os
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt


class EoSDataProcessor:
    """Process neutron star equation of state data files."""
    
    def __init__(self, data_dir: str = "."):
        """Initialize the EoS processor.
        
        Args:
            data_dir: Directory containing the EoS data files
        """
        self.data_dir = data_dir
        self.columns = ['epsilon', 'p', 'n', 'mu']
        self.units = {
            'epsilon': 'MeV/fm³',
            'p': 'MeV/fm³', 
            'n': '1/fm³',
            'mu': 'MeV'
        }
    
    def find_eos_files(self) -> Dict[str, List[str]]:
        """Find all EoS data files in the directory.
        
        Returns:
            Dictionary with model names as keys and file paths as values
        """
        csv_files = glob.glob(os.path.join(self.data_dir, "EoS_*.csv"))
        dat_files = glob.glob(os.path.join(self.data_dir, "EoS_*.dat"))
        
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
    
    def read_csv_data(self, filepath: str) -> pd.DataFrame:
        """Read EoS data from CSV file.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            DataFrame with columns: epsilon, p, n, mu
        """
        df = pd.read_csv(filepath)
        df.columns = self.columns
        return df
    
    def read_dat_data(self, filepath: str) -> pd.DataFrame:
        """Read EoS data from DAT file.
        
        Args:
            filepath: Path to the DAT file
            
        Returns:
            DataFrame with columns: epsilon, p, n, mu
        """
        df = pd.read_csv(filepath, sep='\t', header=None)
        df.columns = self.columns
        return df
    
    def load_eos_data(self, model_name: str, file_type: str = 'csv') -> pd.DataFrame:
        """Load EoS data for a specific model.
        
        Args:
            model_name: Name of the EoS model
            file_type: Type of file to load ('csv' or 'dat')
            
        Returns:
            DataFrame containing the EoS data
        """
        files = self.find_eos_files()
        
        if model_name not in files:
            raise ValueError(f"Model {model_name} not found. Available models: {list(files.keys())}")
        
        if file_type == 'csv':
            return self.read_csv_data(files[model_name]['csv'])
        elif file_type == 'dat':
            if files[model_name]['dat'] is None:
                raise ValueError(f"DAT file not found for model {model_name}")
            return self.read_dat_data(files[model_name]['dat'])
        else:
            raise ValueError("file_type must be 'csv' or 'dat'")
    
    def calculate_derived_quantities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived thermodynamic quantities.
        
        Args:
            df: DataFrame with epsilon, p, n, mu columns
            
        Returns:
            DataFrame with additional derived quantities
        """
        result = df.copy()
        
        # Speed of sound squared (c_s^2 = dp/depsilon)
        result['cs2'] = np.gradient(df['p']) / np.gradient(df['epsilon'])
        
        # Enthalpy density (h = epsilon + p)
        result['h'] = df['epsilon'] + df['p']
        
        # Baryon mass density (rho_b = n * m_b, where m_b ≈ 938.3 MeV)
        m_b = 938.3  # MeV (average nucleon mass)
        result['rho_b'] = df['n'] * m_b
        
        # Adiabatic index (Gamma = (epsilon + p) / p * dp/depsilon)
        result['gamma'] = (df['epsilon'] + df['p']) / df['p'] * result['cs2']
        
        # Polytropic index (n = 1/(Gamma - 1))
        result['polytropic_n'] = 1 / (result['gamma'] - 1)
        
        return result
    
    def interpolate_at_density(self, df: pd.DataFrame, target_density: float) -> Dict[str, float]:
        """Interpolate EoS quantities at a specific density.
        
        Args:
            df: DataFrame with EoS data
            target_density: Target number density in 1/fm³
            
        Returns:
            Dictionary with interpolated values
        """
        if target_density < df['n'].min() or target_density > df['n'].max():
            raise ValueError(f"Target density {target_density} outside data range "
                           f"[{df['n'].min():.2e}, {df['n'].max():.2e}]")
        
        result = {}
        for col in df.columns:
            if col != 'n':
                result[col] = np.interp(target_density, df['n'], df[col])
        
        result['n'] = target_density
        return result
    
    def find_phase_transitions(self, df: pd.DataFrame, cs2_threshold: float = 1.0) -> List[int]:
        """Identify potential phase transitions based on speed of sound.
        
        Args:
            df: DataFrame with derived quantities
            cs2_threshold: Threshold for c_s^2 to identify transitions
            
        Returns:
            List of indices where phase transitions might occur
        """
        if 'cs2' not in df.columns:
            df = self.calculate_derived_quantities(df)
        
        # Find points where c_s^2 > threshold (potentially unphysical)
        violations = df[df['cs2'] > cs2_threshold].index.tolist()
        
        # Find discontinuities in the speed of sound
        cs2_diff = np.abs(np.diff(df['cs2']))
        discontinuities = np.where(cs2_diff > np.std(cs2_diff) * 3)[0]
        
        return sorted(set(violations + discontinuities.tolist()))
    
    def plot_eos(self, df: pd.DataFrame, model_name: str = "", show_derived: bool = True):
        """Plot the equation of state.
        
        Args:
            df: DataFrame with EoS data
            model_name: Name of the model for plot title
            show_derived: Whether to plot derived quantities
        """
        if show_derived and 'cs2' not in df.columns:
            df = self.calculate_derived_quantities(df)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Equation of State: {model_name}', fontsize=16)
        
        # Pressure vs Energy Density
        axes[0, 0].loglog(df['epsilon'], df['p'])
        axes[0, 0].set_xlabel('Energy Density (MeV/fm³)')
        axes[0, 0].set_ylabel('Pressure (MeV/fm³)')
        axes[0, 0].grid(True)
        
        # Chemical Potential vs Number Density
        axes[0, 1].semilogx(df['n'], df['mu'])
        axes[0, 1].set_xlabel('Number Density (1/fm³)')
        axes[0, 1].set_ylabel('Chemical Potential (MeV)')
        axes[0, 1].grid(True)
        
        if show_derived:
            # Speed of Sound
            axes[1, 0].semilogx(df['n'], df['cs2'])
            axes[1, 0].set_xlabel('Number Density (1/fm³)')
            axes[1, 0].set_ylabel('c_s² (c=1)')
            axes[1, 0].axhline(y=1, color='r', linestyle='--', alpha=0.7, label='c_s² = 1')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # Adiabatic Index
            axes[1, 1].semilogx(df['n'], df['gamma'])
            axes[1, 1].set_xlabel('Number Density (1/fm³)')
            axes[1, 1].set_ylabel('Adiabatic Index Γ')
            axes[1, 1].grid(True)
        else:
            # Remove unused subplots
            fig.delaxes(axes[1, 0])
            fig.delaxes(axes[1, 1])
        
        plt.tight_layout()
        return fig
    
    def compare_models(self, model_names: List[str], quantity: str = 'p'):
        """Compare a specific quantity across different EoS models.
        
        Args:
            model_names: List of model names to compare
            quantity: Quantity to plot ('p', 'epsilon', 'mu', 'cs2', etc.)
        """
        plt.figure(figsize=(10, 6))
        
        for model_name in model_names:
            try:
                df = self.load_eos_data(model_name)
                if quantity not in df.columns:
                    df = self.calculate_derived_quantities(df)
                
                if quantity in df.columns:
                    plt.loglog(df['n'], df[quantity], label=model_name, alpha=0.8)
                else:
                    print(f"Warning: {quantity} not available for {model_name}")
                    
            except Exception as e:
                print(f"Error loading {model_name}: {e}")
        
        plt.xlabel('Number Density (1/fm³)')
        plt.ylabel(f'{quantity} ({self.units.get(quantity, "")})')
        plt.title(f'Comparison of {quantity} across EoS models')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def export_processed_data(self, df: pd.DataFrame, model_name: str, output_format: str = 'csv'):
        """Export processed data with derived quantities.
        
        Args:
            df: DataFrame with processed data
            model_name: Name of the model
            output_format: Output format ('csv', 'hdf5', 'json')
        """
        if 'cs2' not in df.columns:
            df = self.calculate_derived_quantities(df)
        
        output_file = f"processed_{model_name}.{output_format}"
        
        if output_format == 'csv':
            df.to_csv(output_file, index=False)
        elif output_format == 'hdf5':
            df.to_hdf(output_file, key='eos_data', mode='w')
        elif output_format == 'json':
            df.to_json(output_file, orient='records', indent=2)
        else:
            raise ValueError("output_format must be 'csv', 'hdf5', or 'json'")
        
        print(f"Processed data exported to {output_file}")


def main():
    """Example usage of the EoS processor."""
    processor = EoSDataProcessor()
    
    # Find available models
    files = processor.find_eos_files()
    print("Available EoS models:")
    for model_name in files.keys():
        print(f"  - {model_name}")
    
    if files:
        # Load and process the first model
        model_name = list(files.keys())[0]
        print(f"\nProcessing model: {model_name}")
        
        # Load data
        df = processor.load_eos_data(model_name)
        print(f"Loaded {len(df)} data points")
        
        # Calculate derived quantities
        df_processed = processor.calculate_derived_quantities(df)
        print(f"Calculated derived quantities: {list(df_processed.columns)}")
        
        # Find phase transitions
        transitions = processor.find_phase_transitions(df_processed)
        if transitions:
            print(f"Potential phase transitions at indices: {transitions}")
        
        # Export processed data
        processor.export_processed_data(df_processed, model_name)
        
        # Plot the EoS
        processor.plot_eos(df_processed, model_name)
        plt.show()


if __name__ == "__main__":
    main()