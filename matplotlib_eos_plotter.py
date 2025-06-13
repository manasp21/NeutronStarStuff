#!/usr/bin/env python3
"""
Professional EoS plotting using matplotlib.
"""

import sys
import os
sys.path.insert(0, '/home/themanaspandey/.local/lib/python3.12/site-packages')

import matplotlib.pyplot as plt
import numpy as np
import csv
import glob
from typing import List, Dict, Tuple


def load_eos_data(filepath: str) -> List[Dict[str, float]]:
    """Load EoS data from CSV file."""
    data = []
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            values = list(row.values())
            data.append({
                'epsilon': float(values[0]),
                'p': float(values[1]),
                'n': float(values[2]),
                'mu': float(values[3])
            })
    return data


def plot_eos_comparison():
    """Create comprehensive EoS comparison plots."""
    # Find all EoS files
    csv_files = glob.glob("EoS_*.csv")
    
    if not csv_files:
        print("No EoS CSV files found")
        return
    
    print(f"Found {len(csv_files)} EoS files:")
    for f in csv_files:
        print(f"  - {f}")
    
    # Load all data
    datasets = []
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for i, csv_file in enumerate(csv_files):
        try:
            data = load_eos_data(csv_file)
            model_name = os.path.basename(csv_file).replace('EoS_', '').replace('_Doroshenko.csv', '')
            # Shorten the name for better display
            short_name = model_name.replace('DD2F80_B=65_set244_HG075_eta005_eta016_', 'DD2F80_')
            short_name = short_name.replace('DD2MEV_p80_CSS1_0.7_ncrit_0.192606_ecrit_185.223_pcrit_10.3131', 'DD2MEV')
            
            datasets.append({
                'name': short_name,
                'full_name': model_name,
                'data': data,
                'color': colors[i % len(colors)]
            })
            print(f"Loaded {short_name}: {len(data)} points")
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    if not datasets:
        print("No data loaded successfully")
        return
    
    # Create comprehensive plot
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Pressure vs Energy Density (log-log)
    ax1 = plt.subplot(2, 3, 1)
    for dataset in datasets:
        data = dataset['data']
        epsilon_vals = [row['epsilon'] for row in data]
        p_vals = [row['p'] for row in data]
        plt.loglog(epsilon_vals, p_vals, label=dataset['name'], 
                  color=dataset['color'], linewidth=2, alpha=0.8)
    
    plt.xlabel('Energy Density ε [MeV/fm³]')
    plt.ylabel('Pressure p [MeV/fm³]')
    plt.title('Equation of State: p vs ε (log-log)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 2: Pressure vs Energy Density (linear, low density)
    ax2 = plt.subplot(2, 3, 2)
    for dataset in datasets:
        data = dataset['data']
        # Filter for energy densities < 100 MeV/fm³
        filtered_data = [row for row in data if row['epsilon'] < 100]
        if filtered_data:
            epsilon_vals = [row['epsilon'] for row in filtered_data]
            p_vals = [row['p'] for row in filtered_data]
            plt.plot(epsilon_vals, p_vals, label=dataset['name'], 
                    color=dataset['color'], linewidth=2, alpha=0.8)
    
    plt.xlabel('Energy Density ε [MeV/fm³]')
    plt.ylabel('Pressure p [MeV/fm³]')
    plt.title('EoS (Linear Scale, ε < 100 MeV/fm³)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 3: Speed of Sound Squared
    ax3 = plt.subplot(2, 3, 3)
    for dataset in datasets:
        data = dataset['data']
        sorted_data = sorted(data, key=lambda x: x['epsilon'])
        
        cs2_vals = []
        eps_mid = []
        for i in range(1, len(sorted_data)):
            deps = sorted_data[i]['epsilon'] - sorted_data[i-1]['epsilon']
            dp = sorted_data[i]['p'] - sorted_data[i-1]['p']
            if deps > 0:
                cs2 = dp / deps
                cs2_vals.append(cs2)
                eps_mid.append((sorted_data[i]['epsilon'] + sorted_data[i-1]['epsilon']) / 2)
        
        if cs2_vals:
            plt.semilogx(eps_mid, cs2_vals, label=dataset['name'], 
                        color=dataset['color'], linewidth=2, alpha=0.8)
    
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.7, label='c² = 1 (causal limit)')
    plt.xlabel('Energy Density ε [MeV/fm³]')
    plt.ylabel('Speed of Sound Squared c_s²')
    plt.title('Speed of Sound')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, 1.2)
    
    # Plot 4: Chemical Potential vs Number Density
    ax4 = plt.subplot(2, 3, 4)
    for dataset in datasets:
        data = dataset['data']
        n_vals = [row['n'] for row in data]
        mu_vals = [row['mu'] for row in data]
        plt.semilogx(n_vals, mu_vals, label=dataset['name'], 
                    color=dataset['color'], linewidth=2, alpha=0.8)
    
    plt.xlabel('Number Density n [1/fm³]')
    plt.ylabel('Chemical Potential μ [MeV]')
    plt.title('Chemical Potential vs Number Density')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 5: Pressure vs Number Density
    ax5 = plt.subplot(2, 3, 5)
    for dataset in datasets:
        data = dataset['data']
        n_vals = [row['n'] for row in data]
        p_vals = [row['p'] for row in data]
        plt.loglog(n_vals, p_vals, label=dataset['name'], 
                  color=dataset['color'], linewidth=2, alpha=0.8)
    
    plt.xlabel('Number Density n [1/fm³]')
    plt.ylabel('Pressure p [MeV/fm³]')
    plt.title('Pressure vs Number Density')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 6: Energy Density vs Number Density
    ax6 = plt.subplot(2, 3, 6)
    for dataset in datasets:
        data = dataset['data']
        n_vals = [row['n'] for row in data]
        eps_vals = [row['epsilon'] for row in data]
        plt.loglog(n_vals, eps_vals, label=dataset['name'], 
                  color=dataset['color'], linewidth=2, alpha=0.8)
    
    plt.xlabel('Number Density n [1/fm³]')
    plt.ylabel('Energy Density ε [MeV/fm³]')
    plt.title('Energy Density vs Number Density')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('eos_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Comprehensive plot saved as 'eos_comprehensive_analysis.png'")


def plot_phase_transition_analysis():
    """Look for phase transitions in detail."""
    csv_files = glob.glob("EoS_*.csv")
    
    if not csv_files:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ['blue', 'red', 'green']
    
    for i, csv_file in enumerate(csv_files[:3]):  # Plot first 3 files
        data = load_eos_data(csv_file)
        model_name = os.path.basename(csv_file).replace('EoS_', '').replace('_Doroshenko.csv', '')
        short_name = model_name.replace('DD2F80_B=65_set244_HG075_eta005_eta016_', 'DD2F80_')
        short_name = short_name.replace('DD2MEV_p80_CSS1_0.7_ncrit_0.192606_ecrit_185.223_pcrit_10.3131', 'DD2MEV')
        
        sorted_data = sorted(data, key=lambda x: x['epsilon'])
        
        epsilon_vals = [row['epsilon'] for row in sorted_data]
        p_vals = [row['p'] for row in sorted_data]
        
        # Calculate derivatives
        cs2_vals = []
        eps_mid = []
        for j in range(1, len(sorted_data)):
            deps = sorted_data[j]['epsilon'] - sorted_data[j-1]['epsilon']
            dp = sorted_data[j]['p'] - sorted_data[j-1]['p']
            if deps > 0:
                cs2 = dp / deps
                cs2_vals.append(cs2)
                eps_mid.append((sorted_data[j]['epsilon'] + sorted_data[j-1]['epsilon']) / 2)
        
        # Plot 1: Linear scale to see fine structure
        axes[0, 0].plot(epsilon_vals, p_vals, label=short_name, 
                       color=colors[i], linewidth=2, alpha=0.8)
        
        # Plot 2: Focus on potential transition region
        if len(epsilon_vals) > 100:
            # Look at middle section where transitions might occur
            start_idx = len(epsilon_vals) // 4
            end_idx = 3 * len(epsilon_vals) // 4
            
            eps_section = epsilon_vals[start_idx:end_idx]
            p_section = p_vals[start_idx:end_idx]
            
            axes[0, 1].plot(eps_section, p_section, label=short_name, 
                           color=colors[i], linewidth=2, alpha=0.8)
        
        # Plot 3: Speed of sound
        if cs2_vals:
            axes[1, 0].semilogx(eps_mid, cs2_vals, label=short_name, 
                               color=colors[i], linewidth=2, alpha=0.8)
        
        # Plot 4: Second derivative (curvature)
        if len(cs2_vals) > 1:
            d2p_de2 = []
            eps_mid2 = []
            for j in range(1, len(cs2_vals)):
                if eps_mid[j] - eps_mid[j-1] > 0:
                    d2 = (cs2_vals[j] - cs2_vals[j-1]) / (eps_mid[j] - eps_mid[j-1])
                    d2p_de2.append(abs(d2))
                    eps_mid2.append((eps_mid[j] + eps_mid[j-1]) / 2)
            
            if d2p_de2:
                axes[1, 1].semilogx(eps_mid2, d2p_de2, label=short_name, 
                                   color=colors[i], linewidth=2, alpha=0.8)
    
    # Format subplots
    axes[0, 0].set_xlabel('Energy Density ε [MeV/fm³]')
    axes[0, 0].set_ylabel('Pressure p [MeV/fm³]')
    axes[0, 0].set_title('Pressure vs Energy Density (Linear)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    axes[0, 1].set_xlabel('Energy Density ε [MeV/fm³]')
    axes[0, 1].set_ylabel('Pressure p [MeV/fm³]')
    axes[0, 1].set_title('Middle Section (Potential Transition Region)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    axes[1, 0].axhline(y=1, color='black', linestyle='--', alpha=0.7)
    axes[1, 0].set_xlabel('Energy Density ε [MeV/fm³]')
    axes[1, 0].set_ylabel('c_s²')
    axes[1, 0].set_title('Speed of Sound Squared')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    axes[1, 1].set_xlabel('Energy Density ε [MeV/fm³]')
    axes[1, 1].set_ylabel('|d²p/dε²|')
    axes[1, 1].set_title('Curvature (Second Derivative)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('eos_phase_transition_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Phase transition analysis saved as 'eos_phase_transition_analysis.png'")


def main():
    """Main plotting function."""
    print("Creating EoS plots with matplotlib...")
    
    # Set matplotlib backend for non-interactive environment
    plt.switch_backend('Agg')  # Use non-interactive backend
    
    try:
        plot_eos_comparison()
        plot_phase_transition_analysis()
        
        print("\nPlots created successfully:")
        print("  - eos_comprehensive_analysis.png")
        print("  - eos_phase_transition_analysis.png")
        
    except Exception as e:
        print(f"Error creating plots: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()