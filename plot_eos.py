#!/usr/bin/env python3
"""
Plot EoS data using matplotlib or simple ASCII plotting as fallback.
"""

import csv
import glob
import os
import math
from typing import List, Dict


from eos_counsell_processor import CounsellEoSProcessor


def load_eos_data(model_name: str) -> List[Dict[str, float]]:
    """Load EoS data from CSV file."""
    csv_files = glob.glob("EoS_*.csv")
    target_file = None
    
    for csv_file in csv_files:
        if model_name in csv_file:
            target_file = csv_file
            break
    
    if not target_file:
        raise ValueError(f"Model {model_name} not found")
    
    data = []
    with open(target_file, 'r', encoding='utf-8-sig') as f:
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


def try_matplotlib_plot(data_sets: List[tuple]):
    """Try to create plots using matplotlib."""
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Pressure vs Energy density (log-log)
        for model_name, data in data_sets:
            epsilon_vals = [row['epsilon'] for row in data]
            p_vals = [row['p'] for row in data]
            ax1.loglog(epsilon_vals, p_vals, label=model_name, alpha=0.8, linewidth=2)
        
        ax1.set_xlabel('Energy Density ε [MeV/fm³]')
        ax1.set_ylabel('Pressure p [MeV/fm³]')
        ax1.set_title('Equation of State: p vs ε')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Linear scale for low-density region
        for model_name, data in data_sets:
            # Filter for lower densities to see any structure
            filtered_data = [row for row in data if row['epsilon'] < 1000]  # MeV/fm³
            if filtered_data:
                epsilon_vals = [row['epsilon'] for row in filtered_data]
                p_vals = [row['p'] for row in filtered_data]
                ax2.plot(epsilon_vals, p_vals, label=model_name, alpha=0.8, linewidth=2)
        
        ax2.set_xlabel('Energy Density ε [MeV/fm³]')
        ax2.set_ylabel('Pressure p [MeV/fm³]')
        ax2.set_title('EoS (Linear Scale, ε < 1000 MeV/fm³)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('eos_plots.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("Plot saved as 'eos_plots.png'")
        return True
        
    except ImportError:
        print("matplotlib not available, falling back to ASCII plot")
        return False
    except Exception as e:
        print(f"Error with matplotlib: {e}")
        return False


def ascii_plot(x_vals: List[float], y_vals: List[float], title: str, 
               x_label: str, y_label: str, width: int = 80, height: int = 20):
    """Create a simple ASCII plot."""
    if not x_vals or not y_vals:
        print("No data to plot")
        return
    
    # Use log scale
    log_x = [math.log10(max(x, 1e-20)) for x in x_vals]
    log_y = [math.log10(max(y, 1e-20)) for y in y_vals]
    
    min_x, max_x = min(log_x), max(log_x)
    min_y, max_y = min(log_y), max(log_y)
    
    if max_x == min_x or max_y == min_y:
        print("Insufficient data range for plotting")
        return
    
    # Create grid
    grid = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Plot points
    for lx, ly in zip(log_x, log_y):
        # Map to grid coordinates
        grid_x = int((lx - min_x) / (max_x - min_x) * (width - 1))
        grid_y = int((ly - min_y) / (max_y - min_y) * (height - 1))
        grid_y = height - 1 - grid_y  # Flip y-axis
        
        if 0 <= grid_x < width and 0 <= grid_y < height:
            grid[grid_y][grid_x] = '*'
    
    # Print plot
    print(f"\n{title}")
    print("=" * len(title))
    
    # Y-axis labels
    for i in range(height):
        y_val = min_y + (max_y - min_y) * (height - 1 - i) / (height - 1)
        y_label_str = f"{10**y_val:.1e}"
        print(f"{y_label_str:>10} |{''.join(grid[i])}")
    
    # X-axis
    print(" " * 11 + "+" + "-" * (width - 1))
    
    # X-axis labels
    x_positions = [0, width//4, width//2, 3*width//4, width-1]
    x_line = " " * 11
    for pos in x_positions:
        x_val = min_x + (max_x - min_x) * pos / (width - 1)
        x_label_str = f"{10**x_val:.1e}"
        x_line += f"{x_label_str:>15}"
    
    print(x_line)
    print(f"\n{x_label:^{width+11}}")
    print(f"{y_label:^10}")


def plot_eos_comparison():
    """Plot comparison of all available EoS models."""
    # Find available files
    csv_files = glob.glob("EoS_*.csv")
    
    if not csv_files:
        print("No EoS CSV files found in current directory")
        return
    
    print(f"Found {len(csv_files)} EoS files:")
    for f in csv_files:
        print(f"  - {f}")
    
    # Load data for all models
    data_sets = []
    for csv_file in csv_files:
        model_name = os.path.basename(csv_file).replace('EoS_', '').replace('_Doroshenko.csv', '')
        try:
            data = load_eos_data(model_name)
            data_sets.append((model_name, data))
            print(f"Loaded {model_name}: {len(data)} points")
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    if not data_sets:
        print("No data loaded successfully")
        return
    
    # Try matplotlib first
    if try_matplotlib_plot(data_sets):
        return
    
    # Fallback to ASCII plots
    print("\nCreating ASCII plots...")
    
    for model_name, data in data_sets:
        print(f"\n{'-'*60}")
        print(f"Model: {model_name}")
        print(f"{'-'*60}")
        
        epsilon_vals = [row['epsilon'] for row in data]
        p_vals = [row['p'] for row in data]
        
        ascii_plot(epsilon_vals, p_vals, 
                  f"Pressure vs Energy Density", 
                  "Energy Density ε [MeV/fm³] (log scale)",
                  "Pressure p [MeV/fm³] (log scale)")
        
        print(f"\nData range:")
        print(f"  ε: {min(epsilon_vals):.2e} - {max(epsilon_vals):.2e} MeV/fm³")
        print(f"  p: {min(p_vals):.2e} - {max(p_vals):.2e} MeV/fm³")


def plot_specific_model(model_name: str):
    """Plot a specific EoS model."""
    try:
        data = load_eos_data(model_name)
        print(f"Loaded {model_name}: {len(data)} points")
        
        epsilon_vals = [row['epsilon'] for row in data]
        p_vals = [row['p'] for row in data]
        
        # Try matplotlib
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 8))
            
            # Main plot (log-log)
            plt.subplot(2, 2, 1)
            plt.loglog(epsilon_vals, p_vals, 'b-', linewidth=2, alpha=0.8)
            plt.xlabel('Energy Density ε [MeV/fm³]')
            plt.ylabel('Pressure p [MeV/fm³]')
            plt.title(f'{model_name}: p vs ε (log-log)')
            plt.grid(True, alpha=0.3)
            
            # Linear plot for structure
            plt.subplot(2, 2, 2)
            plt.plot(epsilon_vals, p_vals, 'r-', linewidth=1, alpha=0.8)
            plt.xlabel('Energy Density ε [MeV/fm³]')
            plt.ylabel('Pressure p [MeV/fm³]')
            plt.title(f'{model_name}: p vs ε (linear)')
            plt.grid(True, alpha=0.3)
            
            # Speed of sound plot
            plt.subplot(2, 2, 3)
            cs2_vals = []
            eps_mid = []
            for i in range(1, len(data)):
                deps = data[i]['epsilon'] - data[i-1]['epsilon']
                dp = data[i]['p'] - data[i-1]['p']
                if deps > 0:
                    cs2 = dp / deps
                    cs2_vals.append(cs2)
                    eps_mid.append((data[i]['epsilon'] + data[i-1]['epsilon']) / 2)
            
            plt.semilogx(eps_mid, cs2_vals, 'g-', linewidth=1, alpha=0.8)
            plt.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='c² = 1 (causal limit)')
            plt.xlabel('Energy Density ε [MeV/fm³]')
            plt.ylabel('Speed of Sound Squared c_s²')
            plt.title(f'{model_name}: Speed of Sound')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Pressure derivative
            plt.subplot(2, 2, 4)
            plt.loglog(eps_mid, cs2_vals, 'm-', linewidth=1, alpha=0.8)
            plt.xlabel('Energy Density ε [MeV/fm³]')
            plt.ylabel('dp/dε')
            plt.title(f'{model_name}: Pressure Derivative')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            filename = f'eos_{model_name.replace("=", "_").replace(".", "_")}.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.show()
            
            print(f"Detailed plot saved as '{filename}'")
            
        except ImportError:
            # ASCII fallback
            ascii_plot(epsilon_vals, p_vals, 
                      f"EoS: {model_name}", 
                      "Energy Density ε [MeV/fm³] (log scale)",
                      "Pressure p [MeV/fm³] (log scale)")
            
    except Exception as e:
        print(f"Error plotting {model_name}: {e}")
    
        

def plot_mass_radius(eos_file: str):
    """Plot mass-radius relation for a given EoS file."""
    try:
        # Load EoS data using CounsellEoSProcessor
        processor = CounsellEoSProcessor(eos_file)
        
        # Load EoS data from .dat file with proper encoding
        try:
            data = processor.load_eos_data_from_dat(eos_file)
        except Exception as e:
            print(f"Error loading EoS data: {e}")
            return
        
        # Get stellar structure for 1.4 solar mass star
        structure = processor.step2_calculate_stellar_structure(data, stellar_mass=1.4)
        
        # Extract mass and radius profiles
        radii = structure['radii']
        masses_km = structure['masses']
        
        # Filter out negative and zero radii
        valid_indices = [i for i, r in enumerate(radii) if r > 0.1]
        radii = [radii[i] for i in valid_indices]
        masses_km = [masses_km[i] for i in valid_indices]
        
        # Convert masses to solar masses (M_sun_km = 1.476)
        solar_masses = [m / 1.476 for m in masses_km]
        
        # Plot using matplotlib
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(radii, solar_masses, 'b-', linewidth=2)
        plt.xlabel('Radius (km)')
        plt.ylabel('Mass (solar masses)')
        plt.title(f'Mass-Radius Profile for {eos_file}')
        plt.grid(True, alpha=0.3)
        plt.savefig('mass_radius_plot.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Plot saved as 'mass_radius_plot.png'")
    except Exception as e:
        print(f"Error generating mass-radius plot: {e}")

def main():
    """Main function."""
    import sys
    
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        plot_specific_model(model_name)
    else:
        # Generate mass-radius plot for the default EoS file
        eos_file = 'EoS_DD2MEV_p15_CSS1_1.4_ncrit_0.335561_ecrit_334.285_pcrit_42.1651.dat'
        plot_mass_radius(eos_file)


if __name__ == "__main__":
    main()