#!/usr/bin/env python3
"""
Display sample data from the first two columns (epsilon and p) of EoS files.
"""

import csv
import glob


def show_data_sample():
    """Show sample data from all EoS files."""
    csv_files = glob.glob("EoS_*.csv")
    
    for csv_file in csv_files:
        print(f"\n{'='*80}")
        print(f"File: {csv_file}")
        print(f"{'='*80}")
        
        # Read and display data
        with open(csv_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            data = list(reader)
        
        print(f"Total data points: {len(data)}")
        print("\nFirst 10 data points:")
        print(f"{'Index':<6} {'Epsilon (MeV/fm³)':<20} {'Pressure (MeV/fm³)':<20}")
        print("-" * 50)
        
        for i, row in enumerate(data[:10]):
            values = list(row.values())
            epsilon = float(values[0])
            pressure = float(values[1])
            print(f"{i+1:<6} {epsilon:<20.3e} {pressure:<20.3e}")
        
        print("\nEvery 100th data point (to see range):")
        print(f"{'Index':<6} {'Epsilon (MeV/fm³)':<20} {'Pressure (MeV/fm³)':<20}")
        print("-" * 50)
        
        for i in range(0, len(data), max(1, len(data)//10)):
            values = list(data[i].values())
            epsilon = float(values[0])
            pressure = float(values[1])
            print(f"{i+1:<6} {epsilon:<20.3e} {pressure:<20.3e}")
        
        # Show last few points
        print("\nLast 5 data points:")
        print(f"{'Index':<6} {'Epsilon (MeV/fm³)':<20} {'Pressure (MeV/fm³)':<20}")
        print("-" * 50)
        
        for i, row in enumerate(data[-5:], len(data)-4):
            values = list(row.values())
            epsilon = float(values[0])
            pressure = float(values[1])
            print(f"{i:<6} {epsilon:<20.3e} {pressure:<20.3e}")
        
        # Statistics
        epsilon_vals = [float(list(row.values())[0]) for row in data]
        pressure_vals = [float(list(row.values())[1]) for row in data]
        
        print(f"\nStatistics:")
        print(f"Epsilon range:  {min(epsilon_vals):.3e} - {max(epsilon_vals):.3e} MeV/fm³")
        print(f"Pressure range: {min(pressure_vals):.3e} - {max(pressure_vals):.3e} MeV/fm³")
        print(f"Epsilon ratio (max/min): {max(epsilon_vals)/min(epsilon_vals):.1e}")
        print(f"Pressure ratio (max/min): {max(pressure_vals)/min(pressure_vals):.1e}")


if __name__ == "__main__":
    show_data_sample()