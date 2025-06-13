# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository contains neutron star physics data, specifically Equations of State (EoS) datasets. The data includes energy density (epsilon), pressure (p), number density (n), and chemical potential (mu) values for different neutron star models.

## Data File Structure

The repository contains two types of data files:
- **CSV files**: Comma-separated values with headers (`epsilon [MeV/fm^3],p [MeV/fm^3],n [1/fm^3],mu [MeV]`)
- **DAT files**: Tab-separated values without headers, same column order as CSV files

Data files follow the naming convention: `EoS_[model]_[parameters]_Doroshenko.[csv|dat]`

Examples:
- `EoS_DD2F80_B=65_set244_HG075_eta005_eta016_μm=860_Doroshenko.csv`
- `EoS_DD2MEV_p80_CSS1_0.7_ncrit_0.192606_ecrit_185.223_pcrit_10.3131_Doroshenko.csv`

## Data Columns

1. **epsilon**: Energy density in MeV/fm³
2. **p**: Pressure in MeV/fm³  
3. **n**: Number density in 1/fm³
4. **mu**: Chemical potential in MeV

## Processing Code

- `eos_counsell_processor.py`: Implements the complete 5-step methodology from Counsell et al. (2025) for analyzing neutron star interface modes and phase transitions
- `pdf_reader.py`: Utility for extracting text from PDF files

## Analysis Steps (Following Counsell et al. 2025)

1. **Step 1**: Identify first-order phase transitions in EoS data by looking for constant pressure regions
2. **Step 2**: Solve Tolman-Oppenheimer-Volkoff (TOV) equations for stellar structure
3. **Step 3**: Calculate i-mode oscillations using relativistic perturbation theory
4. **Step 4**: Compute tidal overlap integrals (Ql) and mode normalization (A²)
5. **Step 5**: Calculate gravitational wave frequency (f) and orbital phase shift (|ΔΦ|)

## Commands

```bash
# Run full Counsell et al. analysis
python3 eos_counsell_processor.py

# Create detailed matplotlib plots
PYTHONPATH=/home/themanaspandey/.local/lib/python3.12/site-packages python3 matplotlib_eos_plotter.py

# Run basic EoS analysis
python3 eos_analyzer.py

# Show data samples
python3 show_data_sample.py

# Extract text from PDF (if needed)
python3 pdf_reader.py EoS_step_by_step.pdf
```

## Git Configuration

Line ending handling is configured in `.gitattributes` to use LF endings for all text files to avoid CRLF warnings on Windows/WSL systems.

## Documentation

- `EoS_step_by_step.pdf`: Contains the detailed 5-step methodology from Counsell et al. (2025) for reproducing Figure 2 analysis of neutron star interface modes