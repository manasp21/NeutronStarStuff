Here is a step-by-step guide in Markdown for reproducing the analysis needed to plot Figure 2 from the paper by Counsell et al. (https://www.google.com/search?q=2025).

# Guide to Reproducing Figure 2 from Counsell et al. (https://www.google.com/search?q=2025)

## Overview

This guide outlines the computational steps required to generate the data points for a plot similar to **Figure 2** in Counsell et al. [cite\_start](https://www.google.com/search?q=2025)[cite: 100]. [cite\_start]Each data point ($f, |\\Delta\\Phi|$) on the plot represents the signature of an interfacial mode (i-mode) from a neutron star with a first-order phase transition[cite: 3]. [cite\_start]The color of the point is determined by the strength of this transition, $\\Delta\\epsilon/\\epsilon\_i$[cite: 101]. The process involves stellar structure calculations, solving perturbation equations, and calculating the resulting gravitational-wave signature.

-----

## Step-by-Step Guide

### Step 1: Identify the Phase Transition in Your EoS

First, you must analyze your Equation of State (EoS) data to locate and quantify the first-order phase transition.

1.  **Locate the transition**: Plot pressure `p` versus energy density `ε`. [cite\_start]A first-order phase transition manifests as a region where `p` remains constant ($p=p\_{\\text{trans}}$) while `ε` increases from a value $\\epsilon\_1$ to $\\epsilon\_2$[cite: 25, 78].
2.  [cite\_start]**Calculate the energy density jump ($\\Delta\\epsilon$)**: This is the magnitude of the discontinuity in energy density[cite: 58].
    $$
    $$$$\\Delta\\epsilon = \\epsilon\_2 - \\epsilon\_1
    $$
    $$$$
    $$
3.  [cite\_start]**Determine the relative jump**: The color axis in Figure 2 represents the relative jump in energy density, $\\Delta\\epsilon/\\epsilon\_i$[cite: 101]. This is calculated as:
    $$
    $$$$\\frac{\\Delta\\epsilon}{\\epsilon\_i}
    $$
    $$$$where $\\epsilon\_i$ is the energy density at the start of the interface, i.e., $\\epsilon\_i = \\epsilon\_1$.

-----

### Step 2: Calculate Stellar Structure

[cite\_start]For a chosen stellar mass **`M`** (e.g., 1.4 $M\_{\\odot}$), solve the general relativistic **Tolman-Oppenheimer-Volkoff (TOV) equations** to determine the star's equilibrium structure[cite: 251].

  * **Inputs**: Your EoS in the form `p(ε)`.
  * **Outputs**: The star's total radius **`R`** and the internal profiles for pressure `p(r)`, energy density `ε(r)`, and the metric potentials `ν(r)` and `λ(r)`.

-----

### Step 3: Calculate the i-Mode Oscillation

This is the most complex computational step. [cite\_start]You must solve the general relativistic fluid perturbation equations for the stellar model obtained in Step 2 to find the i-mode[cite: 84, 85]. [cite\_start]The paper uses the **relativistic Cowling approximation**[cite: 86].

  * [cite\_start]**Inputs**: The stellar profiles (`p(r)`, `ε(r)`, etc.) and the speed of sound profile $c\_s^2(r) = dp/d\\epsilon$[cite: 75].
  * [cite\_start]**Outputs**: The eigenfrequency **`ω`** of the interfacial mode and its corresponding radial and angular eigenfunctions, **$W\_l(r)$** and **$V\_l(r)$**[cite: 257].
  * [cite\_start]**Identification**: The i-mode is identified by its characteristic eigenfunctions, which exhibit a sharp "kink" at the radial location of the phase transition[cite: 263, 268].

-----

### Step 4: Calculate Tidal Overlap ($Q\_l$) and Normalization ($\\mathcal{A}^2$)

[cite\_start]Using the i-mode eigenfunctions and the background stellar structure, compute the tidal overlap integral $Q\_l$ and the mode normalization constant $\\mathcal{A}^2$ for the `l=2` mode[cite: 50, 104].

  * [cite\_start]**Tidal Overlap Integral $Q\_l$** (from Eq. 4)[cite: 90]:
    $$
    $$$$Q\_l = \\frac{l}{c^2} \\int\_{0}^{R} e^{(\\nu+\\lambda)/2} (\\epsilon+p) r^l [W\_l + (l+1)V\_l] dr
    $$
    $$$$
    $$
  * [cite\_start]**Mode Normalization $\\mathcal{A}^2$** (from Eq. 5)[cite: 91]:
    $$
    $$$$\\mathcal{A}^2 = \\frac{1}{c^2} \\int\_{0}^{R} e^{(\\lambda-\\nu)/2} (\\epsilon+p) [e^\\lambda W\_l^2 + l(l+1)V\_l^2] dr
    $$
    $$$$
    $$

-----

### Step 5: Calculate Plot Coordinates ($f$, $|\\Delta\\Phi|$)

With all components calculated, you can now find the final coordinates for your plot.

1.  [cite\_start]**Gravitational-Wave Frequency (`f`)**: The GW frequency of the resonance is related to the mode frequency by `ω ≈ 2πf`[cite: 94].
    $$
    $$$$f = \\frac{\\omega}{2\\pi}
    $$
    $$$$
    $$
2.  [cite\_start]**Orbital Phase Shift ($|\\Delta\\Phi|$)**: This is the primary observable, estimated using **Equation (6)** in the paper[cite: 95]. For an equal-mass binary (`q=M'/M=1`) and the `l=2` mode, the formula is:
    $$
    $$$$|\\Delta\\Phi| \\approx 2\\pi \\times \\frac{5\\pi}{4096} \\left(\\frac{c^2R}{GM}\\right)^5 \\frac{1}{(1+1)} \\frac{GM/R^3}{\\omega^2} \\left(\\frac{Q\_2}{MR^2}\\right)^2 \\frac{MR^2}{\\mathcal{A}^2}
    $$
    $$$$
    $$

-----

### Step 6: Plot the Results

1.  [cite\_start]Create a scatter plot, typically with logarithmic axes, with the GW frequency `f` on the x-axis and the phase shift $|\\Delta\\Phi|$ on the y-axis[cite: 100].
2.  [cite\_start]For each data point corresponding to a specific EoS, set its color based on the value of $\\Delta\\epsilon/\\epsilon\_i$ calculated in Step 1[cite: 101].
3.  (Optional) To complete the plot, you can overlay the detector sensitivity curves. [cite\_start]These are calculated using **Equation (7)**[cite: 98]:
    $$
    $$$$|\\Delta\\Phi(f)| = \\frac{\\sqrt{S\_n(f)}}{2A(f)\\sqrt{f}}
    $$
    $$$$[cite\_start]where $S\_n(f)$ is the detector's noise power spectral density and $A(f)$ is the gravitational-wave amplitude for a binary at a given distance (the paper uses 40 Mpc)[cite: 98, 106].