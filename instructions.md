Of course. [cite_start]Reproducing Figure 2 from the paper "Interface modes in inspiralling neutron stars: A gravitational-wave probe of first-order phase transitions" [cite: 1] is a complex, multi-step process that involves numerical relativity calculations. It goes beyond simple plotting and requires solving differential equations that describe the structure and oscillations of a neutron star.

Below is a detailed, step-by-step guide outlining the theoretical framework, equations, and computational steps required to generate such a plot using your own Equation of State (EoS) data.

***

### **Conceptual Overview**

The core of the task is to, for a given neutron star mass and your EoS:
1.  **Calculate the star's structure:** Determine its radius, mass, and internal pressure/density profile by solving the equations of general relativistic hydrostatic equilibrium.
2.  **Calculate the i-mode properties:** Solve the general relativistic perturbation equations to find the frequency (`ω`) and spatial form (eigenfunctions `W_l`, `V_l`) of the interface mode (`i-mode`) oscillation caused by the phase transition in your EoS.
3.  **Calculate the Observables:** Use the results from the previous steps to compute the two key observables plotted in Figure 2: the gravitational-wave frequency (`f`) and the gravitational-wave phase shift (`ΔΦ`).
4.  **Plot the Results:** Plot your calculated point (`f`, `|ΔΦ|`) and compare it against the theoretical sensitivity curves of gravitational-wave detectors.

[cite_start]The authors of the paper use a family of EoS models and repeat this process for each one to generate the collection of points seen in their figure[cite: 104]. You will do this for your own EoS.

***

### **Step 1: Calculate the Neutron Star Structure (TOV Equations)**

The first step is to model the structure of a non-rotating, spherically symmetric neutron star. This is done by solving the Tolman-Oppenheimer-Volkoff (TOV) equations for general relativistic hydrostatic equilibrium.

Your EoS data, which provides pressure `p` as a function of energy density `ε` (i.e., `p(ε)`), is the essential microphysical input here.

#### **Equations:**

The TOV equations are a set of two coupled first-order ordinary differential equations:

1.  **Equation for Mass:** This describes how the enclosed gravitational mass `m(r)` changes with radius `r`.
    $$\frac{dm(r)}{dr} = 4\pi r^2 \epsilon(r)$$

2.  **Equation for Pressure:** This is the relativistic equation for hydrostatic equilibrium.
    $$\frac{dp(r)}{dr} = - \frac{[\epsilon(r) + p(r)][m(r) + 4\pi r^3 p(r)]}{r[r - 2m(r)]}$$
    *(Note: These equations are often written in units where G=c=1. I will maintain this convention for consistency with the literature.)*

#### **Procedure:**

1.  **Select a Central Density:** Choose a value for the energy density at the center of the star, `ε_c = ε(r=0)`. The properties of the resulting star (its total mass and radius) will depend on this choice.
2.  **Set Initial Conditions:** At the center of the star (`r=0`), the initial conditions are:
    * `p(0) = p(ε_c)` (from your EoS data)
    * `m(0) = 0`
3.  **Numerical Integration:** Numerically integrate the two coupled TOV equations outwards from `r=0` to larger radii. At each step `r`, you use your EoS data to find the energy density `ε(r)` corresponding to the pressure `p(r)`.
4.  **Find the Surface:** The integration stops at the radius `R` where the pressure drops to zero, `p(R) = 0`. This `R` is the star's radius. The total mass of the star is `M = m(R)`.
5.  **Obtain Metric Potentials:** While integrating, you also solve for a component of the spacetime metric, `ν(r)`, which is needed later. It is given by:
    $$\frac{d\nu(r)}{dr} = - \frac{2}{\epsilon(r) + p(r)} \frac{dp(r)}{dr}$$
    The other relevant metric component is `λ(r)`, defined by `e^λ(r) = [1 - 2m(r)/r]⁻¹`.

To generate a family of stars (a mass-radius curve), you repeat this entire procedure for a range of different central densities `ε_c`. [cite_start]For this project, you will focus on a single star, for instance, a `1.4 M_☉` model, as used in Figure 2[cite: 100].

***

### **Step 2: Calculate the Interface Mode Properties**

With the background stellar structure (`p(r)`, `ε(r)`, `m(r)`, `ν(r)`) known, the next step is to calculate the properties of the oscillations. [cite_start]The paper uses the **relativistic Cowling approximation**, which simplifies the problem by ignoring perturbations of the spacetime metric[cite: 86].

You need to solve the equations for polar (spheroidal) oscillations of the stellar fluid. [cite_start]The fluid's displacement is described by two functions, `W_l(r)` (radial) and `V_l(r)` (tangential), for a given multipolar order `l` (the paper uses `l=2` [cite: 101]).

#### **Equations:**

The task is to solve a system of two coupled first-order differential equations for `W_l(r)` and `V_l(r)`. These equations are derived from the linearized relativistic Euler equations. [cite_start]The specific forms are given in Reference [4] of the paper's Supplemental Material [cite: 258][cite_start], which is Reference [39] in the main paper's bibliography[cite: 176]. They are:

$$\frac{dW_l}{dr} = ...$$
$$\frac{dV_l}{dr} = ...$$

Solving these equations is a boundary value problem. You are not just integrating them; you are searching for the specific oscillation frequency `ω` (the eigenvalue) that allows for a non-trivial solution satisfying the boundary conditions at the star's center and surface. This is typically done numerically using a "shooting method."

The **interface mode (i-mode)** you are looking for is a specific solution that is sensitive to the sharp density jump in your EoS. [cite_start]Its eigenfunction will have a characteristic sharp "kink" at the radial location of the phase transition[cite: 66, 268].

#### **Procedure:**

1.  [cite_start]**Implement the Oscillation Equations:** Code the differential equations for `W_l(r)` and `V_l(r)` from the literature (e.g., Ref. [39] [cite: 176]).
2.  **Numerical Search for `ω`:**
    * Guess a value for the mode frequency `ω`.
    * Integrate the oscillation equations from the center to the surface.
    * Check if the solution satisfies the boundary condition at the surface. It generally won't.
    * Adjust your guess for `ω` and repeat until the boundary conditions are met. The value of `ω` that works is the eigenfrequency of a mode.
3.  **Identify the i-mode:** You will find a spectrum of modes (f-modes, p-modes, etc.). [cite_start]The i-mode is the one whose frequency is directly related to the buoyancy force at the phase transition interface and typically lies between the fundamental f-mode (~2 kHz) and g-modes (~100 Hz)[cite: 107].
4.  **Store the Solution:** Once you find the i-mode frequency `ω` and its corresponding eigenfunctions `W_l(r)` and `V_l(r)`, you have completed the most difficult computational step.

***

### **Step 3: Calculate the Plotted Quantities (f, |ΔΦ|, Δε/εᵢ)**

Now you can use the results from the previous steps to calculate the values for the x-axis, y-axis, and color bar of the plot.

#### **1. Gravitational-Wave Frequency (`f`)**

The resonance that excites the mode occurs when the mode's frequency in the star's frame, `ω`, matches the frequency of the tidal forcing from the companion star. [cite_start]For an `l=2` mode, this happens when the gravitational-wave frequency `f` is given by[cite: 94]:
$$f = \frac{\omega}{ \pi}$$

#### **2. Phase Shift (`|ΔΦ|`)**

The paper uses a Newtonian-based estimate for the cumulative phase shift `ΔΦ` in the gravitational waveform due to the resonant energy transfer to the mode. [cite_start]The formula is given in Equation (6)[cite: 95]:

$$\Delta\Phi \approx -\frac{5\pi}{4096} \left(\frac{c^2 R}{GM}\right)^5 \frac{2}{q(1+q)} \frac{GM/R^3}{\omega^2} \left(\frac{Q_l}{MR^l}\right)^2 \frac{MR^2}{\mathcal{A}^2}$$

To use this, you need to calculate two intermediate quantities: the **tidal overlap integral `Q_l`** and the **mode normalization `A²`**.

* [cite_start]**Tidal Overlap `Q_l` (Equation 4):** This quantifies how well the tidal field couples to the oscillation mode[cite: 90].
    $$Q_l = \frac{1}{c^2} l \int_0^R e^{(\nu+\lambda)/2} (\epsilon+p) r^l [W_l(r) + (l+1)V_l(r)] dr$$

* [cite_start]**Mode Normalization `A²` (Equation 5):** This is a normalization constant for the mode's energy[cite: 91].
    $$\mathcal{A}^2 = \frac{1}{c^2} \int_0^R e^{(\lambda-\nu)/2} (\epsilon+p) [e^\lambda W_l(r)^2 + l(l+1)V_l(r)^2] dr$$

**Procedure:**
1.  **Calculate Integrals:** Using the stellar profile (`ε`, `p`, `ν`, `λ`) and the mode eigenfunctions (`W_l`, `V_l`) you found, numerically compute the integrals for `Q_l` and `A²`.
2.  **Calculate `ΔΦ`:** Substitute all the values (`M`, `R`, `ω`, `Q_l`, `A²`) into the formula for `ΔΦ`. [cite_start]The paper uses `l=2` and an equal-mass binary, so the mass ratio `q = M'/M = 1`[cite: 101, 106].

#### **3. Relative Energy Density Jump (`Δε/εᵢ`)**

[cite_start]This value for the color bar quantifies the strength of the first-order phase transition in your EoS[cite: 101].

**Procedure:**
1.  **Inspect your EoS data:** A first-order phase transition appears as a region of constant pressure `p` where the energy density `ε` jumps from a lower value `εᵢ` to a higher value.
2.  **Calculate the Jump:**
    * Identify the energy density at the beginning of the transition, `εᵢ`.
    * Identify the energy density at the end of the transition, `ε_f`.
    * The jump is `Δε = ε_f - εᵢ`.
    * The relative jump is `Δε / εᵢ`.

***

### **Step 4: Plotting and Detector Curves**

You now have a single data point: (`f`, `|ΔΦ|`), colored by `Δε/εᵢ`.

#### **Plotting Your Data:**

* Create a 2D scatter plot with a logarithmic x-axis (`f / Hz`) and a logarithmic y-axis (`|ΔΦ|`).
* Plot your calculated point on these axes.
* Use a colormap to color your point according to its `Δε/εᵢ` value.

#### **Detector Sensitivity Curves:**

To see if your i-mode resonance would be detectable, you compare it against the sensitivity curves of various detectors. [cite_start]The paper provides the formula for this minimum detectable phase shift in Equation (7)[cite: 98]:

$$|\Delta\Phi(f)|_{min} = \frac{\sqrt{S_n(f)}}{2A(f)\sqrt{f}}$$

* **`S_n(f)`:** This is the **noise power spectral density** (PSD) of a given detector (e.g., LIGO A+, Einstein Telescope, Cosmic Explorer). These are publicly available data files. You can find them by searching for, e.g., "Einstein Telescope noise curve data" or "Cosmic Explorer sensitivity curve data".
* **`A(f)`:** This is the amplitude of the gravitational-wave signal in the frequency domain. Calculating this requires a waveform model. [cite_start]The paper uses `IMRPhenomPv2_NRTidal` [cite: 102] [cite_start]for a binary at a luminosity distance of 40 Mpc[cite: 106]. Generating this requires specialized gravitational-wave software packages like `LALSuite`.

**Procedure:**
1.  **Obtain PSDs:** Download the noise PSD data files for the detectors of interest.
2.  **Calculate `A(f)`:** Use a package like `LALSuite` in Python to generate the waveform amplitude `A(f)` for a standard source (e.g., two `1.4 M_☉` neutron stars at 40 Mpc).
3.  **Calculate `|ΔΦ(f)|_min`:** For each detector, use the formula above to compute the minimum detectable phase shift as a function of frequency `f`.
4.  **Overlay on Plot:** Plot these curves `|ΔΦ(f)|_{min}` vs. `f` on your graph. If your calculated point lies above a detector's curve, it is considered detectable by that instrument.

### **Summary of Packages and Tools**

There is no single "package" that performs this entire analysis. It requires a custom pipeline of code.

* **TOV Solver & Mode Calculation:** This requires custom code. You would use standard numerical libraries for integration and root-finding (e.g., `scipy.integrate.solve_ivp` and `scipy.optimize.brentq` in Python) but you must implement the physics equations yourself.
* **Waveform Amplitude & PSDs:** The `LALSuite` library is the standard tool in the gravitational-wave community for these tasks. Some functionality may also be available in higher-level packages like `PyCBC` or `Bilby`.
* **Plotting:** Standard libraries like `Matplotlib` or `Seaborn` in Python are perfectly suitable for creating the final plot.