# ThresholdDecoherence

This repository contains simulation and plotting scripts supporting the theoretical investigation of **threshold-based amplitude-selective decoherence** in open quantum systems. The proposed model explores how sub-threshold quantum amplitudes \( |\Psi(s)|^2 < \epsilon^2 \) may undergo accelerated collapse via feedback-activated Lindblad dynamics, while preserving linear and CPTP evolution at the system–environment (SE) level.

The simulations here are part of a broader study aiming to examine the physical plausibility, mathematical consistency, and falsifiability of threshold-based damping as an emergent measurement-feedback mechanism.

---

## Repository Structure

### Simulation Scripts

| Script | Description |
|--------|-------------|
| `simulate_threshold_pruning.py` | Simulates amplitude evolution under ideal threshold-based damping. Computes entropy cost using a Landauer-inspired heuristic. |
| `simulate_threshold_feedback_noise.py` | Adds random perturbations to the feedback activation function \( g(|\Psi|^2) \) to assess robustness under feedback uncertainty. |
| `simulate_entangled_threshold_decay.py` | Applies threshold damping to one qubit in an entangled Bell pair and verifies no-signaling by observing the unaffected qubit’s local purity. |
| `simulate_landauer_entropy_scan.py` | Scans a range of initial amplitudes to evaluate simulated entropy cost against the Landauer bound \( k_B \log(1/p) \). |

---

### Output Figures

All generated plots are saved to the `figures/` directory.

| Figure | Description |
|--------|-------------|
| `pruning_results.png` | Shows amplitude collapse and entropy accumulation in a two-level system under ideal threshold damping. |
| `feedback_robustness.png` | Demonstrates that the collapse and entropy spike persist under moderate feedback noise. |
| `entangled_decay_comparison.png` | Confirms that threshold damping applied to one qubit in an entangled pair does not induce signaling in the other. |
| `landauer_entropy_scan.png` | Compares entropy costs from simulation to the Landauer limit across a range of initial amplitudes. |

---

## Usage

Dependencies:
- Python 3.8+
- NumPy
- Matplotlib

Run simulations individually:

```bash
python simulate_threshold_pruning.py
