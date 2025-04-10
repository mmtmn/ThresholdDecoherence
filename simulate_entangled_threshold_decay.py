"""
simulate_entangled_threshold_decay.py

Tests whether threshold-based amplitude pruning applied to qubit A affects
qubit B in a Bell state (entanglement). Compares threshold decoherence
to standard Lindblad damping. Tracks amplitude decay on A and purity decay of B.

Outputs:
- Amplitude decay on qubit A
- Purity (Tr[rho_B^2]) of reduced state of qubit B
- Saves plot to: figures/entangled_decay_comparison.png
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# --------------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------------
Gamma = 1e9              # Hz
Gamma_max = 1e10         # Max damping under feedback
epsilon2 = 1e-4          # Threshold for pruning
delta = 1e-6             # Sigmoid sharpness
dt = 0.01e-9             # Time step (s)
t_end = 10e-9            # Total simulation time (s)
output_dir = "figures"
output_file = os.path.join(output_dir, "entangled_decay_comparison.png")

# --------------------------------------------------------------------------------
# Threshold damping function (sigmoid)
# --------------------------------------------------------------------------------
def g(x):
    return Gamma_max / (1 + np.exp((x - epsilon2) / delta))

# --------------------------------------------------------------------------------
# Initialize Bell state: (|00⟩ + |11⟩)/√2
# Full system has 4 components in basis: |00⟩, |01⟩, |10⟩, |11⟩
# --------------------------------------------------------------------------------
def bell_state_density_matrix():
    psi = np.zeros(4, dtype=complex)
    psi[0] = 1/np.sqrt(2)  # |00⟩
    psi[3] = 1/np.sqrt(2)  # |11⟩
    return np.outer(psi, psi.conj())  # ρ = |ψ⟩⟨ψ|

# --------------------------------------------------------------------------------
# Partial trace over qubit A to get reduced state of B
# --------------------------------------------------------------------------------
def partial_trace_A(rho):
    # rho: 4x4 density matrix of 2 qubits
    # Returns 2x2 density matrix for qubit B
    rho_B = np.zeros((2, 2), dtype=complex)
    for i in range(2):
        for j in range(2):
            # Sum over qubit A basis indices (0 and 1)
            rho_B[i, j] = rho[i, j] + rho[i+2, j+2]
    return rho_B

# --------------------------------------------------------------------------------
# Purity of reduced state: Tr[rho^2]
# --------------------------------------------------------------------------------
def purity(rho):
    return np.real(np.trace(rho @ rho))

# --------------------------------------------------------------------------------
# Apply threshold damping to qubit A
# Only |00⟩ and |01⟩ components have qubit A in state |0⟩
# --------------------------------------------------------------------------------
def apply_threshold_damping(rho):
    # Extract population on qubit A = 0 subspace
    p_A0 = np.real(rho[0, 0] + rho[1, 1])
    damping = 2 * Gamma * g(p_A0) * p_A0 * dt

    # Damping acts only on |00⟩ and |01⟩ rows/cols
    damp_indices = [0, 1]
    rho_new = rho.copy()
    for i in damp_indices:
        for j in damp_indices:
            rho_new[i, j] *= (1 - damping)

    return rho_new

# --------------------------------------------------------------------------------
# Apply standard Lindblad damping (linear)
# Equivalent to exponential decay of population on qubit A
# --------------------------------------------------------------------------------
def apply_standard_lindblad(rho, rate):
    damp_indices = [0, 1]
    decay = 2 * rate * dt
    rho_new = rho.copy()
    for i in damp_indices:
        for j in damp_indices:
            rho_new[i, j] *= (1 - decay)
    return rho_new

# --------------------------------------------------------------------------------
# MAIN SIMULATION
# --------------------------------------------------------------------------------
def simulate(thresholded=True):
    t_vals = []
    a_pop = []  # population of qubit A in |0⟩ (p_A0)
    purities = []

    rho = bell_state_density_matrix()
    t = 0

    while t < t_end:
        p_A0 = np.real(rho[0, 0] + rho[1, 1])
        a_pop.append(p_A0)

        rho_B = partial_trace_A(rho)
        purities.append(purity(rho_B))

        rho = apply_threshold_damping(rho) if thresholded else apply_standard_lindblad(rho, Gamma)

        t += dt
        t_vals.append(t)

    return np.array(t_vals), np.array(a_pop), np.array(purities)

# --------------------------------------------------------------------------------
# Run simulations
# --------------------------------------------------------------------------------
t_thr, A_thr, P_thr = simulate(thresholded=True)
t_lin, A_lin, P_lin = simulate(thresholded=False)

# --------------------------------------------------------------------------------
# Save plot
# --------------------------------------------------------------------------------
os.makedirs(output_dir, exist_ok=True)
t_ns = t_thr * 1e9

plt.figure(figsize=(12, 5))

# Amplitude decay
plt.subplot(1, 2, 1)
plt.plot(t_ns, A_thr, label='Threshold Damping', lw=2)
plt.plot(t_ns, A_lin, label='Standard Decoherence', lw=2, linestyle='--')
plt.axhline(epsilon2, color='r', linestyle=':', label='Threshold ε²')
plt.xlabel('Time (ns)')
plt.ylabel('Pop. A in |0⟩')
plt.title('Amplitude Evolution (Qubit A)')
plt.legend()

# Entanglement decay (purity of qubit B)
plt.subplot(1, 2, 2)
plt.plot(t_ns, P_thr, label='Threshold Damping', lw=2)
plt.plot(t_ns, P_lin, label='Standard Decoherence', lw=2, linestyle='--')
plt.xlabel('Time (ns)')
plt.ylabel('Purity Tr[ρ_B²]')
plt.title('Entanglement Decay (Qubit B)')
plt.legend()

plt.tight_layout()
plt.savefig(output_file, dpi=300)
plt.close()

print("\n✅ Simulation complete.")
print(f"📁 Plot saved to: {output_file}")
