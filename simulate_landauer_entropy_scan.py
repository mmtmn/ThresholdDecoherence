import numpy as np
import matplotlib.pyplot as plt
import os

# --------------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------------
Gamma = 1e9             # Hz
Gamma_max = 1e10        # Hz
epsilon2 = 1e-4         # Threshold for pruning
delta = 1e-5            # Steepness of sigmoid
k_B = 1.0               # Entropy units = bits
t_end = 10e-9           # Simulation time (seconds)
dt = 0.01e-9            # Time step
plot_dir = "figures"
os.makedirs(plot_dir, exist_ok=True)

# --------------------------------------------------------------------------------
# Sigmoid decoherence function
# --------------------------------------------------------------------------------
def g(x):
    return Gamma_max / (1 + np.exp((x - epsilon2) / delta))

# --------------------------------------------------------------------------------
# Entropy and evolution simulation
# --------------------------------------------------------------------------------
def simulate_entropy_cost(a2_init):
    t_vals = [0]
    a2_vals = [a2_init]
    entropy_vals = [0]

    a2 = a2_init
    t = 0
    while t < t_end:
        decay_rate = -2 * Gamma * g(a2) * a2
        a2_new = max(a2 + dt * decay_rate, 1e-20)
        da2_dt = (a2_new - a2) / dt
        entropy_rate = da2_dt**2 * np.log2(1 / (a2 + 1e-20))
        entropy_vals.append(entropy_vals[-1] + entropy_rate * dt)

        a2 = a2_new
        t += dt
        t_vals.append(t)
        a2_vals.append(a2)

    return t_vals, a2_vals, entropy_vals

# --------------------------------------------------------------------------------
# Sweep over a2_init values
# --------------------------------------------------------------------------------
a2_inits = np.logspace(-6, -1, 12)
delta_S_model = []
delta_S_landauer = []

for a2_init in a2_inits:
    _, _, entropy_vals = simulate_entropy_cost(a2_init)
    S_env = entropy_vals[-1]
    S_Landauer = k_B * np.log2(1 / a2_init)
    delta_S_model.append(S_env)
    delta_S_landauer.append(S_Landauer)

# --------------------------------------------------------------------------------
# PLOT: Simulated vs. Landauer entropy cost
# --------------------------------------------------------------------------------
log_info = -np.log2(a2_inits)

plt.figure(figsize=(8, 6))
plt.plot(log_info, delta_S_landauer, 'r--', label='Landauer Bound (kB log2(1/|a|²))')
plt.plot(log_info, delta_S_model, 'b-', marker='o', label='Simulated Entropy Cost')
plt.xlabel('Information Erased [bits]  =  -log₂(|a|²)')
plt.ylabel('ΔS_env (bits)')
plt.title('Entropy Cost vs. Information Erased')
plt.grid(True)
plt.legend()
plt.tight_layout()
save_path = os.path.join(plot_dir, "landauer_entropy_scan.png")
plt.savefig(save_path, dpi=300)
print(f"\n✅ Plot saved to: {save_path}")
plt.show()
