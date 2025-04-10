import numpy as np
import matplotlib.pyplot as plt
import os

# --------------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------------
Gamma = 1e9               # Base Lindblad rate (Hz)
Gamma_max = 1e10          # Max feedback damping rate (Hz)
epsilon2 = 1e-4           # Amplitude threshold
delta_ideal = 1e-6        # Ideal sharp sigmoid
delta_soft = 2e-5         # Softened sigmoid
sigma_noise = 1e-5        # Std dev of amplitude noise
dt = 0.01e-9              # Time step (s)
t_end = 10e-9             # Total simulation time (s)
k_B = 1.0                 # Normalized units
a2_init = 5e-5            # Initial |a|²
output_dir = "figures"
output_filename = os.path.join(output_dir, "feedback_robustness.png")

# --------------------------------------------------------------------------------
# Sigmoid function g(x)
# --------------------------------------------------------------------------------
def g(x, delta):
    return Gamma_max / (1 + np.exp((x - epsilon2) / delta))

# --------------------------------------------------------------------------------
# Simulation function
# --------------------------------------------------------------------------------
def run_simulation(delta, noisy=False):
    t_vals = [0]
    a2_vals = [a2_init]
    S_vals = [0]
    entropy_rate_vals = []

    a2 = a2_init
    t = 0

    while t < t_end:
        if noisy:
            x = max(a2 + np.random.normal(0, sigma_noise), 0)
        else:
            x = a2

        g_val = g(x, delta)
        decay = -2 * Gamma * g_val * a2
        a2_new = max(a2 + dt * decay, 0)

        da2_dt = (a2_new - a2) / dt
        entropy_rate = (da2_dt)**2 * np.log2(1 / (a2 + 1e-20))
        S_vals.append(S_vals[-1] + entropy_rate * dt)
        entropy_rate_vals.append(entropy_rate)

        a2 = a2_new
        t += dt

        a2_vals.append(a2)
        t_vals.append(t)

    return np.array(t_vals), np.array(a2_vals), np.array(S_vals), np.array(entropy_rate_vals)

# --------------------------------------------------------------------------------
# Run both simulations
# --------------------------------------------------------------------------------
t_ideal, a2_ideal, S_ideal, E_ideal = run_simulation(delta_ideal, noisy=False)
t_soft,  a2_soft,  S_soft,  E_soft  = run_simulation(delta_soft, noisy=True)

# --------------------------------------------------------------------------------
# Output summary
# --------------------------------------------------------------------------------
print(f"\n--- FEEDBACK ROBUSTNESS TEST ---")
print(f"Initial |a|² = {a2_init:.2e}, threshold ε² = {epsilon2:.2e}")
print(f"Ideal: Final |a|² = {a2_ideal[-1]:.2e}, Max entropy rate = {max(E_ideal):.2e} bits/s")
print(f"Noisy: Final |a|² = {a2_soft[-1]:.2e}, Max entropy rate = {max(E_soft):.2e} bits/s")

collapse_detected = a2_soft[-1] < 1e-10
entropy_spike = max(E_soft) > 1e8

if collapse_detected and entropy_spike:
    print("✅ Collapse + entropy spike survive under noise.")
elif collapse_detected:
    print("⚠️ Collapse detected, but entropy spike weak.")
elif entropy_spike:
    print("⚠️ Entropy spike without full collapse.")
else:
    print("❌ No robust pruning detected under noise.")

# --------------------------------------------------------------------------------
# Create output directory if needed
# --------------------------------------------------------------------------------
os.makedirs(output_dir, exist_ok=True)

# --------------------------------------------------------------------------------
# Plot and save
# --------------------------------------------------------------------------------
t_ns = t_ideal * 1e9

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(t_ns, a2_ideal, label='Ideal |a|²', lw=2)
plt.plot(t_ns, a2_soft, label='Noisy |a|²', lw=2, linestyle='--')
plt.axhline(epsilon2, color='r', linestyle=':', label='Threshold ε²')
plt.xlabel('Time (ns)')
plt.ylabel('|a|²')
plt.title('Amplitude Evolution')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(t_ns, S_ideal, label='Ideal Entropy (bits)', lw=2)
plt.plot(t_ns, S_soft, label='Noisy Entropy (bits)', lw=2, linestyle='--')
plt.xlabel('Time (ns)')
plt.ylabel('ΔS_env (bits)')
plt.title('Entropy Accumulation')
plt.legend()

plt.tight_layout()
plt.savefig(output_filename, dpi=300)
print(f"\n📁 Plot saved to: {output_filename}")
plt.close()
