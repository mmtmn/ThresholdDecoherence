# plot_pruning_results.py — Threshold Decoherence Final Verification and Visualization
# Author: Thiago Munhoz da Nóbrega
# Description:
#   Executes a verdict-level simulation and generates high-quality plots
#   confirming or refuting pruning behavior in amplitude-triggered Lindblad systems

import numpy as np
import matplotlib.pyplot as plt
import os

# --------------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------------
Gamma = 1e9                  # Hz, base decoherence rate
Gamma_max = 5e10             # Hz, strong feedback coupling
epsilon2 = 1e-4              # Pruning threshold
delta = 1e-6                 # Transition sharpness
k_B = 1.0                    # Entropy units (bits)
t_end = 10e-9                # Simulation duration (seconds)
dt = 0.01e-9                 # Time step (10 ps)
a2_init = 5e-5               # Initial |a|^2

# Collapse and entropy thresholds
collapse_threshold = 1e7           # |a|^2/time change to detect collapse
entropy_spike_threshold = 1e8      # bits/s
collapse_floor = 1e-12             # Define "fully collapsed"

# Output and plotting
PLOT = True
SAVE_FIGURE = True
FIGURE_DIR = "figures"
FIGURE_NAME = "plot_pruning_results_pruning_results"
DPI = 300

# --------------------------------------------------------------------------------
# SIGMOID DECOHERENCE FUNCTION
# --------------------------------------------------------------------------------
def g(x):
    return Gamma_max / (1 + np.exp((x - epsilon2) / delta))

# --------------------------------------------------------------------------------
# SIMULATION
# --------------------------------------------------------------------------------
a2_vals = [a2_init]
entropy_vals = [0]
entropy_rate_vals = []
t_vals = [0]

a2 = a2_init
t = 0
collapse_triggered = False
entropy_spike_triggered = False

print(f"\n--- PRUNING VERDICT SIMULATION ---")
print(f"Initial |a|² = {a2:.2e}")
print(f"Gamma_max = {Gamma_max:.2e}, ε² = {epsilon2:.1e}, δ = {delta:.1e}")

while t < t_end:
    decay_rate = -2 * Gamma * g(a2) * a2
    a2_new = max(a2 + dt * decay_rate, 0)
    da2_dt = (a2_new - a2) / dt

    entropy_rate = da2_dt**2 * np.log2(1 / (a2 + 1e-20))
    entropy_vals.append(entropy_vals[-1] + entropy_rate * dt)
    entropy_rate_vals.append(entropy_rate)

    if not collapse_triggered and a2 < epsilon2 and da2_dt < -collapse_threshold:
        print(f"⚡ Collapse triggered at t = {t*1e9:.2f} ns, |a|² = {a2:.2e}")
        collapse_triggered = True

    if not entropy_spike_triggered and entropy_rate > entropy_spike_threshold:
        print(f"🔥 Entropy spike at t = {t*1e9:.2f} ns, rate = {entropy_rate:.2e} bits/s")
        entropy_spike_triggered = True

    a2 = a2_new
    t += dt
    a2_vals.append(a2)
    t_vals.append(t)

# --------------------------------------------------------------------------------
# VERDICT AND FINAL VALUES
# --------------------------------------------------------------------------------
final_a2 = a2
min_a2 = min(a2_vals)
max_entropy_rate = max(entropy_rate_vals)
total_entropy = entropy_vals[-1]
fully_collapsed = final_a2 < collapse_floor

print("\n--- SIMULATION COMPLETE ---")
print(f"Final |a|² = {final_a2:.2e}")
print(f"Min |a|² reached = {min_a2:.2e}")
print(f"Max entropy rate = {max_entropy_rate:.2e} bits/s")
print(f"Total entropy produced = {total_entropy:.2f} bits")

if entropy_spike_triggered and fully_collapsed:
    print("✅ VERDICT: Paper CONFIRMED — pruning behavior observed.")
elif entropy_spike_triggered and not fully_collapsed:
    print("⚠️ VERDICT: Entropy spike occurred, but amplitude didn't fully collapse.")
elif collapse_triggered and not entropy_spike_triggered:
    print("⚠️ VERDICT: Collapse occurred, but no strong entropy signature.")
else:
    print("❌ VERDICT: Paper DISPROVED — no pruning or entropy spike detected.")

# --------------------------------------------------------------------------------
# PLOTTING
# --------------------------------------------------------------------------------
if PLOT or SAVE_FIGURE:
    t_vals_ns = np.array(t_vals) * 1e9
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Amplitude evolution
    axes[0].plot(t_vals_ns, a2_vals, lw=2, label=r'$|a(t)|^2$')
    axes[0].axhline(epsilon2, color='r', linestyle='--', label='Threshold $\epsilon^2$')
    axes[0].set_xlabel('Time (ns)', fontsize=12)
    axes[0].set_ylabel(r'$|a|^2$', fontsize=12)
    axes[0].set_title('Amplitude Evolution')
    axes[0].legend()
    axes[0].grid(True)

    # Entropy accumulation
    axes[1].plot(t_vals_ns, entropy_vals, lw=2, color='darkgreen', label=r'$\Delta S_{\mathrm{env}}$')
    axes[1].set_xlabel('Time (ns)', fontsize=12)
    axes[1].set_ylabel('Entropy (bits)', fontsize=12)
    axes[1].set_title('Entropy Accumulation')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    if SAVE_FIGURE:
        os.makedirs(FIGURE_DIR, exist_ok=True)
        fig.savefig(os.path.join(FIGURE_DIR, f"{FIGURE_NAME}.png"), dpi=DPI)
        fig.savefig(os.path.join(FIGURE_DIR, f"{FIGURE_NAME}.pdf"))
        print(f"\n📁 Figures saved to: {FIGURE_DIR}/{FIGURE_NAME}.(png, pdf)")

    if PLOT:
        plt.show()
