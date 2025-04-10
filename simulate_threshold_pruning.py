# main8.py — Final Verdict Simulation for Threshold Decoherence
# Author: Thiago Munhoz da Nóbrega
# Description:
#   Tests amplitude-triggered decoherence via nonlinear Lindblad operators.
#   Confirms (or refutes) pruning behavior and entropy spike.

import numpy as np

# --------------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------------
Gamma = 1e9                   # Hz, base decoherence rate
Gamma_max = 5e10              # Hz, strong feedback coupling
epsilon2 = 1e-4               # Pruning threshold (|Psi|^2)
delta = 1e-6                  # Sigmoid transition width
k_B = 1.0                     # Boltzmann constant (entropy units = bits)
t_end = 10e-9                 # Total simulation time (s)
dt = 0.01e-9                  # Time step (s)
a2_init = 5e-5                # Initial amplitude squared

# Detection thresholds
collapse_threshold = 1e7            # Slope of |a|² for collapse (per second)
entropy_spike_threshold = 1e8       # Entropy spike rate (bits/s)
collapse_floor = 1e-12              # Value below which |a|² is considered collapsed

# --------------------------------------------------------------------------------
# FEEDBACK-BASED SIGMOID FUNCTION
# --------------------------------------------------------------------------------
def g(x):
    """Amplitude-dependent feedback coupling."""
    return Gamma_max / (1 + np.exp((x - epsilon2) / delta))

# --------------------------------------------------------------------------------
# SIMULATION LOOP
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
# FINAL VERDICT
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

# DECISION TREE
if entropy_spike_triggered and fully_collapsed:
    print("✅ VERDICT: Paper CONFIRMED — pruning behavior observed.")
elif entropy_spike_triggered and not fully_collapsed:
    print("⚠️ VERDICT: Entropy spike occurred, but amplitude didn't fully collapse.")
elif collapse_triggered and not entropy_spike_triggered:
    print("⚠️ VERDICT: Collapse occurred, but no strong entropy signature.")
else:
    print("❌ VERDICT: Paper DISPROVED — no pruning or entropy spike detected.")
