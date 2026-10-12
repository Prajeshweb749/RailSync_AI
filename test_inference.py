import numpy as np
from delay_model import simulate_leg_delay

# Test leg: 12484 (Amritsar Exp) from NDLS (New Delhi) to LDH (Ludhiana)
leg = ("12484", "NDLS", "LDH")

# 1. Standalone leg delay
delays = simulate_leg_delay(leg, n_samples=3000)
print(f"Standalone Leg {leg}:")
print(f"  Mean Delay : {np.mean(delays):.2f} min")
print(f"  Std Dev    : {np.std(delays):.2f} min")
print(f"  90th Pct   : {np.percentile(delays, 90):.2f} min")

# 2. Delay propagation with an incoming 25-minute delay at origin
incoming = np.full(3000, 25.0)
propagated_delays = simulate_leg_delay(leg, incoming_delay_samples=incoming, n_samples=3000)
print(f"\nPropagated Leg (25 min initial delay):")
print(f"  Mean Delay : {np.mean(propagated_delays):.2f} min")
print(f"  Std Dev    : {np.std(propagated_delays):.2f} min")
