import matplotlib.pyplot as plt

# -----------------------
# Data (replace with yours)
# -----------------------
radius = [0,0.05,0.1,0.15,0.2,0.25,0.3]

# Curve 1: Satisfaction probability (left axis)
satisfaction = [0.991,0.989,0.986,0.964,0.744,0.151,0.006]

# Curve 2: Energy (right axis)
energy = [43.115,42.141,41.707,40.39,38.437,37.157,35.521]

# -----------------------
# Plot
# -----------------------
fig, ax1 = plt.subplots()

# Left axis: satisfaction
ax1.set_xlabel("Sphere Radius")
ax1.set_ylabel("Satisfaction Guanratee")
line1 = ax1.plot(radius, satisfaction, marker='o', color='blue', label="Satisfaction")
ax1.set_ylim(0, 1)

# Right axis: energy
ax2 = ax1.twinx()
ax2.set_ylabel("Energy Consumption")
line2 = ax2.plot(radius, energy, marker='s', color='red', label="Energy")

# -----------------------
# Combined legend
# -----------------------
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="best")

# ax2.axhline(y=43.134, color='red', linestyle='--', label='Energy threshold')

plt.title("Satisfaction vs Energy across Radius")
plt.tight_layout()
plt.savefig("radius-energy-comparison.png", dpi=300, bbox_inches="tight")
plt.show()