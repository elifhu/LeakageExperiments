import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Generate synthetic data
T = 260
t = np.arange(T)

# Price process
price = 100 + np.cumsum(np.random.normal(0, 0.6, T))

# Volatility (rolling std proxy)
returns = np.diff(price, prepend=price[0])
volatility = np.convolve(np.abs(returns), np.ones(20)/20, mode='same')

# Liquidity (bid / ask depth)
bid_depth = -np.random.randint(50000, 150000, T)
ask_depth =  np.random.randint(50000, 150000, T)

# Plot
fig, axes = plt.subplots(
    2, 1, figsize=(10, 7),
    gridspec_kw={'height_ratios': [1, 1.1]},
    sharex=True
)

# Price + Volatility
ax1 = axes[0]
ax1.plot(price, color='#1f77b4', linewidth=2, label='Price')
ax1.set_ylabel("Price", color='#1f77b4', fontsize=30)
ax1.tick_params(axis='y', labelcolor='#1f77b4')

ax1b = ax1.twinx()
ax1b.plot(volatility, color='#ff7f0e', linestyle='--', linewidth=2, label='Volatility')
ax1b.set_ylabel("Volatility", color='#ff7f0e', fontsize=30)
ax1b.tick_params(axis='y', labelcolor='#ff7f0e')

ax1.set_title("(a) Price and Volatility", fontsize=30)

# Legend
lines = ax1.get_lines() + ax1b.get_lines()
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="upper left")

# Liquidity
ax2 = axes[1]
ax2.bar(t, bid_depth, color='#ff7f0e', alpha=0.85, label='Bid liquidity')
ax2.bar(t, ask_depth, color='#1f77b4', alpha=0.85, label='Ask liquidity')

ax2.axhline(0, color='black', linewidth=0.8)
ax2.set_ylabel("Liquidity (signed)", fontsize=30)
ax2.set_xlabel("Time", fontsize=30)
ax2.set_title("(b) Market Liquidity (Bid / Ask)", fontsize=30)
ax2.legend(loc="upper right")

plt.tight_layout()
plt.show()

# Save for LaTeX
fig.savefig("return_liquidity_illustration.png", dpi=300)
fig.savefig("return_liquidity_illustration.pdf")
plt.show()