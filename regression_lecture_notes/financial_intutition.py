import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Synthetic "market" data
# -----------------------
np.random.seed(7)
T = 260  # ~1 trading year
t = np.arange(T)

# Price: geometric random walk with a volatility regime shift
sigma = 0.008 + 0.010*(t > 140)   # higher vol later
r = 0.0002 + sigma * np.random.randn(T)  # "returns"
price = 100 * np.exp(np.cumsum(r))        # price from returns

# Simple returns (for display)
simple_ret = np.diff(price) / price[:-1]
simple_ret = np.concatenate([[0.0], simple_ret])  # align lengths

# Volatility: rolling std of returns (window)
w = 20
roll_vol = np.full(T, np.nan)
for i in range(w-1, T):
    roll_vol[i] = np.std(simple_ret[i-w+1:i+1], ddof=1)

# Volume: spiky daily volume (lognormal + occasional bursts)
base_vol = np.random.lognormal(mean=12.0, sigma=0.35, size=T)
spikes = (np.random.rand(T) < 0.06) * np.random.lognormal(mean=13.0, sigma=0.25, size=T)
volume = base_vol + spikes

# Liquidity / Depth proxies:
# bid-ask spread widens when volatility is high; depth drops when spread widens
spread = 0.8 + 30*np.nan_to_num(roll_vol, nan=np.nanmean(roll_vol))  # in "bps" scale-ish
spread += 0.2*np.random.randn(T)
spread = np.clip(spread, 0.4, None)

depth = 1.0 / (spread + 0.3)  # higher spread -> lower depth (proxy)
depth = depth / np.nanmax(depth)  # normalize to [0,1]

# -----------------------
# Plot: 2x2 panel
# -----------------------
fig, axs = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)

# (a) Price + Return
ax = axs[0, 0]
ax.plot(t, price, label="Price")
ax2 = ax.twinx()
ax2.plot(t, simple_ret, linestyle="--", alpha=0.6, label="Return")
ax.set_title("(a) Price / Return")
ax.set_xlabel("Time")
ax.set_ylabel("Price")
ax2.set_ylabel("Return")
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines + lines2, labels + labels2, loc="upper left", frameon=False)

# (b) Volatility (rolling std)
ax = axs[0, 1]
ax.plot(t, roll_vol)
ax.set_title("(b) Volatility (rolling std)")
ax.set_xlabel("Time")
ax.set_ylabel("Rolling σ")

# (c) Volume (bar chart)
ax = axs[1, 0]
ax.bar(t, volume, width=1.0)
ax.set_title("(c) Volume")
ax.set_xlabel("Time")
ax.set_ylabel("Volume")

# (d) Liquidity / Depth (spread + depth proxy)
ax = axs[1, 1]
ax.plot(t, spread, label="Bid–ask spread (proxy)")
ax2 = ax.twinx()
ax2.plot(t, depth, linestyle="--", alpha=0.8, label="Depth / liquidity (proxy)")
ax.set_title("(d) Liquidity / Depth")
ax.set_xlabel("Time")
ax.set_ylabel("Spread")
ax2.set_ylabel("Depth (norm.)")
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines + lines2, labels + labels2, loc="upper left", frameon=False)

# Save for LaTeX
fig.savefig("finance_microstructure_2x2.png", dpi=300)
fig.savefig("finance_microstructure_2x2.pdf")
plt.show()