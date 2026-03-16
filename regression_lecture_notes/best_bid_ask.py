import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Synthetic order book data
# -------------------------
ask_prices = np.array([1.22, 1.23, 1.24, 1.25, 1.26])
ask_sizes  = np.array([200,  400,  750, 1200,  650])

bid_prices = np.array([1.20, 1.19, 1.18, 1.17, 1.16])
bid_sizes  = np.array([240,  400,  800,  650,  450])

best_ask = ask_prices.min()
best_bid = bid_prices.max()
mid = 0.5 * (best_ask + best_bid)
spread = best_ask - best_bid

# -------------------------
# Plot
# -------------------------
fig, ax = plt.subplots(figsize=(12, 6))

# bar thickness in "price" units
h = 0.008

# Ask bars (right side, positive)
ax.barh(ask_prices, ask_sizes, height=h, align="center", label="ASK depth")
# Bid bars (left side, negative)
ax.barh(bid_prices, -bid_sizes, height=h, align="center", label="BID depth")

# Center line (price axis in the middle)
ax.axvline(0, linewidth=1)

# Labels on bars: price near the center, size near the end
for p, s in zip(ask_prices, ask_sizes):
    ax.text(20, p, f"{p:.2f}", va="center")          # price label
    ax.text(s + 50, p, f"{s}", va="center")          # size label

for p, s in zip(bid_prices, bid_sizes):
    ax.text(-20, p, f"{p:.2f}", va="center", ha="right")
    ax.text(-s - 50, p, f"{s}", va="center", ha="right")

# -------------------------
# Spread arrow (SHORTER)
# -------------------------
# Put the arrow near the best prices, not spanning the whole plot region
arrow_x = 400  # move left/right (increase -> more to the right)
ax.annotate(
    "", xy=(arrow_x, best_ask), xytext=(arrow_x, best_bid),
    arrowprops=dict(arrowstyle="<->", lw=2, color="black")
)


# Spread text next to the arrow
ax.text(arrow_x + 60, mid, f"Spread = {spread:.2f}", va="center", fontsize=25, weight="bold")

# -------------------------
# Best bid/ask/mid text (SHIFT RIGHT)
# -------------------------
text_x = 1500  # increase -> move further right
ax.text(text_x, best_ask, f"Best ask = {best_ask:.2f}", va="center", fontsize=15)
ax.text(text_x, mid,      f"Mid = {mid:.2f}", va="center", fontsize=15)
ax.text(text_x, best_bid, f"Best bid = {best_bid:.2f}", va="center", fontsize=15)

# Formatting
ax.set_title("Order Book Ladder and Bid–Ask Spread (Synthetic Example)", fontsize=30)
ax.set_xlabel("Depth / size (left = bid, right = ask)", fontsize=25)
ax.set_ylabel("Price", fontsize=25)

# Eksen üzerindeki sayıların boyutunu büyütür
ax.tick_params(axis='both', which='major', labelsize=10)

# Limits: give room for right-side text
max_depth = max(ask_sizes.max(), bid_sizes.max())
ax.set_xlim(-max_depth - 300, max_depth + 900)

# Tight y-range around shown prices
ax.set_ylim(bid_prices.min() - 0.02, ask_prices.max() + 0.02)

ax.grid(True, alpha=0.25)
ax.legend(loc="upper right", fontsize=16)
plt.tight_layout()

# Save: PDF for LaTeX (crisp), PNG also ok
plt.savefig("orderbook_ladder_spread.pdf", bbox_inches="tight")
plt.savefig("orderbook_ladder_spread.png", dpi=200, bbox_inches="tight")
plt.show()