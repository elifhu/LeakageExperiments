import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm


def plot_residuals(residuals):
    """Plot residual histogram and Q–Q plot with scientific figure titles."""

    # Histogram vs Normal PDF
    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=40, density=True, alpha=0.6, color="gray", label="Empirical Residuals")

    # Theoretical normal distribution
    x = np.linspace(residuals.min(), residuals.max(), 200)
    p = stats.norm.pdf(x, residuals.mean(), residuals.std())
    plt.plot(x, p, "r", linewidth=2, label="Theoretical Normal PDF")

    plt.title("Distribution of Regression Residuals vs Normal PDF", fontsize=12, pad=10)
    plt.xlabel("Residuals (εᵢ, bps)", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Q–Q plot
    sm.qqplot(residuals, line="45", fit=True)
    plt.title("Q–Q Plot of Regression Residuals vs Theoretical Normal Quantiles", fontsize=12, pad=10)
    plt.xlabel("Theoretical Quantiles (Normal Dist.)", fontsize=11)
    plt.ylabel("Empirical Quantiles (Residuals, bps)", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()