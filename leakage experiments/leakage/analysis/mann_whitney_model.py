import pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns


def mann_whitney_reval_test(df, skew_col='skew', reval_col='reval', plot=True):
    """
    Compare reval distributions across positive and negative skew regimes
    using the Mann–Whitney U test.

    Parameters:
        df : pd.DataFrame
            DataFrame containing 'skew' and 'reval' columns.
        skew_col : str
            Column name for skew values.
        reval_col : str
            Column name for reval values (in bps).
        plot : bool
            Whether to plot the distributions.

    Returns:
        result : dict
            Dictionary containing U statistic, p-value, medians, and sample sizes.
    """
    # Split data into positive and negative skew groups
    pos_reval = df[df[skew_col] > 0][reval_col].dropna()
    neg_reval = df[df[skew_col] < 0][reval_col].dropna()

    # Mann–Whitney U test
    U, p_value = mannwhitneyu(pos_reval, neg_reval, alternative='two-sided')

    # Optional visualisation
    if plot:
        plt.figure(figsize=(8, 5))
        sns.violinplot(data=[pos_reval, neg_reval], inner="box", palette=["#3B9AE1", "#E15759"])
        plt.xticks([0, 1], ["Positive Skew", "Negative Skew"])
        plt.title("Figure 2. Distribution of Reval Across Skew Regimes")
        plt.ylabel("Revaluation (bps)")
        plt.xlabel("Skew Regime")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    # Summary stats
    result = {
        "U_statistic": U,
        "p_value": p_value,
        "median_positive": pos_reval.median(),
        "median_negative": neg_reval.median(),
        "n_positive": len(pos_reval),
        "n_negative": len(neg_reval)
    }

    print(f"\nMann–Whitney U Test Results:")
    print(f"U statistic = {U:.2f}")
    print(f"p-value     = {p_value:.4f}")
    print(f"Median (positive skew) = {pos_reval.median():.6f} bps")
    print(f"Median (negative skew) = {neg_reval.median():.6f} bps")

    if p_value < 0.05:
        print("→ Result: Statistically significant difference between regimes (possible information leakage).")
    else:
        print("→ Result: No significant difference between regimes (no evidence of leakage).")

    return result


if __name__ == "__main__":
    import pandas as pd

    # önce regression experiment 1'de ürettiğimiz veriyi oku
    ev = pd.read_csv("data/EURUSD_X_events.csv")  # senin kullandığın isim neyse o

    # test fonksiyonunu çağır
    mann_whitney_reval_test(ev)