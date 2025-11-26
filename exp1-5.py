import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass

# Set random seed for reproducibility
np.random.seed(42)

# Set plotting style (attempt academic style)
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('ggplot')  # fallback style


@dataclass
class MarketConfig:
    """
    Market configuration parameters.
    """
    n_candidates: int = 10000

    # Alpha=2, Beta=8 create a right-skewed distribution (pyramid-like),
    # simulating the Gini Cone structure (Appendix F).
    beta_alpha: float = 2.0
    beta_beta: float = 8.0

    # Compensation parameters
    c_elasticity: float = 0.05        # marginal utility of compensation ε
    c_max_utility_cap: float = 20.0   # Identity Collapse Threshold C*


class GiniMarriageMarket:
    def __init__(self, config: MarketConfig):
        self.cfg = config
        self.df = self._generate_market_depth()

    def _generate_market_depth(self):
        """
        Generate a candidate pool shaped by the 'Gini Cone'
        using a Beta distribution to simulate class scarcity.
        """
        n = self.cfg.n_candidates
        rng = np.random.default_rng(42)

        # 1. Intrinsic value V sampled from a skewed distribution (0-100)
        raw_dist = rng.beta(self.cfg.beta_alpha, self.cfg.beta_beta, size=n)
        v_values = raw_dist * 100

        # 2. Reachability decreases linearly with status
        reach_prob = 1.0 - 0.8 * (v_values / 100)
        is_reachable = rng.random(n) < reach_prob

        # 3. Compensation capacity correlated with V but noisy
        c_values = v_values * rng.uniform(0.5, 1.5, size=n) * 10

        return pd.DataFrame({
            'V_intrinsic': v_values,
            'C_offer': c_values,
            'is_reachable': is_reachable
        })

    def run_female_decision_logic(self, female_uncond_max: float, threshold_t: float):
        """
        State-machine decision logic:
        compute Theta, Delta-V, and Slippage (structural regret).
        """
        df = self.df.copy()

        # --- Core Logic: Compensation Clipping (Theorem 1) ---
        raw_utility_gain = df['C_offer'] * self.cfg.c_elasticity

        # Identity Collapse Threshold:
        # utility gain from compensation is capped unless C is extremely large
        capped_utility_gain = np.minimum(raw_utility_gain, self.cfg.c_max_utility_cap)

        # Effective utility: U = V + h(C)
        df['U_effective'] = df['V_intrinsic'] + capped_utility_gain

        # --- Matching under Market Microstructure ---
        reachable_df = df[df['is_reachable']].copy()

        if reachable_df.empty:
            return {'Match': False, 'Reason': 'No reachable candidates'}

        best_idx = reachable_df['U_effective'].idxmax()
        v_reach_max = reachable_df.loc[best_idx, 'V_intrinsic']
        u_reach_max = reachable_df.loc[best_idx, 'U_effective']
        c_offer_best = reachable_df.loc[best_idx, 'C_offer']

        # Core metrics
        delta_v = female_uncond_max - v_reach_max
        theta = u_reach_max / female_uncond_max

        # Structural regret (Slippage)
        slippage = female_uncond_max - u_reach_max

        return {
            "Theta": theta,
            "Delta_V": delta_v,
            "Slippage": slippage,
            "Match": theta >= threshold_t,
            "Details": {
                "V_best": v_reach_max,
                "C_best": c_offer_best,
                "U_best": u_reach_max
            }
        }


# ==========================================
# Visualization Module
# ==========================================
def generate_academic_plots():
    """Produce the three-panel academic-style figure for the paper."""
    print(">>> Generating visualization figure (simulation_results.png)...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cfg = MarketConfig()

    # --- Plot A: Settling Dynamics (Theta vs Age) ---
    ax1 = axes[0]
    ages = np.arange(22, 38)

    # Threshold decay function T(t)
    thresholds = np.clip(1.0 - (ages - 22) * 0.035, 0.65, 1.0)
    theta_reality = 0.78

    ax1.plot(ages, thresholds, 'r--', linewidth=2.5, label=r'Threshold $T(t)$')
    ax1.plot(ages, [theta_reality] * len(ages), 'b-', linewidth=2.5, label=r'Reality $\theta$')

    cross_idx = np.where(theta_reality >= thresholds)[0]
    if len(cross_idx) > 0:
        idx = cross_idx[0]
        ax1.scatter([ages[idx]], [thresholds[idx]], color='black', s=100, zorder=5)
        ax1.annotate('Commit Event\n(Settling)',
                     xy=(ages[idx], thresholds[idx]),
                     xytext=(ages[idx] - 4, thresholds[idx] - 0.15),
                     arrowprops=dict(facecolor='black', shrink=0.05))
        ax1.fill_between(ages, theta_reality, thresholds,
                         where=(theta_reality >= thresholds),
                         color='green', alpha=0.2, label='Match Zone')

    ax1.set_title('A. The Dynamics of "Settling"', fontsize=14)
    ax1.set_xlabel('Age (Years)', fontsize=12)
    ax1.set_ylabel('Ratio / Threshold', fontsize=12)
    ax1.legend()
    ax1.set_ylim(0.5, 1.1)

    # --- Plot B: Compensation Utility Clipping (Theorem 1) ---
    ax2 = axes[1]
    C_values = np.linspace(0, 1000, 100)
    utility_gain = np.minimum(C_values * cfg.c_elasticity, cfg.c_max_utility_cap)

    ax2.plot(C_values, utility_gain, 'purple', linewidth=3)
    ax2.axhline(y=cfg.c_max_utility_cap, color='gray', linestyle='--', alpha=0.7)
    ax2.text(100, cfg.c_max_utility_cap + 0.5, r'Identity Collapse ($C_{max}$)', color='gray')

    clip_start = cfg.c_max_utility_cap / cfg.c_elasticity
    ax2.axvline(x=clip_start, color='orange', linestyle=':')
    ax2.fill_between(C_values, utility_gain,
                     [cfg.c_max_utility_cap] * len(C_values),
                     where=(C_values > clip_start),
                     color='gray', alpha=0.1)

    ax2.text(600, 10, 'Ineffective Region\n(Zero Marginal Utility)', ha='center', fontsize=10)
    ax2.set_title('B. Compensation Utility Clipping', fontsize=14)
    ax2.set_xlabel('Compensation Amount $C$ (k)', fontsize=12)
    ax2.set_ylabel(r'Effective Utility Gain $h(C)$', fontsize=12)

    # --- Plot C: Structural Slippage ---
    ax3 = axes[2]
    scenarios = ['Low Gap\n($\Delta V \\approx 5$)',
                 'Mid Gap\n($\Delta V \\approx 15$)',
                 'High Gap\n($\Delta V \\approx 30$)']
    slippage_values = [5.0, 15.0, 30.0]
    colors = ['#2ecc71', '#f1c40f', '#e74c3c']

    bars = ax3.bar(scenarios, slippage_values, color=colors, alpha=0.8)

    ax3.axhline(y=20, color='red', linestyle='--', label='Divorce Risk Threshold')

    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.1f}',
                 ha='center', va='bottom')

    ax3.set_title('C. Structural Regret (Slippage)', fontsize=14)
    ax3.set_ylabel(r'Regret Magnitude ($\Delta V$)', fontsize=12)
    ax3.legend()

    plt.tight_layout()
    plt.savefig('simulation_results.png', dpi=800)
    print(">>> Figure saved as: simulation_results.png")
    plt.show()


# ==========================================
# Key Experiments (Sections 4.2–4.6)
# ==========================================
def run_key_experiments():
    print(">>> Initializing Gini-Cone Marriage Market simulation...")
    market = GiniMarriageMarket(MarketConfig())
    female_ideal = 95.0

    # --- Experiment 1: Compensation Failure ---
    print("\n--- Experiment 1: Compensation Failure ---")
    low_tier_male = pd.DataFrame({'V_intrinsic': [60.0],
                                  'C_offer': [500.0],
                                  'is_reachable': [True]})
    market.df = low_tier_male
    res1 = market.run_female_decision_logic(female_ideal, threshold_t=0.8)
    print(f"V=60, C=500k -> Match? {res1['Match']} (Reason: Cap triggered)")

    # --- Experiment 2: Settling Dynamics ---
    print("\n--- Experiment 2: Settling Dynamics ---")
    market = GiniMarriageMarket(MarketConfig())
    ages, thresholds = [24, 28, 32], [1.0, 0.85, 0.70]
    for age, t in zip(ages, thresholds):
        res = market.run_female_decision_logic(female_ideal, t)
        status = "COMMIT" if res['Match'] else "WAIT"
        print(f"Age {age} (T={t:.2f}) -> Status: {status}")

    # --- Experiment 3: Instant Commitment Logic ---
    print("\n--- Experiment 3: Instant Commitment Logic ---")
    high_tier = pd.DataFrame({'V_intrinsic': [94.0],
                              'C_offer': [0.0],
                              'is_reachable': [True]})
    market.df = high_tier
    res3 = market.run_female_decision_logic(female_ideal, 0.95)
    print(f"V=94, C=0 -> Match? {res3['Match']} (Theta={res3['Theta']:.2f})")

    # --- Experiment 4: Regional Differences (Invariant Rankings) ---
    print("\n--- Experiment 4: Regional Differences ---")
    base = pd.DataFrame({'id': ['A', 'B'],
                         'V_intrinsic': [85.0, 75.0],
                         'is_reachable': [True, True]})

    df_js = base.copy()
    df_js['C_offer'] = [210, 250]  # Jiangsu compensation pattern
    market.df = df_js
    best_js = market.run_female_decision_logic(female_ideal, 0.8)['Details']['V_best']

    df_gd = base.copy()
    df_gd['C_offer'] = [40, 80]   # Guangdong compensation pattern
    market.df = df_gd
    best_gd = market.run_female_decision_logic(female_ideal, 0.8)['Details']['V_best']

    print(f"Jiangsu Best V: {best_js}, Guangdong Best V: {best_gd}")
    print("Result: Ranking Invariant." if best_js == best_gd else "Result: Changed.")

    # --- Experiment 5: Regret Prediction ---
    print("\n--- Experiment 5: Regret Prediction ---")
    v_husb, v_ideal = 75.0, 90.0
    print(f"Pre-shock Theta: {v_husb / v_ideal:.2f}")
    v_ideal_shock = v_ideal * 1.10  # shock
    print(f"Post-shock Theta: {v_husb / v_ideal_shock:.2f} (If < 0.8 then regret likely)")


if __name__ == "__main__":
    # 1. Run key simulations
    run_key_experiments()

    # 2. Produce academic visualizations
    generate_academic_plots()
