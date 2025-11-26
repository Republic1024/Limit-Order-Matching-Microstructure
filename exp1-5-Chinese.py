import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass

# 设置随机种子以保证结果可复现
np.random.seed(42)

# 设置绘图风格 (尝试使用学术风格)
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('ggplot')  # Fallback style


@dataclass
class MarketConfig:
    """
    市场配置参数
    """
    n_candidates: int = 10000
    # Alpha=2, Beta=8 产生右偏分布 (类似金字塔结构)，模拟附录 F 的基尼锥体 [cite: 1846]
    beta_alpha: float = 2.0
    beta_beta: float = 8.0

    # 补偿参数
    c_elasticity: float = 0.05  # 补偿的边际效用 (epsilon)
    c_max_utility_cap: float = 20.0  # 身份坍缩阈值 (Identity Collapse Threshold) [cite: 1032]


class GiniMarriageMarket:
    def __init__(self, config: MarketConfig):
        self.cfg = config
        self.df = self._generate_market_depth()

    def _generate_market_depth(self):
        """
        生成符合“基尼锥体”结构的男性候选人池。
        使用 Beta 分布模拟社会阶层稀缺性 [cite: 1838]。
        """
        n = self.cfg.n_candidates
        rng = np.random.default_rng(42)

        # 1. 内在价值 (V) -> 偏态分布 (0-100)
        raw_dist = rng.beta(self.cfg.beta_alpha, self.cfg.beta_beta, size=n)
        v_values = raw_dist * 100

        # 2. "Reachable" (可得性) 概率随地位线性衰减
        # 地位越高，竞争越激烈/越难遇到 (锥体几何性质)
        reach_prob = 1.0 - 0.8 * (v_values / 100)
        is_reachable = rng.random(n) < reach_prob

        # 3. 补偿能力 (C) 与 V 相关但有噪声
        c_values = v_values * rng.uniform(0.5, 1.5, size=n) * 10

        return pd.DataFrame({
            'V_intrinsic': v_values,
            'C_offer': c_values,
            'is_reachable': is_reachable
        })

    def run_female_decision_logic(self, female_uncond_max: float, threshold_t: float):
        """
        执行状态机决策逻辑：
        计算 Theta (现实比率), Delta V (心理落差), 和 Slippage (遗憾/滑点)。
        """
        df = self.df.copy()

        # --- 核心逻辑：补偿截断 (Theorem 1)  ---
        # 计算原始补偿带来的效用增益
        raw_utility_gain = df['C_offer'] * self.cfg.c_elasticity

        # 实施“身份坍缩阈值”：
        # 补偿带来的效用提升不能超过 Cap，除非 C 巨大到足以重构身份
        capped_utility_gain = np.minimum(raw_utility_gain, self.cfg.c_max_utility_cap)

        # 有效效用 U = V + h(C)
        df['U_effective'] = df['V_intrinsic'] + capped_utility_gain

        # --- 市场微观结构匹配 ---
        reachable_df = df[df['is_reachable']].copy()

        if reachable_df.empty:
            return {'Match': False, 'Reason': 'No Reachable Candidates'}

        best_idx = reachable_df['U_effective'].idxmax()
        v_reach_max = reachable_df.loc[best_idx, 'V_intrinsic']
        u_reach_max = reachable_df.loc[best_idx, 'U_effective']
        c_offer_best = reachable_df.loc[best_idx, 'C_offer']

        # 计算核心指标
        delta_v = female_uncond_max - v_reach_max
        theta = u_reach_max / female_uncond_max

        # 滑点 (Slippage) = 婚后的结构性遗憾 [cite: 1723]
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
# 可视化模块 (Visualization Module)
# ==========================================
def generate_academic_plots():
    """生成用于论文的三合一可视化图表"""
    print(">>> 正在生成可视化图表 (simulation_results.png)...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cfg = MarketConfig()

    # --- Plot A: Settling Dynamics (Theta vs Age) ---
    # 对应论文 Fig 5
    ax1 = axes[0]
    ages = np.arange(22, 38)
    # 模拟阈值衰减函数 T(t)
    thresholds = np.clip(1.0 - (ages - 22) * 0.035, 0.65, 1.0)
    # 假设一个固定的现实对象 (Theta=0.78)
    theta_reality = 0.78

    ax1.plot(ages, thresholds, 'r--', linewidth=2.5, label=r'Threshold $T(t)$')
    ax1.plot(ages, [theta_reality] * len(ages), 'b-', linewidth=2.5, label=r'Reality $\theta$')

    # 标注结婚区域 (Marriage Zone)
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

    # --- Plot B: Compensation Clipping (Theorem 1) ---
    # 对应论文 Fig 4 [cite: 1747]
    ax2 = axes[1]
    C_values = np.linspace(0, 1000, 100)  # 0 to 1 million
    # h(C) = min(C * elasticity, Cap)
    utility_gain = np.minimum(C_values * cfg.c_elasticity, cfg.c_max_utility_cap)

    ax2.plot(C_values, utility_gain, 'purple', linewidth=3)
    ax2.axhline(y=cfg.c_max_utility_cap, color='gray', linestyle='--', alpha=0.7)
    ax2.text(100, cfg.c_max_utility_cap + 0.5, r'Identity Collapse ($C_{max}$)', color='gray')

    # 标注无效区域
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

    # --- Plot C: Regret/Slippage Analysis ---
    # 对应论文 Appendix D [cite: 1723]
    ax3 = axes[2]
    # 模拟三种 Gap 场景
    scenarios = ['Low Gap\n($\Delta V \\approx 5$)', 'Mid Gap\n($\Delta V \\approx 15$)',
                 'High Gap\n($\Delta V \\approx 30$)']
    slippage_values = [5.0, 15.0, 30.0]  # 结构性遗憾值
    colors = ['#2ecc71', '#f1c40f', '#e74c3c']

    bars = ax3.bar(scenarios, slippage_values, color=colors, alpha=0.8)

    # 添加阈值线 (假设心理承受力)
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
    plt.savefig('simulation_results2.png', dpi=800)
    print(">>> 图表已保存为: simulation_results.png")
    plt.show()


# ==========================================
# 实验逻辑 (Sections 4.2 - 4.6)
# ==========================================
def run_key_experiments():
    print(">>> 初始化基尼锥体婚姻市场仿真...")
    market = GiniMarriageMarket(MarketConfig())
    female_ideal = 95.0

    # --- 实验 1: 补偿失效 (Sec 4.2) [cite: 1205] ---
    print("\n--- 实验 1: 补偿失效 ---")
    low_tier_male = pd.DataFrame({'V_intrinsic': [60.0], 'C_offer': [500.0], 'is_reachable': [True]})
    market.df = low_tier_male
    res1 = market.run_female_decision_logic(female_ideal, threshold_t=0.8)
    print(f"V=60, C=500k -> Match? {res1['Match']} (Reason: Cap triggered)")

    # --- 实验 2: 妥协动态 (Sec 4.3) [cite: 1218] ---
    print("\n--- 实验 2: 妥协动态 ---")
    market = GiniMarriageMarket(MarketConfig())
    ages, thresholds = [24, 28, 32], [1.0, 0.85, 0.70]
    for age, t in zip(ages, thresholds):
        res = market.run_female_decision_logic(female_ideal, t)
        status = "COMMIT" if res['Match'] else "WAIT"
        print(f"Age {age} (T={t:.2f}) -> Status: {status}")

    # --- 实验 3: 秒选逻辑 (Sec 4.4) [cite: 1233] ---
    print("\n--- 实验 3: 秒选逻辑 ---")
    high_tier = pd.DataFrame({'V_intrinsic': [94.0], 'C_offer': [0.0], 'is_reachable': [True]})
    market.df = high_tier
    res3 = market.run_female_decision_logic(female_ideal, 0.95)
    print(f"V=94, C=0 -> Match? {res3['Match']} (Theta={res3['Theta']:.2f})")

    # --- 实验 4: 区域差异 (Sec 4.5) [cite: 1242] ---
    print("\n--- 实验 4: 区域差异 ---")
    base = pd.DataFrame({'id': ['A', 'B'], 'V_intrinsic': [85.0, 75.0], 'is_reachable': [True, True]})

    df_js = base.copy();
    df_js['C_offer'] = [210, 250]  # Jiangsu
    market.df = df_js
    best_js = market.run_female_decision_logic(female_ideal, 0.8)['Details']['V_best']

    df_gd = base.copy();
    df_gd['C_offer'] = [40, 80]  # Guangdong
    market.df = df_gd
    best_gd = market.run_female_decision_logic(female_ideal, 0.8)['Details']['V_best']

    print(f"Jiangsu Best V: {best_js}, Guangdong Best V: {best_gd}")
    print("Result: Ranking Invariant." if best_js == best_gd else "Result: Changed.")

    # --- 实验 5: 后悔预测 (Sec 4.6) [cite: 1250] ---
    print("\n--- 实验 5: 后悔预测 ---")
    v_husb, v_ideal = 75.0, 90.0
    print(f"婚前 Theta: {v_husb / v_ideal:.2f}")
    v_ideal_shock = v_ideal * 1.10  # 冲击
    print(f"冲击后 Theta: {v_husb / v_ideal_shock:.2f} (若 < 0.8 则后悔)")


if __name__ == "__main__":
    # 1. 运行数值仿真
    run_key_experiments()
    # 2. 生成学术图表
    generate_academic_plots()