# Limit-Order-Matching-Microstructure
Paper: https://arxiv.org/abs/2511.20606  
Code: https://github.com/Republic1024/Limit-Order-Matching-Microstructure
### Unifying Matching Markets and Limit Order Books through Microstructure Dynamics  
### Code Release for: *Limit Order Book Dynamics in Matching Markets: Microstructure, Spread, and Execution Slippage*

![simulation_results.png](simulation_results.png)
---

## 📌 Overview

This repository contains the full simulation code, experiments, and visualization pipeline for the paper:

**“Limit Order Book Dynamics in Matching Markets: Microstructure, Spread, and Execution Slippage”**  
arXiv: https://arxiv.org/abs/2511.20606

The project proposes a unified framework where **matching markets** (e.g., marriage, partner choice, labor matching) are modeled as **limit order books**, with:

- **Intrinsic value** → `ask`  
- **Reachability constraint** → `bid-depth / liquidity drought`  
- **ΔV gap** → structural **spread**  
- **Compensation C** → imperfect price improvement  
- **Slippage (regret)** → execution shortfall  
- **Settling** → threshold-decay crossing event

The framework shows that **linear compensation cannot close structural preference gaps**, unless it triggers a **categorical identity shift** (`Identity Collapse Threshold`).

---

## 🔍 Core Concepts

### **1. Unconditional vs. Reachable Maximum**
- `V_uncond_max`: Best perceived partner that exists in the population.  
- `V_reach_max`: Best partner currently reachable under social liquidity constraints.  
- `ΔV = V_uncond_max - V_reach_max`:  
  → The **structural preference gap**, analogous to a *bid-ask spread*.

### **2. Theorem 1 — Compensation Clipping & Identity Collapse**
If compensation utility is:

```

h(C) = min(εC, C_max)

```

Then:

- If `εC < C_max` → **Compensation is ineffective**: ΔV persists  
- If `εC ≥ C_max` → **Identity Collapse**: category shift occurs

This mirrors slippage-bounded execution in microstructure.

### **3. Threshold Dynamics (Settling)**
Commitment occurs when:

```

θ = U_effective / V_uncond_max ≥ T(t)

```

Where `T(t)` is a decaying liquidity threshold (similar to urgency-driven execution).

---

## 📁 Repository Structure

```

Limit-Order-Matching-Microstructure/
│
├── exp1-5.py               # Main experiments (Sections 4.2–4.6)
├── exp1-5-Chinese.py       # Chinese commented version
├── simulation_results.png  # Fig 5 replication
├── simulation_results2.png # Slippage + Clipping + Settling plots
├── data/                   # (Empty / Ignored) placeholder for datasets
├── img1.jpg                # Paper figure assets
├── img2.jpg
├── img3.jpg
├── .gitignore
└── README.md

```

---

## 📊 Experiments Included (Sections 4.2–4.6)

### **Experiment 1 — Compensation Failure**
Shows why compensation cannot close ΔV under clipping.

### **Experiment 2 — Settling Dynamics**
Implements the threshold-decay commitment model.

### **Experiment 3 — Instant Commitment**
High-tier reachable candidate → immediate match.

### **Experiment 4 — Regional Differences**
Despite different compensation norms (Jiangsu vs Guangdong),  
**ranking is invariant** → structural gaps dominate.

### **Experiment 5 — Regret Prediction**
Shock to `V_uncond_max` yields post-match θ decline → slippage regret.

---

## 🎨 Visualization

`generate_academic_plots()` reproduces Figures:

- Settling curve `T(t)` vs θ  
- Compensation utility clipping (Theorem 1)  
- Structural slippage bars  

Outputs:

```

simulation_results2.png

```

---

## ▶️ How to Run

### **1. Install dependencies**
```

pip install numpy pandas matplotlib

```

### **2. Run the experiments**
```

python exp1-5.py

```

### **3. Generate visualizations**
(automatically triggered at the end)

---

## 📚 Citation

If you use this framework, please cite:

```

Wu, Y. (2025). Limit Order Book Dynamics in Matching Markets:
Microstructure, Spread, and Execution Slippage.
arXiv:2511.20606.

```

---

## 🧠 Philosophy Behind the Model (Short)

This project formalizes a fundamental principle:

> **Compensation cannot close structural gaps.  
> Only identity shifts can.**

This emerges naturally from the microstructure mapping between  
ΔV → spread,  
C → bounded price improvement,  
and slippage → structural regret.

