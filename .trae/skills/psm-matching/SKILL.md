---
name: psm-matching
description: 倾向得分匹配（PSM）的实施、平衡性检验和报告规范。用于DID前确保处理组与对照组可比性。
---

# 倾向得分匹配（PSM）实施技能

## 何时使用

- DID 回归前需要确保处理组与对照组可比性
- 处理分配非随机，需要通过匹配控制混淆因素
- 需要提供 Rosenbaum 界限检验评估未观测混淆敏感性

## 核心原理

PSM 通过倾向得分（Propensity Score）在处理组和对照组之间进行匹配，降低选择性偏差：

```
P(X) = Pr(Treat=1 | X) = Pr(Treat=1 | X_1, X_2, ..., X_k)
```

## 实施步骤

### Step 1: 选择协变量

```python
from typing import Optional
import pandas as pd
import numpy as np

def select_covariates(
    df: pd.DataFrame,
    outcome_var: str,
    treat_var: str,
    candidate_vars: list[str],
    max_vars: int = 10
) -> list[str]:
    """
    基于理论相关性和统计显著性选择协变量

    原则：
    1. 与处理分配相关（解释倾向得分）
    2. 与被解释变量相关（控制混淆）
    3. 不受处理影响（不是中介变量）
    """
    selected = []

    for var in candidate_vars:
        treat_corr = df[[treat_var, var]].corr().iloc[0, 1]
        outcome_corr = df[[outcome_var, var]].corr().iloc[0, 1]

        if abs(treat_corr) > 0.05 and abs(outcome_corr) > 0.05:
            selected.append(var)

    return selected[:max_vars]
```

### Step 2: 估计倾向得分

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def estimate_propensity_score(
    df: pd.DataFrame,
    treat_var: str,
    covariate_vars: list[str],
    scale: bool = True
) -> pd.Series:
    """
    使用 Logit 模型估计倾向得分

    Parameters
    ----------
    df : pd.DataFrame
        样本数据
    treat_var : str
        处理变量名
    covariate_vars : list[str]
        协变量列表

    Returns
    -------
    pd.Series
        倾向得分（概率值 0-1）
    """
    X = df[covariate_vars].copy()

    if scale:
        scaler = StandardScaler()
        X = pd.DataFrame(
            scaler.fit_transform(X),
            columns=covariate_vars,
            index=df.index
        )

    model = LogisticRegression(max_iter=1000)
    model.fit(X, df[treat_var])

    propensity_scores = model.predict_proba(X)[:, 1]

    return pd.Series(propensity_scores, index=df.index, name='propensity_score')
```

### Step 3: 执行匹配

```python
def kernel_matching(
    treatment_df: pd.DataFrame,
    control_df: pd.DataFrame,
    propensity_col: str = 'propensity_score',
    bandwidth: float = 0.06
) -> pd.DataFrame:
    """
    核匹配（Kernel Matching）

    处理组每个样本使用加权对照组构建反事实
    """
    matched_data = []

    for idx, row in treatment_df.iterrows():
        ps = row[propensity_col]

        weights = control_df[propensity_col].apply(
            lambda ps_c: kernel_epanechnikov(ps, ps_c, bandwidth)
        )

        normalized_weights = weights / weights.sum()

        matched_row = {
            'matched_idx': idx,
            'weight': 1.0,
            'counterfactual': (control_df.drop('propensity_score', axis=1).multiply(
                normalized_weights, axis=0
            ).sum())
        }
        matched_data.append(matched_row)

    return pd.DataFrame(matched_data)


def nearest_neighbor_matching(
    treatment_df: pd.DataFrame,
    control_df: pd.DataFrame,
    propensity_col: str = 'propensity_score',
    n_neighbors: int = 1,
    with_replacement: bool = False
) -> dict:
    """
    最近邻匹配（Nearest Neighbor Matching）

    1:1 或 1:k 匹配
    """
    matched_indices = []

    for idx, row in treatment_df.iterrows():
        ps = row[propensity_col]

        distances = abs(control_df[propensity_col] - ps)
        nearest_idx = distances.nsmallest(n_neighbors).index

        for ni in nearest_idx:
            matched_indices.append({
                'treated_idx': idx,
                'control_idx': ni,
                'distance': distances[ni]
            })

    return pd.DataFrame(matched_indices)


def kernel_epanechnikov(
    t: float,
    c: float,
    h: float
) -> float:
    """Epanechnikov 核函数"""
    u = (t - c) / h
    if abs(u) <= 1:
        return 0.75 * (1 - u ** 2)
    return 0.0


def caliper_matching(
    treatment_df: pd.DataFrame,
    control_df: pd.DataFrame,
    propensity_col: str = 'propensity_score',
    caliper: float = 0.01
) -> dict:
    """
    卡尺匹配（Caliper Matching）

    只匹配倾向得分差异在卡尺范围内的样本对
    """
    matched = []

    for idx, row in treatment_df.iterrows():
        ps = row[propensity_col]

        eligible = control_df[
            abs(control_df[propensity_col] - ps) <= caliper
        ]

        if len(eligible) > 0:
            nearest = eligible.loc[
                abs(eligible[propensity_col] - ps).idxmin()
            ]
            matched.append({
                'treated_idx': idx,
                'control_idx': nearest.name,
                'ps_treated': ps,
                'ps_control': nearest[propensity_col],
                'distance': abs(ps - nearest[propensity_col])
            })

    return pd.DataFrame(matched)
```

### Step 4: 平衡性检验

```python
def balance_check(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    covariate_vars: list[str],
    treat_var: str
) -> pd.DataFrame:
    """
    平衡性检验：比较匹配前后各协变量的标准化偏差

    通过标准：标准化偏差 < 10%（理想 < 5%）
    """
    results = []

    for var in covariate_vars:
        mean_treated_before = df_before.loc[
            df_before[treat_var] == 1, var
        ].mean()
        mean_control_before = df_before.loc[
            df_before[treat_var] == 0, var
        ].mean()
        std_pooled_before = df_before[var].std()

        std_bias_before = (
            (mean_treated_before - mean_control_before) / std_pooled_before * 100
        )

        mean_treated_after = df_after.loc[
            df_after[treat_var] == 1, var
        ].mean()
        mean_control_after = df_after.loc[
            df_after[treat_var] == 0, var
        ].mean()
        std_pooled_after = df_after[var].std()

        std_bias_after = (
            (mean_treated_after - mean_control_after) / std_pooled_after * 100
        )

        results.append({
            'variable': var,
            'mean_treated_raw': mean_treated_before,
            'mean_control_raw': mean_control_before,
            'std_bias_raw': abs(std_bias_before),
            'mean_treated_matched': mean_treated_after,
            'mean_control_matched': mean_control_after,
            'std_bias_matched': abs(std_bias_after),
            'bias_reduction': (
                abs(std_bias_before) - abs(std_bias_after)
            ) / abs(std_bias_before) * 100
        })

    return pd.DataFrame(results)


def joint_f_test(
    df: pd.DataFrame,
    outcome: str,
    treat_var: str,
    covariate_vars: list[str]
) -> dict:
    """
    联合 F 检验：检验匹配后处理组与对照组是否仍有显著差异
    """
    import statsmodels.formula.api as smf

    formula = f"{treat_var} ~ " + " + ".join(covariate_vars)
    model = smf.ols(formula, data=df).fit()

    f_stat = model.fvalue
    p_value = model.f_pvalue

    return {
        'f_statistic': f_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }
```

### Step 5: Rosenbaum 界限检验

```python
def rosenbaum_bounds(
    matched_df: pd.DataFrame,
    outcome_var: str,
    treat_var: str,
    gamma_range: tuple[float, float] = (1.0, 3.0),
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Rosenbaum 界限检验：评估结论对未观测混淆的敏感性

    Γ = 1: 无混淆（理想情况）
    Γ > 1: 允许一定程度的未观测混淆
    结论稳健要求：在一定 Γ 范围内 p值仍不显著

    参考论文标准：Γ=3.0 时 p值上界 > 0.05，结论稳健
    """
    results = []

    treated_outcomes = matched_df.loc[
        matched_df[treat_var] == 1, outcome_var
    ].values

    control_outcomes = matched_df.loc[
        matched_df[treat_var] == 0, outcome_var
    ].values

    for gamma in np.arange(gamma_range[0], gamma_range[1] + 0.25, 0.25):
        if gamma == 1.0:
            p_upper = compute_holm_gamma1(treated_outcomes, control_outcomes)
        else:
            p_upper = compute_rosenbaum_bound(
                treated_outcomes, control_outcomes, gamma
            )

        results.append({
            'gamma': gamma,
            'p_value_upper': p_upper,
            'significant_at_05': p_upper < alpha,
            'robust': p_upper > alpha
        })

    return pd.DataFrame(results)


def compute_rosenbaum_bound(
    treated: np.ndarray,
    control: np.ndarray,
    gamma: float
) -> float:
    """
    计算给定 Γ 下的 p值上界
    """
    import scipy.stats as stats

    n_t = len(treated)
    n_c = len(control)

    observed_diff = treated.mean() - control.mean()

    count_extreme = 0
    n_simulations = 10000

    for _ in range(n_simulations):
        indices_c = np.random.choice(
            n_c, size=n_t, replace=True
        )
        synthetic_control = control[indices_c]

        diff = treated.mean() - synthetic_control.mean()

        if abs(diff) >= abs(observed_diff):
            count_extreme += 1

    p_upper = count_extreme / n_simulations

    return min(p_upper * gamma, 1.0)
```

## PSM+DID 完整流程

```python
def run_psm_did(
    df: pd.DataFrame,
    outcome: str,
    treat_var: str,
    post_var: str,
    covariate_vars: list[str],
    matching_method: str = 'kernel',
    bandwidth: float = 0.06
) -> dict:
    """
    完整 PSM+DID 分析流程

    1. 估计倾向得分
    2. 执行匹配
    3. 平衡性检验
    4. 匹配后 DID 回归
    5. Rosenbaum 界限检验
    """
    df = df.copy()

    df['propensity_score'] = estimate_propensity_score(
        df, treat_var, covariate_vars
    )

    treated_df = df[df[treat_var] == 1].copy()
    control_df = df[df[treat_var] == 0].copy()

    if matching_method == 'kernel':
        matched = kernel_matching(
            treated_df, control_df,
            bandwidth=bandwidth
        )
    elif matching_method == 'nn':
        matched = nearest_neighbor_matching(
            treated_df, control_df
        )
    elif matching_method == 'caliper':
        matched = caliper_matching(
            treated_df, control_df
        )

    df_matched = pd.concat([
        treated_df,
        control_df.loc[matched['control_idx']]
    ])

    balance_results = balance_check(
        df, df_matched, covariate_vars, treat_var
    )

    formula = f"{outcome} ~ {treat_var}:{post_var} + C(分行代码) + C(交易日期)"
    model = smf.ols(formula, data=df_matched).fit(
        cov_type='cluster',
        cov_kwds={'groups': df_matched['分行代码']}
    )

    att = model.params[f'{treat_var}:{post_var}']

    rosenbaum = rosenbaum_bounds(
        df_matched, outcome, treat_var
    )

    return {
        'att': att,
        'att_se': model.bse[f'{treat_var}:{post_var}'],
        'att_pvalue': model.pvalues[f'{treat_var}:{post_var}'],
        'balance_check': balance_results,
        'rosenbaum_bounds': rosenbaum,
        'matched_n': len(df_matched),
        'model_summary': model.summary().as_text()
    }
```

## 报告规范

### 必须报告的内容

| 内容 | 要求 |
|------|------|
| 匹配方法 | 核匹配/最近邻/卡尺匹配 |
| 协变量选择 | 列出使用的协变量及其理由 |
| 倾向得分模型 | Logit/Probit，报告伪R² |
| 平衡性检验 | 匹配前后标准化偏差表格 |
| 共同支撑检验 | 倾向得分分布重叠图 |
| Rosenbaum界限 | 不同Γ值下的p值上界 |

### 平衡性检验通过标准

```
标准化偏差：
- < 10%: 可接受
- < 5%: 理想
- > 20%: 不可接受，需重新匹配

t检验：
- p > 0.05: 通过
- p > 0.1: 理想
```

## 参考论文

本文档规范参考《生成式AI在招商银行零售客户经营中的应用及成效分析》第四章 PSM 设计：匹配后标准化偏差 < 3.1%，Rosenbaum 界限 Γ=3.0 时 p值上界 0.634 > 0.05。