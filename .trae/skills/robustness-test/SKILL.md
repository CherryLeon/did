---
name: robustness-test
description: DID模型稳健性检验的实施规范，包括安慰剂检验、排除政策干扰、样本调整等。
---

# 稳健性检验技能

## 何时使用

- DID 基准回归完成后
- 报告最终结论前
- 评估结论对假设违背的敏感性

## 稳健性检验体系

| 检验类型 | 目的 | 通过标准 |
|----------|------|----------|
| 安慰剂检验 | 排除随机性干扰 | 真实系数位于伪系数分布 2.5% 尾部之外 |
| 排除政策干扰 | 控制其他同时发生的政策 | 核心系数保持稳健 |
| 样本调整 | 排除极端样本影响 | 核心系数保持稳健 |
| 变量替换 | 检验结果对指标定义的敏感性 | 核心系数方向一致 |
| 时间窗口调整 | 检验对观测期选择的敏感性 | 核心系数保持稳健 |

## 1. 安慰剂检验

```python
import numpy as np
import pandas as pd
from typing import Callable

def placebo_test(
    df: pd.DataFrame,
    outcome: str,
    treat_var: str,
    unit_var: str,
    time_var: str,
    n_simulations: int = 1000,
    random_seed: int = 42
) -> dict:
    """
    安慰剂检验：随机虚构处理组，检验真实效应是否仅为偶然

    原理：如果真实系数仅由随机因素驱动，
    则随机打乱处理分配后得到的系数分布应包含真实系数

    通过标准：真实系数位于伪系数分布的 2.5% 尾部之外（伪p值 < 0.05）
    """
    np.random.seed(random_seed)

    units = df[unit_var].unique()
    true_coef = None
    placebo_coefs = []

    formula = f"{outcome} ~ {treat_var}:Post + C({unit_var}) + C({time_var})"

    for sim in range(n_simulations):
        df_sim = df.copy()

        if sim == 0:
            pass
        else:
            treated_sim = np.random.choice(
                units,
                size=len(units) // 2,
                replace=False
            )
            df_sim[treat_var] = df_sim[unit_var].isin(treated_sim).astype(int)

        model = smf.ols(formula, data=df_sim).fit(
            cov_type='cluster',
            cov_kwds={'groups': df_sim[unit_var]}
        )

        if sim == 0:
            true_coef = model.params.get(f'{treat_var}:Post', np.nan)
        else:
            placebo_coefs.append(
                model.params.get(f'{treat_var}:Post', np.nan)
            )

    placebo_coefs = np.array(placebo_coefs)

    pseudo_p_value = (
        np.sum(np.abs(placebo_coefs) >= np.abs(true_coef)) / n_simulations
    )

    return {
        'true_coefficient': true_coef,
        'placebo_coefficients': placebo_coefs,
        'placebo_mean': np.mean(placebo_coefs),
        'placebo_std': np.std(placebo_coefs),
        'pseudo_p_value': pseudo_p_value,
        'significant_at_05': pseudo_p_value < 0.05,
        'pass_test': pseudo_p_value > 0.05
    }


def plot_placebo(
    placebo_result: dict,
    title: str = "Placebo Test: Distribution of Faux Coefficients"
) -> None:
    """
    可视化安慰剂检验结果
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))

    plt.hist(
        placebo_result['placebo_coefficients'],
        bins=50,
        alpha=0.7,
        color='steelblue',
        edgecolor='white'
    )

    plt.axvline(
        x=placebo_result['true_coefficient'],
        color='red',
        linestyle='--',
        linewidth=2,
        label=f"True Coef = {placebo_result['true_coefficient']:.3f}"
    )

    plt.xlabel('Coefficient Estimate')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()

    textstr = f"Pseudo p-value = {placebo_result['pseudo_p_value']:.3f}"
    plt.text(
        0.05, 0.95, textstr,
        transform=plt.gca().transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    plt.tight_layout()
    plt.show()
```

## 2. 排除政策干扰检验

```python
def exclusion_test(
    df: pd.DataFrame,
    outcome: str,
    treat_var: str,
    post_var: str,
    exclusion_vars: list[dict],
    unit_fe: str,
    time_fe: str
) -> pd.DataFrame:
    """
    排除其他政策干扰：逐步控制可能的混淆变量

    通过标准：核心 DID 系数在不同控制组合下保持稳健
    """
    results = []

    base_formula = f"{outcome} ~ {treat_var}:{post_var}"

    for exclusion in exclusion_vars:
        control_vars = exclusion.get('controls', [])
        description = exclusion.get('description', '')

        formula = base_formula
        if control_vars:
            formula += " + " + " + ".join(control_vars)
        formula += f" + C({unit_fe}) + C({time_fe})"

        model = smf.ols(formula, data=df).fit(
            cov_type='cluster',
            cov_kwds={'groups': df[unit_fe]}
        )

        did_coef = model.params[f'{treat_var}:{post_var}']
        did_se = model.bse[f'{treat_var}:{post_var}']
        did_p = model.pvalues[f'{treat_var}:{post_var}']

        results.append({
            'exclusion_description': description,
            'control_variables': ', '.join(control_vars) if control_vars else 'None',
            'did_coefficient': did_coef,
            'did_std_error': did_se,
            'did_p_value': did_p,
            'significant': did_p < 0.05
        })

    return pd.DataFrame(results)


def common_exclusion_tests() -> list[dict]:
    """
    常见的排除政策干扰检验组合

    参考论文：控制存款利率变化、剔除一线城市样本、收窄样本窗口
    """
    return [
        {
            'description': '基准回归',
            'controls': []
        },
        {
            'description': '控制存款利率变化',
            'controls': ['存款利率', 'shibor_rate']
        },
        {
            'description': '剔除一线城市样本',
            'controls': []
        },
        {
            'description': '收窄样本窗口至政策前后6个月',
            'controls': []
        },
        {
            'description': '剔除疫情期间观测值',
            'controls': []
        }
    ]
```

## 3. 样本调整检验

```python
def sample_adjustment_test(
    df: pd.DataFrame,
    outcome: str,
    treat_var: str,
    post_var: str,
    unit_var: str,
    time_var: str,
    adjustment_functions: list[Callable]
) -> pd.DataFrame:
    """
    样本调整检验：剔除极端或特殊样本后重新估计

    通过标准：核心系数在各种样本调整下保持稳健
    """
    results = []

    formula = f"{outcome} ~ {treat_var}:{post_var} + C({unit_var}) + C({time_var})"
    model = smf.ols(formula, data=df).fit(
        cov_type='cluster',
        cov_kwds={'groups': df[unit_var]}
    )

    results.append({
        'adjustment': '完整样本（基准）',
        'n_obs': int(model.nobs),
        'did_coef': model.params[f'{treat_var}:{post_var}'],
        'did_se': model.bse[f'{treat_var}:{post_var}'],
        'did_p': model.pvalues[f'{treat_var}:{post_var}']
    })

    for adj_func in adjustment_functions:
        df_adj = adj_func(df)
        n_removed = len(df) - len(df_adj)

        model_adj = smf.ols(formula, data=df_adj).fit(
            cov_type='cluster',
            cov_kwds={'groups': df_adj[unit_var]}
        )

        results.append({
            'adjustment': f'剔除{n_removed}个观测值',
            'n_obs': int(model_adj.nobs),
            'did_coef': model_adj.params[f'{treat_var}:{post_var}'],
            'did_se': model_adj.bse[f'{treat_var}:{post_var}'],
            'did_p': model_adj.pvalues[f'{treat_var}:{post_var}']
        })

    return pd.DataFrame(results)


def common_adjustments() -> list[dict]:
    """
    常见的样本调整函数
    """
    return [
        {
            'name': '剔除一线城市',
            'func': lambda df: df[~df['city_tier'].isin([1])]
        },
        {
            'name': '剔除上市不满2年新股',
            'func': lambda df: df[df['listed_years'] >= 2]
        },
        {
            'name': '剔除ST股票',
            'func': lambda df: df[~df['is_st']]
        },
        {
            'name': '剔除极端市值（上下1%）',
            'func': lambda df: df[
                (df['market_cap'] > df['market_cap'].quantile(0.01)) &
                (df['market_cap'] < df['market_cap'].quantile(0.99))
            ]
        }
    ]
```

## 4. 变量替换检验

```python
def variable_substitution_test(
    df: pd.DataFrame,
    original_outcome: str,
    alternative_outcomes: list[str],
    treat_var: str,
    post_var: str,
    unit_var: str,
    time_var: str
) -> pd.DataFrame:
    """
    变量替换检验：使用不同的被解释变量指标进行估计

    通过标准：核心系数方向一致，显著性水平可比较
    """
    results = []

    for outcome_name, outcome_var in alternative_outcomes.items():
        formula = f"{outcome_var} ~ {treat_var}:{post_var} + C({unit_var}) + C({time_var})"

        model = smf.ols(formula, data=df).fit(
            cov_type='cluster',
            cov_kwds={'groups': df[unit_var]}
        )

        results.append({
            'outcome_variable': outcome_name,
            'did_coef': model.params[f'{treat_var}:{post_var}'],
            'did_se': model.bse[f'{treat_var}:{post_var}'],
            'did_p': model.pvalues[f'{treat_var}:{post_var}'],
            'nobs': int(model.nobs)
        })

    return pd.DataFrame(results)
```

## 5. 聚合稳健性报告

```python
def aggregate_robustness_results(
    placebo_result: dict,
    exclusion_df: pd.DataFrame,
    sample_df: pd.DataFrame
) -> dict:
    """
    汇总所有稳健性检验结果

    返回适合直接放入论文表格的数据
    """
    return {
        'placebo_test': {
            'true_coef': placebo_result['true_coefficient'],
            'pseudo_p': placebo_result['pseudo_p_value'],
            'pass': placebo_result['pseudo_p_value'] > 0.05
        },
        'exclusion_tests': {
            'coef_range': [
                exclusion_df['did_coef'].min(),
                exclusion_df['did_coef'].max()
            ],
            'all_significant': exclusion_df['significant'].all()
        },
        'sample_adjustments': {
            'coef_range': [
                sample_df['did_coef'].min(),
                sample_df['did_coef'].max()
            ],
            'coef_std': sample_df['did_coef'].std()
        }
    }


def format_robustness_table(
    exclusion_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    caption: str = "稳健性检验结果"
) -> str:
    """
    格式化稳健性检验结果表格（LaTeX格式）
    """
    combined = pd.concat([
        exclusion_df[['adjustment', 'did_coef', 'did_se', 'did_p']],
        sample_df[['adjustment', 'did_coef', 'did_se', 'did_p']]
    ])

    table = combined.to_latex(index=False)
    return table
```

## 报告规范

### 必须包含的稳健性检验

| 检验 | 最低要求 | 理想要求 |
|------|----------|----------|
| 安慰剂检验 | 500次模拟 | 1000次模拟 |
| 排除干扰 | 控制1-2个主要混淆变量 | 控制所有可能的混淆变量 |
| 样本调整 | 1-2种调整 | 4种以上调整 |
| 时间窗口 | 基准窗口 | 多种窗口宽度 |

### 显著性符号规范

| 符号 | p值范围 |
|------|---------|
| \*\*\* | p < 0.01 |
| \*\* | p < 0.05 |
| \* | p < 0.1 |
| (括号) | 标准误 |

## 参考论文

本文档参考《生成式AI在招商银行零售客户经营中的应用及成效分析》第四章稳健性检验：
- 安慰剂检验：1000次随机模拟，伪p值=0.002
- 排除政策干扰：核心系数在0.263-0.284之间波动
- PSM+Rosenbaum：Γ=3.0时p值上界0.634 > 0.05