---
name: parallel-trend-test
description: 平行趋势假设检验的实施规范，包括事件研究法、联合F检验和可视化。
---

# 平行趋势检验技能

## 何时使用

- DID 基准回归前（必须步骤）
- 验证处理组与对照组在政策前可比较
- 分析处理效应的动态演变

## 平行趋势假设

**核心假设**：如果不存在处理，处理组和对照组的结果趋势应该平行。

```
E[Y_it(1) - Y_it(0) | X] = E[Y_jt(1) - Y_jt(0) | X]  for all t < T
```

即：政策前，处理组和对照组应该有相同的时间趋势。

## 实施方法

### 1. 事件研究法（Event Study）

```python
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

def event_study_specification(
    df: pd.DataFrame,
    outcome: str,
    treat_var: str,
    time_var: str,
    event_time_var: str,
    pre_periods: int = 6,
    post_periods: int = 6,
    unit_fe: str = 'unit_id',
    time_fe: str = 'time_id'
) -> pd.DataFrame:
    """
    事件研究法 DID 规范

    模型：
    Y_it = α + Σ(k=-K,K≠0) β_k · D_{i,k} + γ_i + δ_t + ε_it

    检验：
    - 处理前系数（k<0）：应不显著（支持平行趋势）
    - 处理后系数（k>0）：应逐渐增大（支持动态效应）
    """
    df = df.copy()

    df['relative_period'] = (
        (df[time_var] - df[event_time_var]).dt.days / 30
    ).round().astype(int)

    results = []

    for period in range(-pre_periods, post_periods + 1):
        if period == 0:
            continue

        df[f'D_{period}'] = (df['relative_period'] == period).astype(int)
        df[f'Treat_D_{period}'] = df[treat_var] * df[f'D_{period}']

        formula = (
            f"{outcome} ~ Treat_D_{period} "
            f"+ C({unit_fe}) + C({time_fe})"
        )

        model = smf.ols(formula, data=df).fit(
            cov_type='cluster',
            cov_kwds={'groups': df[unit_fe]}
        )

        results.append({
            'period': period,
            'coefficient': model.params[f'Treat_D_{period}'],
            'std_error': model.bse[f'Treat_D_{period}'],
            'p_value': model.pvalues[f'Treat_D_{period}'],
            'conf_int_lower': model.conf_int().loc[f'Treat_D_{period}', 0],
            'conf_int_upper': model.conf_int().loc[f'Treat_D_{period}', 1],
            'nobs': int(model.nobs)
        })

        df.drop(f'D_{period}', axis=1, inplace=True)
        df.drop(f'Treat_D_{period}', axis=1, inplace=True)

    return pd.DataFrame(results)


def joint_f_test_pre_periods(
    event_study_df: pd.DataFrame,
    pre_periods: list[int]
) -> dict:
    """
    处理前各期系数的联合 F 检验

    H0: 所有处理前系数均为0（支持平行趋势）
    """
    pre_results = event_study_df[
        event_study_df['period'].isin(pre_periods)
    ]

    k = len(pre_results)
    n = pre_results['nobs'].iloc[0]
    p = len(pre_periods)

    coefs = pre_results['coefficient'].values
    ses = pre_results['std_error'].values

    f_stat = np.sum((coefs / ses) ** 2) / p

    from scipy.stats import f as f_dist
    p_value = 1 - f_dist.cdf(f_stat, p, n - p)

    return {
        'f_statistic': f_stat,
        'p_value': p_value,
        'parallel_trend_supported': p_value > 0.05
    }
```

### 2. 图形可视化

```python
import matplotlib.pyplot as plt

def plot_parallel_trend(
    event_study_results: pd.DataFrame,
    title: str = "Event Study: Parallel Trend Test",
    figsize: tuple = (12, 6)
) -> None:
    """
    可视化事件研究结果

    要求：
    - 处理前（period < 0）：系数应在0附近，置信区间包含0
    - 处理后（period > 0）：系数应为正，置信区间不包含0
    """
    fig, ax = plt.subplots(figsize=figsize)

    pre_periods = event_study_results[
        event_study_results['period'] < 0
    ]
    post_periods = event_study_results[
        event_study_results['period'] > 0
    ]

    ax.fill_between(
        pre_periods['period'],
        pre_periods['conf_int_lower'],
        pre_periods['conf_int_upper'],
        alpha=0.3,
        color='gray',
        label='Pre-treatment'
    )

    ax.fill_between(
        post_periods['period'],
        post_periods['conf_int_lower'],
        post_periods['conf_int_upper'],
        alpha=0.3,
        color='steelblue',
        label='Post-treatment'
    )

    ax.errorbar(
        pre_periods['period'],
        pre_periods['coefficient'],
        yerr=1.96 * pre_periods['std_error'],
        fmt='o',
        color='gray',
        markersize=8,
        capsize=5,
        capthick=2
    )

    ax.errorbar(
        post_periods['period'],
        post_periods['coefficient'],
        yerr=1.96 * post_periods['std_error'],
        fmt='s',
        color='steelblue',
        markersize=8,
        capsize=5,
        capthick=2
    )

    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.axvline(x=-0.5, color='red', linestyle=':', alpha=0.7, linewidth=2)

    ax.set_xlabel('Period Relative to Treatment', fontsize=12)
    ax.set_ylabel('Treatment Effect Coefficient', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    ax.legend(loc='upper left')

    pre_significant = (pre_periods['p_value'] < 0.05).any()
    post_significant = (post_periods['p_value'] < 0.05).all()

    annotation_texts = []
    if not pre_significant:
        annotation_texts.append("Parallel trend: Supported ✓")
    else:
        annotation_texts.append("Parallel trend: NOT Supported ✗")

    if post_significant:
        annotation_texts.append("Post-treatment effects: Significant ✓")
    else:
        annotation_texts.append("Post-treatment effects: Mixed")

    annotation = '\n'.join(annotation_texts)
    ax.annotate(
        annotation,
        xy=(0.02, 0.98),
        xycoords='axes fraction',
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )

    plt.tight_layout()
    plt.show()


def plot_treatment_timing(
    df: pd.DataFrame,
    unit_var: str,
    event_time_var: str,
    time_var: str
) -> None:
    """
    可视化处理时间的分布
    """
    treatment_timing = df.groupby(unit_var)[event_time_var].first()

    plt.figure(figsize=(10, 5))
    treatment_timing.value_counts().sort_index().plot(kind='bar')
    plt.xlabel('Treatment Start Date')
    plt.ylabel('Number of Units')
    plt.title('Distribution of Treatment Timing')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
```

### 3. 动态效应分析

```python
def dynamic_effects_summary(
    event_study_df: pd.DataFrame
) -> dict:
    """
    动态效应汇总：分析处理后各期效应的大小演变
    """
    post_effects = event_study_df[event_study_df['period'] > 0]

    return {
        'immediate_effect': post_effects[
            post_effects['period'] == post_effects['period'].min()
        ]['coefficient'].values[0],

        'long_run_effect': post_effects[
            post_effects['period'] == post_effects['period'].max()
        ]['coefficient'].values[0],

        'effect_growth': post_effects['coefficient'].iloc[-1] / max(
            post_effects['coefficient'].iloc[0], 0.001
        ) - 1,

        'months_to_significance': int(
            post_effects[post_effects['p_value'] < 0.05]['period'].min()
        ),

        'cumulative_effect': post_effects['coefficient'].sum()
    }


def compare_pre_trends(
    df: pd.DataFrame,
    outcome: str,
    treat_var: str,
    time_var: str,
    pre_periods: int = 6
) -> dict:
    """
    比较处理组与对照组在政策前的趋势差异

    方法：计算各期的均值差异，进行t检验
    """
    df_sorted = df.sort_values(time_var)

    results = []

    for t in range(1, pre_periods + 1):
        period_data = df_sorted.groupby(treat_var)[outcome].nth(t)

        treated_mean = period_data.get(1, np.nan)
        control_mean = period_data.get(0, np.nan)

        from scipy.stats import ttest_ind
        treated_vals = df_sorted[
            (df_sorted[treat_var] == 1) &
            (df_sorted[time_var] <= df_sorted[time_var].nlargest(t))
        ][outcome]
        control_vals = df_sorted[
            (df_sorted[treat_var] == 0) &
            (df_sorted[time_var] <= df_sorted[time_var].nlargest(t))
        ][outcome]

        t_stat, p_value = ttest_ind(treated_vals, control_vals)

        results.append({
            'period': -t,
            'treated_mean': treated_mean,
            'control_mean': control_mean,
            'difference': treated_mean - control_mean,
            't_statistic': t_stat,
            'p_value': p_value
        })

    return pd.DataFrame(results)
```

## 检验标准

### 通过条件

| 检验 | 通过标准 |
|------|----------|
| 处理前系数显著性 | 所有 p > 0.05 |
| 联合 F 检验 | p > 0.05 |
| 处理后系数 | 逐渐显著（至少 1 期 p < 0.1） |

### 论文报告要求

必须包含：
1. 事件研究图（Figure）
2. 处理前各期系数和标准误表格
3. 联合 F 检验统计量和 p 值
4. 平行趋势假设是否成立的明确结论

## 常见问题

### 问题1：处理前系数显著

**可能原因**：
- 平行趋势假设不满足
- 处理存在预期效应
- 混淆因素未被控制

**解决方案**：
- 考虑使用更早的基线
- 增加控制变量
- 使用交错 DID（Callaway & Sant'Anna 2021）

### 问题2：处理后效应不增反而下降

**可能原因**：
- 存在长期负效应
- 数据问题
- 处理效应确实随时间衰减

**解决方案**：
- 检查数据处理是否正确
- 报告衰减模式
- 考虑非单调处理效应

## 参考论文

本文档参考《生成式AI在招商银行零售客户经营中的应用及成效分析》第四章平行趋势检验：
- 处理前 6 期系数均不显著（p值 0.447-0.812）
- 联合 F 检验：F(6,85)=0.43, p=0.858
- 动态效应：t+1期0.156 → t+6期0.358（翻倍）