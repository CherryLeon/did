---
name: did-method
description: 双重差分法（DID）的实施、检验和结果报告。用于因果推断、政策效果评估。
---

# 双重差分法（DID）实施技能

## 何时使用

- 评估政策/技术部署的因果效应
- 处理组 vs 对照组的准自然实验分析
- 需要识别净效应、排除混淆因素的场景

## 核心原理

DID 通过"双重差分"消除处理组和对照组之间不可观测的异质性偏差：

```
DID = (Y_treat,post - Y_treat,pre) - (Y_control,post - Y_control,pre)
    = 处理组时间差异 - 对照组时间差异
```

## 标准 DID 模型

```python
# 双向固定效应 DID 模型
# Y_it = α + β·Treat_i × Post_t + γ_i + δ_t + ε_it

import statsmodels.formula.api as smf
from typing import Optional

def run_baseline_did(
    df: pd.DataFrame,
    outcome: str,
    treat: str,
    post: str,
    unit_fe: str,
    time_fe: str,
    controls: Optional[list[str]] = None
) -> dict:
    """
    基准 DID 回归

    Parameters
    ----------
    df : pd.DataFrame
        面板数据
    outcome : str
        被解释变量列名
    treat : str
        处理组虚拟变量（0/1）
    post : str
        政策后虚拟变量（0/1）
    unit_fe : str
        个体固定效应变量
    time_fe : str
        时间固定效应变量

    Returns
    -------
    dict
        包含系数、p值、置信区间、拟合指标的字典
    """
    formula = f"{outcome} ~ {treat}:{post}"

    if controls:
        formula += " + " + " + ".join(controls)

    model = smf.ols(
        formula,
        data=df
    ).fit(cov_type='cluster', cov_kwds={'groups': df[unit_fe]})

    return {
        'coefficient': model.params.get(f'{treat}:{post}', np.nan),
        'p_value': model.pvalues.get(f'{treat}:{post}', np.nan),
        'ci_lower': model.conf_int().loc[f'{treat}:{post}', 0],
        'ci_upper': model.conf_int().loc[f'{treat}:{post}', 1],
        'nobs': int(model.nobs),
        'r_squared': model.rsquared,
        'aic': model.aic,
        'summary': model.summary().as_text()
    }
```

## 实施步骤

### Step 1: 构造准自然实验

```python
def construct_quasi_experiment(
    df: pd.DataFrame,
    treat_unit_col: str,
    treat_date_col: str,
    data_time_col: str,
    outcome_cols: list[str]
) -> pd.DataFrame:
    """
    构造 DID 所需的核心变量

    - Treat_i: 处理组虚拟变量
    - Post_t: 政策后虚拟变量
    - CEMI: 客户经营成效综合指数（多指标合成）
    """
    df = df.copy()

    df['Treat'] = df[treat_unit_col].isin(df[treat_unit_col].unique()).astype(int)
    df['Post'] = (df[data_time_col] >= df[treat_date_col]).astype(int)
    df['Treat_Post'] = df['Treat'] * df['Post']

    return df


def construct_cemi(
    df: pd.DataFrame,
    indicators: list[str],
    weights: Optional[list[float]] = None
) -> pd.Series:
    """
    构造客户经营成效综合指数（CEMI）
    等权合成或多指标加权平均
    """
    if weights is None:
        weights = [1.0 / len(indicators)] * len(indicators)

    cemi = pd.Series(0.0, index=df.index)
    for indicator, weight in zip(indicators, weights):
        cemi += df[indicator].fillna(0) * weight

    return cemi
```

### Step 2: 平行趋势检验（见 parallel-trend-test 技能）

### Step 3: 基准 DID 回归

```python
def run_did_regression(
    df: pd.DataFrame,
    outcome: str = 'CEMI',
    unit_var: str = '分行代码',
    time_var: str = '交易日期',
    cluster_var: str = '分行代码'
) -> dict:
    """
    运行基准 DID 回归，报告标准误聚类到分行层面
    """
    did_formula = f"{outcome} ~ Treat * Post + C(分行代码) + C(交易日期)"

    model = smf.ols(did_formula, data=df).fit(
        cov_type='cluster',
        cov_kwds={'groups': df[cluster_var]}
    )

    return {
        'did_coef': model.params['Treat:Post'],
        'did_se': model.bse['Treat:Post'],
        'did_pvalue': model.pvalues['Treat:Post'],
        'did_ci': model.conf_int().loc['Treat:Post'].values,
        'nobs': model.nobs,
        'rsquared': model.rsquared
    }
```

## 动态效应（事件研究法）

```python
def event_study_did(
    df: pd.DataFrame,
    outcome: str,
    treat_var: str,
    time_var: str,
    event_time_col: str,
    leads: int = 6,
    lags: int = 6
) -> pd.DataFrame:
    """
    事件研究法 DID：检验处理前后的动态效应

    处理前系数应不显著（支持平行趋势）
    处理后系数应逐渐增大（支持数据飞轮效应）
    """
    df = df.copy()
    df['relative_time'] = (
        (df[time_var] - df[event_time_col]).dt.days / 30
    ).round()

    results = []
    for t in range(-leads, lags + 1):
        if t == 0:
            continue

        df[f'D_{t}'] = (df['relative_time'] == t).astype(int)
        df[f'Treat_D_{t}'] = df[treat_var] * df[f'D_{t}']

        formula = f"{outcome} ~ Treat_D_{t} + C(分行代码) + C(交易日期)"
        model = smf.ols(formula, data=df).fit(
            cov_type='cluster',
            cov_kwds={'groups': df['分行代码']}
        )

        results.append({
            'period': t,
            'coefficient': model.params[f'Treat_D_{t}'],
            'std_error': model.bse[f'Treat_D_{t}'],
            'p_value': model.pvalues[f'Treat_D_{t}'],
            'ci_lower': model.conf_int().loc[f'Treat_D_{t}', 0],
            'ci_upper': model.conf_int().loc[f'Treat_D_{t}', 1]
        })

    return pd.DataFrame(results)


def plot_event_study(
    event_study_df: pd.DataFrame,
    title: str = "Event Study: Dynamic Treatment Effects"
) -> None:
    """
    可视化事件研究结果
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.axvline(x=-0.5, color='gray', linestyle=':', alpha=0.5)

    plt.errorbar(
        event_study_df['period'],
        event_study_df['coefficient'],
        yerr=1.96 * event_study_df['std_error'],
        fmt='o-',
        capsize=4,
        color='steelblue'
    )

    pre_treatment = event_study_df[event_study_df['period'] < 0]
    if (pre_treatment['p_value'] > 0.05).all():
        plt.text(0.02, 0.98, 'Parallel trend supported',
                 transform=plt.gca().transAxes, verticalalignment='top')

    plt.xlabel('Period Relative to Treatment')
    plt.ylabel('Treatment Effect Coefficient')
    plt.title(title)
    plt.tight_layout()
    plt.show()
```

## 分场景异质性分析

```python
def heterogeneity_analysis(
    df: pd.DataFrame,
    outcome: str,
    by_group: str,
    treat_var: str = 'Treat',
    post_var: str = 'Post'
) -> pd.DataFrame:
    """
    分组异质性分析：检验处理效应在不同子样本中的差异
    """
    results = []

    for group_value in df[by_group].unique():
        subset = df[df[by_group] == group_value]

        formula = f"{outcome} ~ {treat_var}:{post_var} + C(分行代码) + C(交易日期)"
        model = smf.ols(formula, data=subset).fit(
            cov_type='cluster',
            cov_kwds={'groups': subset['分行代码']}
        )

        results.append({
            'group': group_value,
            'coefficient': model.params[f'{treat_var}:{post_var}'],
            'std_error': model.bse[f'{treat_var}:{post_var}'],
            'p_value': model.pvalues[f'{treat_var}:{post_var}'],
            'nobs': int(model.nobs)
        })

    return pd.DataFrame(results)
```

## 结果报告规范

### 必须包含的内容

| 内容 | 要求 |
|------|------|
| 核心系数 | β = X.XXX，标准误在括号中 |
| 显著性标记 | * p<0.1, ** p<0.05, *** p<0.01 |
| 聚类标准误 | 说明聚类维度（如"分行层面聚类"） |
| 观测值数量 | N = XXX |
| R² | 报告拟合优度 |
| 固定效应 | 说明个体和时间固定效应 |

### 示例报告

```
基准DID回归结果：
CEMI = 0.287*** (0.042)
       [0.205, 0.369]

控制变量：个体固定效应、时间固定效应
标准误：分行层面聚类稳健标准误
观测值：12,350,424
R²：0.423
```

## 常见陷阱与检验

| 陷阱 | 检验方法 |
|------|----------|
| 平行趋势不满足 | 事件研究法（处理前系数不显著） |
| 处理溢出 | 检验对照组不受处理影响 |
| 处理时机不同 | 交错 DID 或统一处理日期 |
| 预期效应 | 检验政策宣布前是否有提前响应 |

## 参考论文

本文档规范参考《生成式AI在招商银行零售客户经营中的应用及成效分析》第四章 DID 模型设计。