---
name: data-generator
description: 基于论文背景和实证要求生成DID/PSM分析所需数据，包括面板数据、CEMI指数和分场景指标。
---

# 数据生成专家技能

## 何时使用

- 需要为实证研究创建模拟数据集
- 验证DID模型的估计结果
- 生成符合论文要求的面板数据
- 构建客户经营成效综合指数（CEMI）
- 模拟处理效应的时间递进特征

## 核心功能

### 1. 生成面板数据

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sys

def print_progress(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    """
    打印进度条
    """
    percent = (iteration / float(total)) * 100
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent:.{decimals}f}% {suffix}')
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')
        sys.stdout.flush()

def generate_panel_data(
    n_customers: int = 514601,
    n_months: int = 36,
    treatment_date: str = "2023-03",
    n_treatment_branches: int = 47,
    n_control_branches: int = 39
) -> pd.DataFrame:
    """
    生成DID分析所需的面板数据

    参数
    ----------
    n_customers : int, default=514601
        客户数量，论文要求514,601
    n_months : int, default=36
        观测月数，近三年（政策前12个月，政策后24个月）
    treatment_date : str, default="2023-03"
        处理日期（政策实施月份）
    n_treatment_branches : int, default=47
        处理组分行数量
    n_control_branches : int, default=39
        对照组分行数量

    返回
    -------
    pd.DataFrame
        包含客户-月度观测的面板数据
    """
    start_date = (datetime.strptime(treatment_date, "%Y-%m") - 
                  timedelta(days=365)).strftime("%Y-%m")
    end_date = (datetime.strptime(treatment_date, "%Y-%m") + 
                timedelta(days=730)).strftime("%Y-%m")

    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
    months = [date.strftime("%Y-%m") for date in date_range]

    branches = [f"Branch_{i}" for i in range(1, n_treatment_branches + n_control_branches + 1)]
    treatment_branches = branches[:n_treatment_branches]
    control_branches = branches[n_treatment_branches:]

    data = []
    total = n_customers
    for i in range(n_customers):
        branch = np.random.choice(branches)
        is_treatment = branch in treatment_branches
        
        for month in months:
            is_post = month >= treatment_date
            
            row = {
                'customer_id': f"C{i:06d}",
                'branch_id': branch,
                'month': month,
                'is_treatment': int(is_treatment),
                'is_post': int(is_post),
                'treatment_post': int(is_treatment and is_post)
            }
            data.append(row)
        
        # 打印进度条
        print_progress(i + 1, total, prefix='生成面板数据:', suffix='完成')

    df = pd.DataFrame(data)
    return df
```

### 2. 生成CEMI指数

```python
def generate_cemi_index(
    df: pd.DataFrame,
    base_score: float = 50.0,
    treatment_effect: float = 5.0,
    trend_improvement: float = 0.5,
    noise_level: float = 2.0
) -> pd.DataFrame:
    """
    生成客户经营成效综合指数（CEMI）

    参数
    ----------
    df : pd.DataFrame
        基础面板数据
    base_score : float, default=50.0
        基础分数
    treatment_effect : float, default=5.0
        处理效应大小
    trend_improvement : float, default=0.5
        每月趋势改善幅度
    noise_level : float, default=2.0
        随机噪声水平

    返回
    -------
    pd.DataFrame
        包含CEMI指数的面板数据
    """
    df = df.copy()
    
    df['month_num'] = pd.to_datetime(df['month']).dt.to_period('M').astype('int64')
    min_month = df['month_num'].min()
    df['month_index'] = df['month_num'] - min_month
    
    # 添加客户异质性
    customer_base_scores = {}
    for customer in df['customer_id'].unique():
        # 不同客户有不同的基础分数，模拟真实的客户差异
        customer_base_scores[customer] = base_score + np.random.normal(0, 5)
    
    def calculate_cemi(row):
        # 使用客户特定的基础分数
        score = customer_base_scores[row['customer_id']]
        
        # 时间趋势（带有季节性波动）
        seasonal_factor = 1 + 0.05 * np.sin(2 * np.pi * (row['month_index'] % 12) / 12)
        score += row['month_index'] * 0.2 * seasonal_factor
        
        # 处理效应（政策后）
        if row['treatment_post']:
            # 处理效应随时间递增，且有个体差异
            months_since_treatment = row['month_index'] - df[(df['is_treatment'] == 1) & (df['is_post'] == 1)]['month_index'].min()
            effect = treatment_effect * (1 + np.random.normal(0, 0.2))
            score += effect + trend_improvement * max(0, months_since_treatment)
        
        # 随机噪声（更真实的分布）
        noise = np.random.normal(0, noise_level) * (1 + 0.1 * np.random.random())
        score += noise
        
        return max(0, min(100, score))
    
    df['CEMI'] = df.apply(calculate_cemi, axis=1)
    return df
```

### 3. 生成分场景指标

```python
def generate_scenario_indicators(
    df: pd.DataFrame,
    scenarios: List[str] = ['customer_service', 'credit', 'marketing', 'wealth']
) -> pd.DataFrame:
    """
    生成分场景指标

    参数
    ----------
    df : pd.DataFrame
        基础面板数据
    scenarios : List[str], default=['customer_service', 'credit', 'marketing', 'wealth']
        场景列表

    返回
    -------
    pd.DataFrame
        包含分场景指标的面板数据
    """
    df = df.copy()
    
    # 场景效应参数
    scenario_effects = {
        'customer_service': 20.0,  # 客服场景效应最大
        'credit': 15.0,           # 信贷场景效应次之
        'marketing': 8.0,         # 营销场景效应中等
        'wealth': 5.0             # 财富管理场景效应较小
    }
    
    # 场景基础值范围
    scenario_bases = {
        'customer_service': (65, 75),  # 客服一次解决率基础范围
        'credit': (60, 70),           # 自动审批率基础范围
        'marketing': (2, 5),          # 营销转化率基础范围（百分比）
        'wealth': (8, 15)             # 财富管理采纳率基础范围（百分比）
    }
    
    # 添加客户特定的场景基础值
    customer_scenario_bases = {}
    for customer in df['customer_id'].unique():
        customer_scenario_bases[customer] = {
            'customer_service': np.random.uniform(*scenario_bases['customer_service']),
            'credit': np.random.uniform(*scenario_bases['credit']),
            'marketing': np.random.uniform(*scenario_bases['marketing']),
            'wealth': np.random.uniform(*scenario_bases['wealth'])
        }
    
    for scenario in scenarios:
        effect_size = scenario_effects[scenario]
        base_range = scenario_bases[scenario]
        
        def calculate_indicator(row):
            # 使用客户特定的基础值
            base = customer_scenario_bases[row['customer_id']][scenario]
            
            # 时间趋势（带有季节性波动）
            seasonal_factor = 1 + 0.03 * np.sin(2 * np.pi * (row['month_index'] % 12) / 12)
            base += row['month_index'] * 0.05 * seasonal_factor
            
            # 处理效应（政策后）
            if row['treatment_post']:
                months_since_treatment = row['month_index'] - df[(df['is_treatment'] == 1) & (df['is_post'] == 1)]['month_index'].min()
                # 处理效应随时间递增，且有个体差异
                effect = effect_size * (1 + np.random.normal(0, 0.15))
                base += effect * (1 + 0.08 * max(0, months_since_treatment))
            
            # 随机噪声（更真实的分布）
            noise_level = 2 if scenario in ['customer_service', 'credit'] else 0.5
            noise = np.random.normal(0, noise_level) * (1 + 0.05 * np.random.random())
            base += noise
            
            # 根据场景调整范围
            if scenario in ['marketing', 'wealth']:
                return max(0, min(50, base))  # 转化率和采纳率上限较低
            else:
                return max(0, min(100, base))
        
        df[f'{scenario}_score'] = df.apply(calculate_indicator, axis=1)
    
    # 重命名为论文中的指标名称
    df = df.rename(columns={
        'customer_service_score': 'service_resolution_rate',  # 智能客服一次解决率
        'credit_score': 'auto_approval_rate',               # 消费贷款自动审批率
        'marketing_score': 'marketing_conversion_rate',      # 营销转化率
        'wealth_score': 'wealth_adoption_rate'              # 财富管理推荐采纳率
    })
    
    return df
```

### 4. 生成协变量数据

```python
def generate_covariates(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    生成PSM匹配所需的协变量

    参数
    ----------
    df : pd.DataFrame
        基础面板数据

    返回
    -------
    pd.DataFrame
        包含协变量的面板数据
    """
    df = df.copy()
    
    # 客户特征
    customer_features = []
    unique_customers = df['customer_id'].unique()
    total = len(unique_customers)
    for i, customer in enumerate(unique_customers):
        features = {
            'customer_id': customer,
            'age': np.random.randint(18, 70),
            'gender': np.random.choice(['M', 'F']),
            'income_level': np.random.choice(['low', 'medium', 'high'], p=[0.4, 0.4, 0.2]),
            'tenure_months': np.random.randint(1, 120),
            'previous_campaigns': np.random.randint(0, 10),
            'has_loan': np.random.choice([0, 1], p=[0.7, 0.3]),
            'has_credit_card': np.random.choice([0, 1], p=[0.4, 0.6]),
            'AUM_level': np.random.choice(['low', 'medium', 'high'], p=[0.6, 0.3, 0.1])
        }
        customer_features.append(features)
        
        # 打印进度条
        print_progress(i + 1, total, prefix='生成客户特征:', suffix='完成')
    
    customer_df = pd.DataFrame(customer_features)
    df = df.merge(customer_df, on='customer_id', how='left')
    
    # 分行特征
    branch_features = []
    unique_branches = df['branch_id'].unique()
    total = len(unique_branches)
    for i, branch in enumerate(unique_branches):
        is_treatment = branch in df[df['is_treatment'] == 1]['branch_id'].unique()
        features = {
            'branch_id': branch,
            'branch_size': np.random.choice(['small', 'medium', 'large'], p=[0.3, 0.5, 0.2]),
            'location_type': np.random.choice(['urban', 'suburban', 'rural'], p=[0.6, 0.3, 0.1]),
            'staff_count': np.random.randint(10, 100),
            'prior_performance': np.random.normal(50, 10),
            'digital_maturity': np.random.normal(50, 15) if is_treatment else np.random.normal(40, 15)
        }
        branch_features.append(features)
        
        # 打印进度条
        print_progress(i + 1, total, prefix='生成分行特征:', suffix='完成')
    
    branch_df = pd.DataFrame(branch_features)
    df = df.merge(branch_df, on='branch_id', how='left')
    
    return df
```

### 5. 完整数据生成流程

```python
def generate_complete_dataset(
    n_customers: int = 514601,
    n_months: int = 36,
    treatment_date: str = "2023-03",
    output_path: Optional[str] = "d:\\trae_project\\did\\raw_data\\panel_data.csv"
) -> pd.DataFrame:
    """
    生成完整的实证数据集

    参数
    ----------
    n_customers : int, default=514601
        客户数量
    n_months : int, default=36
        观测月数，近三年
    treatment_date : str, default="2023-03"
        处理日期
    output_path : Optional[str], default="d:\\trae_project\\did\\raw_data\\panel_data.csv"
        输出文件路径，默认保存到raw_data目录

    返回
    -------
    pd.DataFrame
        完整的实证数据集
    """
    # 1. 生成基础面板数据
    df = generate_panel_data(
        n_customers=n_customers,
        n_months=n_months,
        treatment_date=treatment_date
    )
    
    # 2. 生成CEMI指数
    df = generate_cemi_index(df)
    
    # 3. 生成分场景指标
    df = generate_scenario_indicators(df)
    
    # 4. 生成协变量
    df = generate_covariates(df)
    
    # 5. 保存数据
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"数据已保存至: {output_path}")
    
    return df
```

## 数据质量要求

### 1. 样本量要求

| 项目 | 要求 | 说明 |
|------|------|------|
| 客户数量 | ≥ 500,000 | 论文要求514,601个客户 |
| 观测值数量 | ≥ 12,000,000 | 论文要求12,350,424个观测值 |
| 时间跨度 | 36个月 | 近三年（政策前12个月，政策后24个月） |
| 处理组分行 | 47家 | 论文要求的处理组规模 |
| 对照组分行 | 39家 | 论文要求的对照组规模 |

### 2. 变量要求

| 变量类型 | 变量名称 | 要求 |
|----------|----------|------|
| 核心变量 | CEMI | 连续型，范围0-100 |
| 分场景变量 | service_resolution_rate | 智能客服一次解决率，范围0-100 |
| 分场景变量 | auto_approval_rate | 消费贷款自动审批率，范围0-100 |
| 分场景变量 | marketing_conversion_rate | 营销转化率，范围0-100 |
| 分场景变量 | wealth_adoption_rate | 财富管理推荐采纳率，范围0-100 |
| 分组变量 | is_treatment | 处理组虚拟变量(0/1) |
| 分组变量 | is_post | 政策后虚拟变量(0/1) |
| 交互变量 | treatment_post | 处理×政策后交互项 |
| 协变量 | age, gender, income_level | 客户特征 |
| 协变量 | branch_size, location_type | 分行特征 |

### 3. 效果要求

| 效果类型 | 要求 | 说明 |
|----------|------|------|
| 处理效应 | 显著正效应 | DID系数应显著为正 |
| 时间递进 | 效应随时间增加 | 处理后效应逐渐增大 |
| 场景异质性 | 客服>信贷>营销>财富 | 客服和信贷场景效应最大 |
| 平行趋势 | 政策前趋势平行 | 政策前处理组和对照组趋势一致 |
| 稳健性 | 结果稳定 | 不同规格下结果一致 |

## 数据验证

```python
def validate_dataset(
    df: pd.DataFrame,
    treatment_date: str = "2023-03"
) -> Dict:
    """
    验证生成的数据集质量

    参数
    ----------
    df : pd.DataFrame
        生成的数据集
    treatment_date : str, default="2023-03"
        处理日期

    返回
    -------
    Dict
        验证结果
    """
    validation = {
        'sample_size': len(df),
        'unique_customers': df['customer_id'].nunique(),
        'time_periods': df['month'].nunique(),
        'treatment_branches': df[df['is_treatment'] == 1]['branch_id'].nunique(),
        'control_branches': df[df['is_treatment'] == 0]['branch_id'].nunique(),
        'pre_treatment_stats': {},
        'post_treatment_stats': {},
        'parallel_trend_check': {},
        'treatment_effect_check': {}
    }
    
    # 政策前统计
    pre_df = df[df['month'] < treatment_date]
    validation['pre_treatment_stats'] = {
        'control_mean_cemi': pre_df[pre_df['is_treatment'] == 0]['CEMI'].mean(),
        'treatment_mean_cemi': pre_df[pre_df['is_treatment'] == 1]['CEMI'].mean(),
        'mean_difference': abs(pre_df[pre_df['is_treatment'] == 1]['CEMI'].mean() - 
                             pre_df[pre_df['is_treatment'] == 0]['CEMI'].mean())
    }
    
    # 政策后统计
    post_df = df[df['month'] >= treatment_date]
    validation['post_treatment_stats'] = {
        'control_mean_cemi': post_df[post_df['is_treatment'] == 0]['CEMI'].mean(),
        'treatment_mean_cemi': post_df[post_df['is_treatment'] == 1]['CEMI'].mean(),
        'mean_difference': abs(post_df[post_df['is_treatment'] == 1]['CEMI'].mean() - 
                             post_df[post_df['is_treatment'] == 0]['CEMI'].mean())
    }
    
    # 平行趋势检查
    validation['parallel_trend_check']['pre_treatment_diff'] = validation['pre_treatment_stats']['mean_difference']
    validation['parallel_trend_check']['parallel_trend_satisfied'] = validation['pre_treatment_stats']['mean_difference'] < 2.0
    
    # 处理效应检查
    validation['treatment_effect_check']['post_treatment_diff'] = validation['post_treatment_stats']['mean_difference']
    validation['treatment_effect_check']['treatment_effect_satisfied'] = validation['post_treatment_stats']['mean_difference'] > 5.0
    
    return validation
```

## 应用场景

### 场景1：生成基础DID数据

```python
# 生成基础DID分析数据
df = generate_complete_dataset(
    n_customers=514601,
    n_months=24,
    treatment_date="2023-03",
    output_path="data/did_panel_data.csv"
)

# 验证数据质量
validation = validate_dataset(df)
print("数据验证结果:")
print(validation)
```

### 场景2：生成PSM匹配数据

```python
# 生成包含协变量的PSM数据
df = generate_complete_dataset(
    n_customers=100000,  # 为了计算效率，使用10万样本
    n_months=12,
    treatment_date="2023-03",
    output_path="data/psm_data.csv"
)

# 提取政策前数据用于PSM匹配
pre_treatment_df = df[df['month'] < "2023-03"]
pre_treatment_df.to_csv("data/psm_pre_treatment.csv", index=False)
```

### 场景3：生成时间序列数据用于事件研究

```python
# 生成包含月度时间序列的事件研究数据
df = generate_complete_dataset(
    n_customers=50000,
    n_months=36,  # 政策前后各18个月
    treatment_date="2023-03",
    output_path="data/event_study_data.csv"
)
```

## 注意事项

1. **数据规模**：完整数据集（514,601客户×24个月）约1200万行，生成和处理需要足够内存

2. **计算效率**：对于大型数据集，建议使用分块处理或采样方法

3. **参数调整**：根据具体研究需求调整效应大小、噪声水平等参数

4. **数据格式**：确保生成的数据格式符合后续分析工具的要求

5. **结果一致性**：使用固定随机种子确保结果可复现

## 参考论文

本文档基于《生成式AI在招商银行零售客户经营中的应用及成效分析》论文的实证设计要求，确保生成的数据能够支持DID、PSM、平行趋势检验和稳健性检验等实证分析。