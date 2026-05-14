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
    n_months: int = 37,
    treatment_date: str = "2023-03",
    n_treatment_branches: int = 47,
    n_control_branches: int = 39
) -> pd.DataFrame:
    """
    生成DID分析所需的面板数据
    """
    # 计算时间范围：2022-03 到 2025-03，共37个月
    start_date = "2022-03"
    end_date = "2025-03"

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

def generate_cemi_index(
    df: pd.DataFrame,
    base_score: float = 50.0,
    treatment_effect: float = 5.0,
    trend_improvement: float = 0.5,
    noise_level: float = 2.0
) -> pd.DataFrame:
    """
    生成客户经营成效综合指数（CEMI）
    """
    df = df.copy()
    
    df['month_num'] = pd.to_datetime(df['month']).dt.to_period('M').astype('int64')
    min_month = df['month_num'].min()
    df['month_index'] = df['month_num'] - min_month
    
    def calculate_cemi(row):
        score = base_score
        
        # 时间趋势
        score += row['month_index'] * 0.2
        
        # 处理效应（政策后）
        if row['treatment_post']:
            # 处理效应随时间递增
            months_since_treatment = row['month_index'] - df[(df['is_treatment'] == 1) & (df['is_post'] == 1)]['month_index'].min()
            score += treatment_effect + trend_improvement * max(0, months_since_treatment)
        
        # 随机噪声
        score += np.random.normal(0, noise_level)
        
        return max(0, score)
    
    df['CEMI'] = df.apply(calculate_cemi, axis=1)
    return df

def generate_scenario_indicators(
    df: pd.DataFrame,
    scenarios: List[str] = ['customer_service', 'credit', 'marketing', 'wealth']
) -> pd.DataFrame:
    """
    生成分场景指标
    """
    df = df.copy()
    
    # 场景效应参数
    scenario_effects = {
        'customer_service': 20.0,  # 客服场景效应最大
        'credit': 15.0,           # 信贷场景效应次之
        'marketing': 8.0,         # 营销场景效应中等
        'wealth': 5.0             # 财富管理场景效应较小
    }
    
    for scenario in scenarios:
        effect_size = scenario_effects[scenario]
        
        def calculate_indicator(row):
            base = np.random.uniform(60, 70)
            
            # 时间趋势
            base += row['month_index'] * 0.1
            
            # 处理效应
            if row['treatment_post']:
                months_since_treatment = row['month_index'] - df[(df['is_treatment'] == 1) & (df['is_post'] == 1)]['month_index'].min()
                base += effect_size * (1 + 0.1 * max(0, months_since_treatment))
            
            # 随机噪声
            base += np.random.normal(0, 3)
            
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

def generate_covariates(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    生成PSM匹配所需的协变量
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

def generate_complete_dataset(
    n_customers: int = 514601,
    n_months: int = 37,
    treatment_date: str = "2023-03",
    output_path: Optional[str] = "d:\trae_project\did\raw_data\panel_data.csv"
) -> pd.DataFrame:
    """
    生成完整的实证数据集
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

def validate_dataset(
    df: pd.DataFrame,
    treatment_date: str = "2023-03"
) -> Dict:
    """
    验证生成的数据集质量
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

# 测试数据生成
if __name__ == "__main__":
    print("开始生成测试数据...")
    df = generate_complete_dataset(n_customers=1000, n_months=37, treatment_date='2023-03')
    print(f"数据生成完成，共 {len(df)} 条记录")
    
    # 验证数据质量
    validation = validate_dataset(df, treatment_date='2023-03')
    
    print('\n数据验证结果:')
    print(f'样本量: {validation["sample_size"]}')
    print(f'唯一客户数: {validation["unique_customers"]}')
    print(f'时间周期数: {validation["time_periods"]}')
    print(f'处理组分行数: {validation["treatment_branches"]}')
    print(f'对照组分行数: {validation["control_branches"]}')
    print(f'政策前处理组均值: {validation["pre_treatment_stats"]["treatment_mean_cemi"]:.2f}')
    print(f'政策前对照组均值: {validation["pre_treatment_stats"]["control_mean_cemi"]:.2f}')
    print(f'政策前差异: {validation["pre_treatment_stats"]["mean_difference"]:.2f}')
    print(f'政策后处理组均值: {validation["post_treatment_stats"]["treatment_mean_cemi"]:.2f}')
    print(f'政策后对照组均值: {validation["post_treatment_stats"]["control_mean_cemi"]:.2f}')
    print(f'政策后差异: {validation["post_treatment_stats"]["mean_difference"]:.2f}')
    print(f'平行趋势满足: {validation["parallel_trend_check"]["parallel_trend_satisfied"]}')
    print(f'处理效应满足: {validation["treatment_effect_check"]["treatment_effect_satisfied"]}')
    
    # 检查变量覆盖
    print('\n变量覆盖检查:')
    required_vars = ['customer_id', 'branch_id', 'month', 'is_treatment', 'is_post', 'treatment_post', 'CEMI', 'service_resolution_rate', 'auto_approval_rate', 'marketing_conversion_rate', 'wealth_adoption_rate', 'age', 'gender', 'income_level', 'tenure_months', 'previous_campaigns', 'has_loan', 'has_credit_card', 'AUM_level', 'branch_size', 'location_type', 'staff_count', 'prior_performance', 'digital_maturity']
    for var in required_vars:
        if var in df.columns:
            print(f'✓ {var}')
        else:
            print(f'✗ {var}')
    
    # 检查时间范围
    print('\n时间范围检查:')
    min_month = df['month'].min()
    max_month = df['month'].max()
    print(f'最小月份: {min_month}')
    print(f'最大月份: {max_month}')
    print(f'时间跨度: {df["month"].nunique()} 个月')
    
    print('\n测试完成！')