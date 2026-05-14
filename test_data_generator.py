import pandas as pd
import numpy as np
from datetime import datetime


def generate_branch_panel_data(
    n_treatment_branches=47,
    n_control_branches=39,
    treatment_date="2023-03",
    start_date="2022-03",
    end_date="2025-03",
    random_seed=42
):
    np.random.seed(random_seed)
    
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
    months = [date.strftime("%Y-%m") for date in date_range]
    
    n_branches = n_treatment_branches + n_control_branches
    branches = ["Branch_{}".format(i) for i in range(1, n_branches + 1)]
    treatment_branches = branches[:n_treatment_branches]
    control_branches = branches[n_treatment_branches:]
    
    data = []
    for branch_id in branches:
        is_treatment = 1 if branch_id in treatment_branches else 0
        
        for month in months:
            if month >= treatment_date:
                is_post = 1
            else:
                is_post = 0
            
            row = {
                'branch_id': branch_id,
                'month': month,
                'is_treatment': is_treatment,
                'is_post': is_post,
                'treatment_post': is_treatment * is_post
            }
            data.append(row)
    
    df = pd.DataFrame(data)
    return df


def generate_branch_covariates(df, random_seed=42):
    np.random.seed(random_seed)
    
    unique_branches = df['branch_id'].unique()
    branch_features = []
    
    for branch_id in unique_branches:
        is_treatment = df[df['branch_id'] == branch_id]['is_treatment'].iloc[0]
        
        branch_size = np.random.lognormal(mean=11.72, sigma=0.673)
        ln_branch_size = np.log(branch_size)
        
        city_gdp = np.random.lognormal(mean=13.451, sigma=0.887)
        ln_city_gdp = np.log(city_gdp)
        
        staff_count = np.random.randint(10, 100)
        
        if is_treatment:
            digital_maturity = np.random.normal(50, 15)
        else:
            digital_maturity = np.random.normal(45, 15)
        digital_maturity = max(0, min(100, digital_maturity))
        
        features = {
            'branch_id': branch_id,
            'ln_branch_size': ln_branch_size,
            'ln_city_gdp': ln_city_gdp,
            'staff_count': staff_count,
            'digital_maturity': digital_maturity
        }
        branch_features.append(features)
    
    branch_df = pd.DataFrame(branch_features)
    df = df.merge(branch_df, on='branch_id', how='left')
    
    return df


def generate_cemi_sub_indicators(df, random_seed=42):
    np.random.seed(random_seed)
    
    df = df.copy()
    df['month_date'] = pd.to_datetime(df['month'])
    df['month_index'] = (df['month_date'] - df['month_date'].min()).dt.days // 30
    treatment_month_idx = df[df['month'] == '2023-03']['month_index'].iloc[0]
    
    def generate_indicator(row, base_mean, base_std, effect_size):
        base = np.random.normal(base_mean, base_std)
        base += row['month_index'] * 0.05
        
        if row['treatment_post']:
            months_since_treatment = row['month_index'] - treatment_month_idx
            base += effect_size * (1 + 0.08 * max(0, months_since_treatment))
        
        base += np.random.normal(0, base_std * 0.1)
        return base
    
    df['marketing_conversion_rate'] = df.apply(
        lambda row: generate_indicator(row, 2.3, 0.7, 1.5), axis=1
    ).clip(0, 10)
    
    df['cac_reduction_rate'] = df.apply(
        lambda row: generate_indicator(row, 0, 5, 8), axis=1
    ).clip(-20, 30)
    
    df['retention_rate'] = df.apply(
        lambda row: generate_indicator(row, 61.3, 8.4, 5), axis=1
    ).clip(0, 100)
    
    df['aum_growth_rate'] = df.apply(
        lambda row: generate_indicator(row, 0.87, 0.34, 0.5), axis=1
    ).clip(-5, 10)
    
    df['cross_sell_rate'] = df.apply(
        lambda row: generate_indicator(row, 8.6, 2.3, 2), axis=1
    ).clip(0, 30)
    
    return df


def synthesize_cemi_from_sub_indicators(df):
    df = df.copy()
    
    sub_indicators = [
        'marketing_conversion_rate',
        'cac_reduction_rate',
        'retention_rate',
        'aum_growth_rate',
        'cross_sell_rate'
    ]
    
    pre_df = df[df['is_post'] == 0]
    std_indicators = []
    for indicator in sub_indicators:
        pre_mean = pre_df[indicator].mean()
        pre_std = pre_df[indicator].std()
        df['{}_std'.format(indicator)] = (df[indicator] - pre_mean) / pre_std
        std_indicators.append('{}_std'.format(indicator))
    
    df['CEMI'] = df[std_indicators].mean(axis=1) * 10 + 50
    df['CEMI'] = df['CEMI'].clip(0, 100)
    
    return df


def generate_scenario_indicators(df, random_seed=42):
    np.random.seed(random_seed)
    df = df.copy()
    
    treatment_month_idx = df[df['month'] == '2023-03']['month_index'].iloc[0]
    
    def generate_scenario_indicator(row, base_mean, base_std, effect_size):
        base = np.random.normal(base_mean, base_std)
        base += row['month_index'] * 0.08
        
        if row['treatment_post']:
            months_since_treatment = row['month_index'] - treatment_month_idx
            base += effect_size * (1 + 0.1 * max(0, months_since_treatment))
        
        base += np.random.normal(0, base_std * 0.08)
        return max(0, min(100, base))
    
    df['service_resolution_rate'] = df.apply(
        lambda row: generate_scenario_indicator(row, 67.4, 6.8, 30.25), axis=1
    )
    
    df['auto_approval_rate'] = df.apply(
        lambda row: generate_scenario_indicator(row, 64.3, 8.2, 27.36), axis=1
    )
    
    df['wealth_adoption_rate'] = df.apply(
        lambda row: generate_scenario_indicator(row, 45, 8, 10.75), axis=1
    )
    
    return df


def calculate_pre_treatment_cemi_mean(df):
    df = df.copy()
    
    pre_df = df[df['is_post'] == 0]
    pre_cemi_by_branch = pre_df.groupby('branch_id')['CEMI'].mean().reset_index()
    pre_cemi_by_branch.columns = ['branch_id', 'pre_treatment_cemi_mean']
    df = df.merge(pre_cemi_by_branch, on='branch_id', how='left')
    
    return df


def generate_complete_branch_dataset(
    n_treatment_branches=47,
    n_control_branches=39,
    output_path="d:\\trae_project\\did\\raw_data\\branch_panel_data.csv",
    random_seed=42
):
    print("开始生成分行粒度面板数据...")
    
    df = generate_branch_panel_data(
        n_treatment_branches=n_treatment_branches,
        n_control_branches=n_control_branches,
        random_seed=random_seed
    )
    print("[OK] 基础面板生成完成：{} 条记录，{} 家分行".format(len(df), df['branch_id'].nunique()))
    
    df = generate_branch_covariates(df, random_seed=random_seed)
    print("[OK] 分行协变量生成完成")
    
    df = generate_cemi_sub_indicators(df, random_seed=random_seed)
    print("[OK] CEMI子指标生成完成")
    
    df = synthesize_cemi_from_sub_indicators(df)
    print("[OK] CEMI合成完成")
    
    df = generate_scenario_indicators(df, random_seed=random_seed)
    print("[OK] 分场景指标生成完成")
    
    df = calculate_pre_treatment_cemi_mean(df)
    print("[OK] 政策前CEMI均值计算完成")
    
    if output_path:
        df.to_csv(output_path, index=False)
        print("\n数据已保存至: {}".format(output_path))
    
    return df


def validate_branch_dataset(df):
    validation = {
        'sample_size': len(df),
        'unique_branches': df['branch_id'].nunique(),
        'time_periods': df['month'].nunique(),
        'treatment_branches': df[df['is_treatment'] == 1]['branch_id'].nunique(),
        'control_branches': df[df['is_treatment'] == 0]['branch_id'].nunique(),
        'pre_treatment_cemi_mean_treatment': 0,
        'pre_treatment_cemi_mean_control': 0,
        'pre_treatment_cemi_diff': 0,
        'cemi_range': [0, 0]
    }
    
    pre_df = df[df['is_post'] == 0]
    validation['pre_treatment_cemi_mean_treatment'] = pre_df[pre_df['is_treatment'] == 1]['CEMI'].mean()
    validation['pre_treatment_cemi_mean_control'] = pre_df[pre_df['is_treatment'] == 0]['CEMI'].mean()
    validation['pre_treatment_cemi_diff'] = abs(validation['pre_treatment_cemi_mean_treatment'] - validation['pre_treatment_cemi_mean_control'])
    validation['cemi_range'] = [df['CEMI'].min(), df['CEMI'].max()]
    
    return validation


if __name__ == "__main__":
    print("="*60)
    print("    生成式AI在招商银行零售客户经营中的应用成效分析")
    print("    分行粒度数据生成器")
    print("="*60)
    
    df = generate_complete_branch_dataset(random_seed=42)
    validation = validate_branch_dataset(df)
    
    print("\n" + "-"*60)
    print("数据验证结果:")
    print("-"*60)
    print("总观测值: {:,}".format(validation['sample_size']))
    print("分行数: {}".format(validation['unique_branches']))
    print("时间周期数: {}".format(validation['time_periods']))
    print("处理组分行数: {}".format(validation['treatment_branches']))
    print("对照组分行数: {}".format(validation['control_branches']))
    print("\n政策前CEMI均值 (处理组): {:.2f}".format(validation['pre_treatment_cemi_mean_treatment']))
    print("政策前CEMI均值 (对照组): {:.2f}".format(validation['pre_treatment_cemi_mean_control']))
    print("政策前差异: {:.2f}".format(validation['pre_treatment_cemi_diff']))
    print("CEMI范围: [{:.2f}, {:.2f}]".format(validation['cemi_range'][0], validation['cemi_range'][1]))
    
    print("\n数据预览 (前3行):")
    print(df[['branch_id', 'month', 'is_treatment', 'is_post', 'treatment_post', 'CEMI']].head(3))
    
    print("\n所有变量:")
    for col in sorted(df.columns):
        print("  - {}".format(col))
    
    print("\n" + "="*60)
    print("数据生成完成！")
    print("="*60)
