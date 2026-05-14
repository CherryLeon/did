import pandas as pd
import numpy as np

# 设置参数
n_treated_branches = 47  # 处理组分行数量
n_control_branches = 39  # 对照组分行数量
start_date = '2022-03-01'  # 开始日期
end_date = '2024-02-29'  # 结束日期

# 生成日期范围
date_range = pd.date_range(start=start_date, end=end_date, freq='ME')

# 处理组处理时间
treatment_start = pd.to_datetime('2023-03-01')
# 对照组处理时间
treatment_start_control = pd.to_datetime('2025-01-01')

# 生成基础数据
print("生成基础数据...")
data = []

# 为每个分行生成控制变量（客户总数和GDP，不随时间变化）
np.random.seed(42)
branch_controls = {}

# 生成处理组的控制变量
for branch in range(n_treated_branches):
    branch_id = f'T{branch}'
    # 客户总数（取对数前）
    customer_count = np.random.uniform(10000, 500000)
    # 城市GDP（取对数前，单位：亿元）
    city_gdp = np.random.uniform(500, 20000)
    branch_controls[branch_id] = {
        'ln_customer_count': np.log(customer_count),
        'ln_city_gdp': np.log(city_gdp),
        'customer_count_raw': customer_count,
        'city_gdp_raw': city_gdp
    }

# 生成对照组的控制变量
for branch in range(n_treated_branches, n_treated_branches + n_control_branches):
    branch_id = f'C{branch}'
    customer_count = np.random.uniform(10000, 500000)
    city_gdp = np.random.uniform(500, 20000)
    branch_controls[branch_id] = {
        'ln_customer_count': np.log(customer_count),
        'ln_city_gdp': np.log(city_gdp),
        'customer_count_raw': customer_count,
        'city_gdp_raw': city_gdp
    }

# 生成处理组数据
for branch in range(n_treated_branches):
    branch_id = f'T{branch}'
    for date in date_range:
        is_treated = date >= treatment_start
        data.append({
            'branch_id': branch_id,
            'group': 'treated',
            'date': date.strftime('%Y-%m-%d'),
            'year': date.year,
            'month': date.month,
            'treatment_start_date': treatment_start.strftime('%Y-%m-%d'),
            'is_treated': is_treated,
            'ln_customer_count': branch_controls[branch_id]['ln_customer_count'],
            'ln_city_gdp': branch_controls[branch_id]['ln_city_gdp'],
            'customer_count_raw': branch_controls[branch_id]['customer_count_raw'],
            'city_gdp_raw': branch_controls[branch_id]['city_gdp_raw']
        })

# 生成对照组数据
for branch in range(n_treated_branches, n_treated_branches + n_control_branches):
    branch_id = f'C{branch}'
    for date in date_range:
        is_treated = date >= treatment_start_control
        data.append({
            'branch_id': branch_id,
            'group': 'control',
            'date': date.strftime('%Y-%m-%d'),
            'year': date.year,
            'month': date.month,
            'treatment_start_date': treatment_start_control.strftime('%Y-%m-%d'),
            'is_treated': is_treated,
            'ln_customer_count': branch_controls[branch_id]['ln_customer_count'],
            'ln_city_gdp': branch_controls[branch_id]['ln_city_gdp'],
            'customer_count_raw': branch_controls[branch_id]['customer_count_raw'],
            'city_gdp_raw': branch_controls[branch_id]['city_gdp_raw']
        })

# 转换为DataFrame
df = pd.DataFrame(data)

# 生成五个独立的结果变量
print("生成结果变量...")
np.random.seed(42)

# 为每个变量定义不同的参数，大幅增加异质性
var_params = {
    'outcome1': {
        'branch_fe_std': 1.5, 
        'time_fe_std': 0.4, 
        'treatment_effect_factor': 1.0, 
        'noise_multiplier': 4.0, 
        'seasonal_amp': 0.08,
        'seasonal_phase': 0,
        'trend_slope': 0.05,
        'response_speed': 1.0,
        'base_level': 0.0,
        'outlier_prob': 0.01,
        'outlier_magnitude': 2.0,
        'smoothness': 0.7
    },
    'outcome2': {
        'branch_fe_std': 2.5, 
        'time_fe_std': 0.8, 
        'treatment_effect_factor': 0.6, 
        'noise_multiplier': 6.0, 
        'seasonal_amp': 0.12,
        'seasonal_phase': np.pi/3,
        'trend_slope': 0.08,
        'response_speed': 0.6,
        'base_level': 0.3,
        'outlier_prob': 0.02,
        'outlier_magnitude': 3.0,
        'smoothness': 0.5
    },
    'outcome3': {
        'branch_fe_std': 1.2, 
        'time_fe_std': 0.3, 
        'treatment_effect_factor': 1.4, 
        'noise_multiplier': 3.5, 
        'seasonal_amp': 0.06,
        'seasonal_phase': np.pi/2,
        'trend_slope': 0.03,
        'response_speed': 1.2,
        'base_level': -0.2,
        'outlier_prob': 0.015,
        'outlier_magnitude': 2.5,
        'smoothness': 0.8
    },
    'outcome4': {
        'branch_fe_std': 3.0, 
        'time_fe_std': 0.6, 
        'treatment_effect_factor': 0.7, 
        'noise_multiplier': 8.0, 
        'seasonal_amp': 0.15,
        'seasonal_phase': 2*np.pi/3,
        'trend_slope': 0.1,
        'response_speed': 0.8,
        'base_level': 0.5,
        'outlier_prob': 0.025,
        'outlier_magnitude': 3.5,
        'smoothness': 0.6
    },
    'outcome5': {
        'branch_fe_std': 1.8, 
        'time_fe_std': 0.5, 
        'treatment_effect_factor': 1.1, 
        'noise_multiplier': 5.0, 
        'seasonal_amp': 0.1,
        'seasonal_phase': np.pi,
        'trend_slope': 0.12,
        'response_speed': 0.9,
        'base_level': -0.1,
        'outlier_prob': 0.02,
        'outlier_magnitude': 2.8,
        'smoothness': 0.75
    }
}

# 为每个分行存储累积的政策效应（避免突然变化）
cumulative_effects = {var: {} for var in var_params.keys()}

for branch in df['branch_id'].unique():
    # 为每个分行和每个变量生成不同的固定效应
    branch_fes = {}
    for var in var_params.keys():
        branch_fes[var] = np.random.normal(0, var_params[var]['branch_fe_std'])
    
    # 获取该分行的所有观测
    branch_data = df[df['branch_id'] == branch]
    
    # 按时间排序
    branch_data = branch_data.sort_values(['year', 'month'])
    
    for idx in branch_data.index:
        row = branch_data.loc[idx]
        
        # 为每个变量计算不同的时间效应和政策效应
        for var in var_params.keys():
            params = var_params[var]
            
            # 计算时间效应（每个变量不同，包含趋势）
            time_fe = np.random.normal(0, params['time_fe_std']) * (row['year'] - 2022)
            # 添加线性趋势
            time_fe += params['trend_slope'] * (row['year'] - 2022)
            # 添加基准水平
            time_fe += params['base_level']
            
            # 计算政策效应（使用累积效应，避免突然变化）
            if row['is_treated']:
                if row['group'] == 'treated':
                    # 实验组：2023年实施
                    treatment_month = pd.to_datetime(row['treatment_start_date']).month
                    months_since_treatment = (row['year'] - 2023) * 12 + (row['month'] - treatment_month)
                else:
                    # 对照组：2025年实施
                    months_since_treatment = (row['year'] - 2025) * 12 + row['month']
                
                # 使用平滑的政策效应函数
                if months_since_treatment <= 0:
                    treatment_intensity = 0
                else:
                    # 使用S形曲线，避免突然变化
                    x = months_since_treatment / 24.0  # 标准化到0-2
                    treatment_intensity = 0.3 * (1 / (1 + np.exp(-5 * (x - 0.5))))
                
                # 添加随机波动，使效果不那么明显
                random_factor = np.random.uniform(0.3, 1.7)
                
                # 使用累积效应，避免突然变化
                if branch not in cumulative_effects[var]:
                    cumulative_effects[var][branch] = 0
                
                # 计算当前时期的政策效应
                current_treatment_effect = 0.3 * treatment_intensity * params['treatment_effect_factor'] * random_factor
                
                # 应用平滑因子
                if params['smoothness'] > 0:
                    # 使用指数平滑
                    cumulative_effects[var][branch] = (params['smoothness'] * cumulative_effects[var][branch] + 
                                                          (1 - params['smoothness']) * current_treatment_effect)
                else:
                    cumulative_effects[var][branch] = current_treatment_effect
                
                treatment_effect_dynamic = cumulative_effects[var][branch]
            else:
                treatment_effect_dynamic = 0
            
            # 生成变量值
            base_value = branch_fes[var] + time_fe + treatment_effect_dynamic
            
            # 添加异常值
            if np.random.random() < params['outlier_prob']:
                outlier = np.random.choice([-1, 1]) * params['outlier_magnitude']
                base_value += outlier
            
            # 添加噪声
            df.loc[idx, var] = base_value + np.random.normal(0, 0.1 * params['noise_multiplier'])

# 计算平均值
df['outcome'] = (df['outcome1'] + df['outcome2'] + df['outcome3'] + df['outcome4'] + df['outcome5']) / 5

# 保存完整数据
output_path = 'staggered_did_data_with_outcomes.csv'
df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"\n数据处理完成:")
print(f"  总观测数: {len(df)}")
print(f"  处理组: {n_treated_branches}家分行")
print(f"  对照组: {n_control_branches}家分行")
print(f"  时间跨度: {start_date} 至 {end_date}")
print(f"  生成的变量: outcome1-5, outcome, ln_customer_count, ln_city_gdp")
print(f"  数据已保存到: {output_path}")

# 显示控制变量统计信息
print(f"\n控制变量统计信息:")
print(f"  ln_customer_count (客户总数对数):")
print(f"    均值: {df['ln_customer_count'].mean():.4f}")
print(f"    标准差: {df['ln_customer_count'].std():.4f}")
print(f"    最小值: {df['ln_customer_count'].min():.4f}")
print(f"    最大值: {df['ln_customer_count'].max():.4f}")
print(f"  ln_city_gdp (城市GDP对数):")
print(f"    均值: {df['ln_city_gdp'].mean():.4f}")
print(f"    标准差: {df['ln_city_gdp'].std():.4f}")
print(f"    最小值: {df['ln_city_gdp'].min():.4f}")
print(f"    最大值: {df['ln_city_gdp'].max():.4f}")
