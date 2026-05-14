import pandas as pd

df = pd.read_csv("d:\\trae_project\\did\\raw_data\\branch_panel_data.csv")

print("="*80)
print("分行粒度面板数据验证报告")
print("="*80)

print("\n1. 数据基本信息:")
print("-"*80)
print(f"总观测值数: {len(df):,}")
print(f"分行数: {df['branch_id'].nunique()}")
print(f"时间周期数: {df['month'].nunique()}")
print(f"时间范围: {df['month'].min()} - {df['month'].max()}")
print(f"处理组分行数: {df[df['is_treatment'] == 1]['branch_id'].nunique()}")
print(f"对照组分行数: {df[df['is_treatment'] == 0]['branch_id'].nunique()}")

print("\n2. 数据预览:")
print("-"*80)
print(df[['branch_id', 'month', 'is_treatment', 'is_post', 'treatment_post', 'CEMI',
           'marketing_conversion_rate', 'cac_reduction_rate', 'retention_rate',
           'aum_growth_rate', 'cross_sell_rate']].head())

print("\n3. 政策前CEMI平衡验证:")
print("-"*80)
pre_df = df[df['is_post'] == 0]
pre_treat_mean = pre_df[pre_df['is_treatment'] == 1]['CEMI'].mean()
pre_ctrl_mean = pre_df[pre_df['is_treatment'] == 0]['CEMI'].mean()
diff = abs(pre_treat_mean - pre_ctrl_mean)

print(f"政策前CEMI均值 (处理组): {pre_treat_mean:.2f}")
print(f"政策前CEMI均值 (对照组): {pre_ctrl_mean:.2f}")
print(f"两组差异: {diff:.2f}")
print(f"是否满足要求: {'是 (差异<2.0)' if diff < 2.0 else '否'}")

print("\n4. 处理效应趋势验证:")
print("-"*80)
post_df = df[df['is_post'] == 1]
post_treat_mean = post_df[post_df['is_treatment'] == 1]['CEMI'].mean()
post_ctrl_mean = post_df[post_df['is_treatment'] == 0]['CEMI'].mean()
post_diff = post_treat_mean - post_ctrl_mean

print(f"政策后CEMI均值 (处理组): {post_treat_mean:.2f}")
print(f"政策后CEMI均值 (对照组): {post_ctrl_mean:.2f}")
print(f"DID效应: {post_diff:.2f}")
print(f"处理效应是否显著为正: {'是' if post_diff > 0 else '否'}")

print("\n5. 分场景指标政策前验证:")
print("-"*80)
scenario_indicators = ['service_resolution_rate', 'auto_approval_rate', 'wealth_adoption_rate']
for indicator in scenario_indicators:
    pre_treat = pre_df[pre_df['is_treatment'] == 1][indicator].mean()
    pre_ctrl = pre_df[pre_df['is_treatment'] == 0][indicator].mean()
    print(f"{indicator}:")
    print(f"  处理组: {pre_treat:.2f}, 对照组: {pre_ctrl:.2f}, 差异: {abs(pre_treat - pre_ctrl):.2f}")

print("\n6. 协变量政策前验证:")
print("-"*80)
covariates = ['ln_branch_size', 'ln_city_gdp', 'staff_count', 'digital_maturity']
for cov in covariates:
    pre_treat = pre_df[pre_df['is_treatment'] == 1][cov].mean()
    pre_ctrl = pre_df[pre_df['is_treatment'] == 0][cov].mean()
    print(f"{cov}:")
    print(f"  处理组: {pre_treat:.2f}, 对照组: {pre_ctrl:.2f}")

print("\n7. 所有变量列表:")
print("-"*80)
for col in sorted(df.columns):
    print(f"  - {col}")

print("\n" + "="*80)
print("验证完成！")
print("="*80)
