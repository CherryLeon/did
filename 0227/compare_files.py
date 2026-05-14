import pandas as pd

df1 = pd.read_csv('code/staggered_did_data_with_outcomes.csv')
df2 = pd.read_csv('code/staggered_did_data_matched.csv')

print('文件1: staggered_did_data_with_outcomes.csv')
print(f'总行数: {len(df1)}')
print(f'分行数量: {df1["branch_id"].nunique()}')
print(f'处理组分行数: {(df1["group"] == "treated").sum() // 24}')
print(f'对照组分行数: {(df1["group"] == "control").sum() // 24}')
print(f'时间范围: {df1["date"].min()} 到 {df1["date"].max()}')

print('\n文件2: staggered_did_data_matched.csv')
print(f'总行数: {len(df2)}')
print(f'分行数量: {df2["branch_id"].nunique()}')
print(f'处理组分行数: {(df2["group"] == "treated").sum() // 24}')
print(f'对照组分行数: {(df2["group"] == "control").sum() // 24}')
print(f'时间范围: {df2["date"].min()} 到 {df2["date"].max()}')

print('\n差异分析:')
print(f'行数差异: {len(df1) - len(df2)}')
print(f'分行数量差异: {df1["branch_id"].nunique() - df2["branch_id"].nunique()}')

branches1 = set(df1['branch_id'].unique())
branches2 = set(df2['branch_id'].unique())
print(f'\n文件1独有的分行: {branches1 - branches2}')
print(f'文件2独有的分行: {branches2 - branches1}')
