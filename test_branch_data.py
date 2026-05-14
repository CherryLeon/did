import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from test_data_generator import generate_complete_branch_dataset

print("测试生成分行数据...")
df = generate_complete_branch_dataset(
    output_path="d:\\trae_project\\did\\raw_data\\branch_panel_data.csv",
    random_seed=42
)

print("\n验证数据：")
print(f"数据形状: {df.shape}")
print(f"唯一分行数: {df['branch_id'].nunique()}")
print(f"处理组分行数: {df[df['is_treatment'] == 1]['branch_id'].nunique()}")
print(f"对照组分行数: {df[df['is_treatment'] == 0]['branch_id'].nunique()}")

pre_df = df[df['is_post'] == 0]
print(f"\n政策前CEMI均值（处理组）: {pre_df[pre_df['is_treatment'] == 1]['CEMI'].mean():.2f}")
print(f"政策前CEMI均值（对照组）: {pre_df[pre_df['is_treatment'] == 0]['CEMI'].mean():.2f}")
print(f"政策前差异: {abs(pre_df[pre_df['is_treatment'] == 1]['CEMI'].mean() - pre_df[pre_df['is_treatment'] == 0]['CEMI'].mean()):.2f}")

print("\n保存成功！")
