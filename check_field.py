import pandas as pd

print("检查 branch_panel_data.csv 中的字段...")
df = pd.read_csv("d:\\trae_project\\did\\raw_data\\branch_panel_data.csv")

print("\n数据形状: {}".format(df.shape))

print("\n检查 pre_treatment_cemi_mean 是否存在:")
if 'pre_treatment_cemi_mean' in df.columns:
    print("  [OK] 字段存在")
    print("  非空值数量: {}".format(df['pre_treatment_cemi_mean'].notna().sum()))
    print("  示例值: {}".format(list(df['pre_treatment_cemi_mean'].head())))
else:
    print("  [ERROR] 字段不存在！")
    print("\n缺失该字段将导致 PSM 匹配失败。")
