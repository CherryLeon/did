import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def main():
    print("="*80)
    print("  生成式AI在招商银行零售客户经营中的应用成效分析")
    print("  实证分析主程序 - 分行粒度")
    print("="*80)

    print("\n[步骤1] 读取分行粒度面板数据...")
    df = pd.read_csv("d:\\trae_project\\did\\raw_data\\branch_panel_data.csv")
    print("  [OK] 数据读取成功: {:,} 条记录, {} 家分行".format(len(df), df['branch_id'].nunique()))

    print("\n" + "="*80)
    print("[步骤2] 描述性统计")
    print("="*80)

    print("\n核心变量统计:")
    core_vars = ['CEMI', 'is_treatment', 'is_post', 'treatment_post']
    print(df[core_vars].describe())

    print("\n政策前CEMI对比:")
    pre_df = df[df['is_post'] == 0]
    pre_treat_mean = pre_df[pre_df['is_treatment'] == 1]['CEMI'].mean()
    pre_ctrl_mean = pre_df[pre_df['is_treatment'] == 0]['CEMI'].mean()
    print("  处理组: {:.2f}".format(pre_treat_mean))
    print("  对照组: {:.2f}".format(pre_ctrl_mean))
    print("  差异: {:.2f}".format(abs(pre_treat_mean - pre_ctrl_mean)))

    print("\n" + "="*80)
    print("[步骤3] 倾向得分匹配（PSM）")
    print("="*80)

    from branch_psm_matching import BranchLevelPSM
    psm = BranchLevelPSM(df)
    psm.prepare_branch_level_data()

    covariates = ['ln_branch_size', 'ln_city_gdp', 'pre_treatment_cemi_mean']
    print("\n使用的协变量: {}".format(covariates))

    psm.estimate_propensity_score(covariates)
    psm.perform_matching(method='nearest', caliper=0.05)
    psm.assess_balance(covariates)

    matched_data = psm.matched_data
    if matched_data is not None:
        print("\n[OK] PSM匹配完成，匹配后样本量: {}".format(len(matched_data)))
    else:
        print("\n[WARNING] PSM匹配未成功，使用原始数据继续分析")
        matched_data = df

    print("\n" + "="*80)
    print("[步骤4] 基准DID回归")
    print("="*80)

    from baseline_did import BaselineDID
    did = BaselineDID(matched_data)
    did.estimate_baseline()

    print("\n" + "="*80)
    print("[步骤5] 平行趋势检验")
    print("="*80)

    from parallel_trend_test import ParallelTrendTest
    pt_test = ParallelTrendTest(matched_data)
    pt_test.prepare_relative_time()
    pt_test.estimate_event_study()
    pt_test.joint_f_test()

    print("\n" + "="*80)
    print("[步骤6] 异质性分析")
    print("="*80)

    from heterogeneity_analysis import HeterogeneityAnalysis
    ha = HeterogeneityAnalysis(matched_data)
    ha.analyze_by_scenario()

    print("\n" + "="*80)
    print("[步骤7] 稳健性检验")
    print("="*80)

    from robustness_test import RobustnessTest
    rt = RobustnessTest(matched_data)
    rt.placebo_test(n_permutations=200)
    rt.subset_test()

    print("\n" + "="*80)
    print("  实证分析全部完成！")
    print("="*80)


if __name__ == "__main__":
    main()
