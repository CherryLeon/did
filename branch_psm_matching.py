import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')


class BranchLevelPSM:
    """
    分行层面的倾向得分匹配（PSM）类
    用于检验处理组与对照组的基线可比性
    """

    def __init__(self, data):
        self.data = data.copy()
        self.matched_data = None
        self.propensity_scores = None
        self.model = None
        self.matched_branch_pairs = None

    def _validate_required_fields(self):
        """
        验证所需字段是否存在
        """
        required_fields = ['is_treatment', 'is_post']
        optional_fields = ['ln_branch_size', 'ln_city_gdp', 'pre_treatment_cemi_mean',
                          'digital_maturity', 'staff_count']

        missing_required = [f for f in required_fields if f not in self.data.columns]
        if missing_required:
            raise ValueError("数据中缺少必需字段: {}".format(missing_required))

        missing_optional = [f for f in optional_fields if f not in self.data.columns]
        if missing_optional:
            print("[WARNING] 以下可选字段不存在，将不使用: {}".format(missing_optional))

        return True

    def prepare_branch_level_data(self):
        """
        准备分行层面的基线数据（政策前）
        """
        print("="*70)
        print("准备分行层面的基线数据")
        print("="*70)

        self._validate_required_fields()

        pre_df = self.data[self.data['is_post'] == 0].copy()

        agg_dict = {
            'is_treatment': 'first'
        }

        if 'ln_branch_size' in self.data.columns:
            agg_dict['ln_branch_size'] = 'first'
        if 'ln_city_gdp' in self.data.columns:
            agg_dict['ln_city_gdp'] = 'first'
        if 'pre_treatment_cemi_mean' in self.data.columns:
            agg_dict['pre_treatment_cemi_mean'] = 'first'
        if 'digital_maturity' in self.data.columns:
            agg_dict['digital_maturity'] = 'first'
        if 'staff_count' in self.data.columns:
            agg_dict['staff_count'] = 'first'

        branch_baseline = pre_df.groupby('branch_id').agg(agg_dict).reset_index()

        print("\n基线数据信息:")
        print("  时间范围: 政策前 (2022年3月 - 2023年2月)")
        print("  分行数量: {} 家".format(len(branch_baseline)))
        print("  处理组: {} 家".format(sum(branch_baseline['is_treatment'] == 1)))
        print("  对照组: {} 家".format(sum(branch_baseline['is_treatment'] == 0)))

        self.branch_baseline = branch_baseline
        return branch_baseline

    def estimate_propensity_score(self, covariates):
        """
        估计倾向得分

        参数:
            covariates: 用于估计倾向得分的协变量列表
        """
        print("\n" + "="*70)
        print("步骤1: 估计倾向得分")
        print("="*70)

        available_covariates = [c for c in covariates if c in self.branch_baseline.columns]
        if len(available_covariates) < len(covariates):
            missing = set(covariates) - set(available_covariates)
            print("[WARNING] 以下协变量不可用，将不使用: {}".format(missing))

        X = self.branch_baseline[available_covariates].copy()
        X = sm.add_constant(X)
        y = self.branch_baseline['is_treatment'].values

        self.model = sm.Logit(y, X)
        result = self.model.fit(disp=0)

        self.propensity_scores = result.predict(X)
        self.branch_baseline['propensity_score'] = self.propensity_scores

        print("\nLogistic回归模型结果:")
        print(result.summary())

        treated_ps = self.branch_baseline[self.branch_baseline['is_treatment'] == 1]['propensity_score']
        control_ps = self.branch_baseline[self.branch_baseline['is_treatment'] == 0]['propensity_score']

        print("\n倾向得分分布:")
        print("  处理组均值: {:.4f}, 标准差: {:.4f}".format(treated_ps.mean(), treated_ps.std()))
        print("  对照组均值: {:.4f}, 标准差: {:.4f}".format(control_ps.mean(), control_ps.std()))

        return self.branch_baseline

    def perform_matching(self, method='nearest', caliper=0.05):
        """
        执行倾向得分匹配
        """
        print("\n" + "="*70)
        print("步骤2: 执行倾向得分匹配")
        print("="*70)

        treated_branches = self.branch_baseline[self.branch_baseline['is_treatment'] == 1].copy()
        control_branches = self.branch_baseline[self.branch_baseline['is_treatment'] == 0].copy()

        print("\n匹配前分行数量:")
        print("  处理组: {} 家".format(len(treated_branches)))
        print("  对照组: {} 家".format(len(control_branches)))

        matched_pairs = []
        used_control = set()

        for idx, treated_row in treated_branches.iterrows():
            treated_id = treated_row['branch_id']
            treated_ps = treated_row['propensity_score']

            distances = np.abs(control_branches['propensity_score'].values - treated_ps)

            if method == 'caliper':
                valid_indices = np.where(distances <= caliper)[0]
                if len(valid_indices) == 0:
                    continue
                min_idx = valid_indices[np.argmin(distances[valid_indices])]
            else:
                min_idx = np.argmin(distances)

            control_id = control_branches.iloc[min_idx]['branch_id']

            if control_id in used_control:
                sorted_indices = np.argsort(distances)
                for idx2 in sorted_indices:
                    potential_control = control_branches.iloc[idx2]['branch_id']
                    if potential_control not in used_control:
                        if method == 'caliper' and distances[idx2] > caliper:
                            break
                        control_id = potential_control
                        min_idx = idx2
                        break
                else:
                    continue

            used_control.add(control_id)
            matched_pairs.append({
                'treated_branch': treated_id,
                'control_branch': control_id,
                'treated_ps': treated_ps,
                'control_ps': control_branches.iloc[min_idx]['propensity_score'],
                'distance': distances[min_idx]
            })

        self.matched_branch_pairs = matched_pairs

        matched_treated_ids = [pair['treated_branch'] for pair in matched_pairs]
        matched_control_ids = [pair['control_branch'] for pair in matched_pairs]
        matched_branch_ids = matched_treated_ids + matched_control_ids

        self.matched_data = self.data[self.data['branch_id'].isin(matched_branch_ids)].copy()

        print("\n匹配后结果:")
        print("  成功匹配: {} 对分行".format(len(matched_pairs)))
        print("  匹配率: {:.2f}%".format(len(matched_pairs)/len(treated_branches)*100))
        print("  总观测数: {}".format(len(self.matched_data)))

        if len(matched_pairs) > 0:
            distances = [pair['distance'] for pair in matched_pairs]
            print("\n匹配对倾向得分差异统计:")
            print("  均值: {:.4f}".format(np.mean(distances)))
            print("  标准差: {:.4f}".format(np.std(distances)))
            print("  最小值: {:.4f}".format(np.min(distances)))
            print("  最大值: {:.4f}".format(np.max(distances)))

        return self.matched_data

    def assess_balance(self, covariates):
        """
        评估匹配后的平衡性
        """
        print("\n" + "="*70)
        print("步骤3: 平衡性检验")
        print("="*70)

        if self.matched_data is None:
            print("错误: 请先执行匹配!")
            return None

        available_covariates = [c for c in covariates if c in self.matched_data.columns]

        results = []

        print("\n{:<25} {:<20} {:<20} {:<15}".format(
            '变量', '匹配前标准化差异', '匹配后标准化差异', '改善幅度'))
        print("-" * 80)

        pre_df = self.data[self.data['is_post'] == 0]
        pre_matched_df = self.matched_data[self.matched_data['is_post'] == 0]

        for var in available_covariates:
            treated_before = pre_df[pre_df['is_treatment'] == 1][var]
            control_before = pre_df[pre_df['is_treatment'] == 0][var]

            mean_diff_before = treated_before.mean() - control_before.mean()
            pooled_std_before = np.sqrt((treated_before.var() + control_before.var()) / 2)
            std_diff_before = mean_diff_before / pooled_std_before if pooled_std_before > 0 else 0

            treated_after = pre_matched_df[pre_matched_df['is_treatment'] == 1][var]
            control_after = pre_matched_df[pre_matched_df['is_treatment'] == 0][var]

            mean_diff_after = treated_after.mean() - control_after.mean()
            pooled_std_after = np.sqrt((treated_after.var() + control_after.var()) / 2)
            std_diff_after = mean_diff_after / pooled_std_after if pooled_std_after > 0 else 0

            improvement = abs(std_diff_before) - abs(std_diff_after)

            results.append({
                'variable': var,
                'std_diff_before': std_diff_before,
                'std_diff_after': std_diff_after,
                'improvement': improvement
            })

            print("{:<25} {:>18.4f} {:>18.4f} {:>13.4f}".format(
                var, std_diff_before, std_diff_after, improvement))

        print("\n" + "="*70)
        print("平衡性评估标准:")
        print("  - 标准化差异 < 0.1: 平衡性良好")
        print("  - 0.1 <= 标准化差异 < 0.25: 平衡性可接受")
        print("  - 标准化差异 >= 0.25: 平衡性较差")
        print("="*70)

        balanced_vars = sum(1 for r in results if abs(r['std_diff_after']) < 0.1)
        print("\n匹配后平衡性良好的变量: {}/{} ({:.1f}%)".format(
            balanced_vars, len(available_covariates), balanced_vars/len(available_covariates)*100))

        return pd.DataFrame(results)

    def save_matched_data(self, output_path='d:\\trae_project\\did\\raw_data\\matched_branch_panel_data.csv'):
        if self.matched_data is not None:
            self.matched_data.to_csv(output_path, index=False)
            print("\n匹配后的数据已保存到: {}".format(output_path))
        else:
            print("错误: 没有匹配后的数据可保存!")


def main():
    print("="*70)
    print("分行层面倾向得分匹配（PSM）分析")
    print("="*70)

    print("\n读取数据...")
    df = pd.read_csv("d:\\trae_project\\did\\raw_data\\branch_panel_data.csv")

    psm = BranchLevelPSM(df)

    psm.prepare_branch_level_data()

    covariates = ['ln_branch_size', 'ln_city_gdp', 'pre_treatment_cemi_mean']
    print("\n使用的协变量: {}".format(covariates))

    psm.estimate_propensity_score(covariates)

    psm.perform_matching(method='nearest', caliper=0.05)

    psm.assess_balance(covariates)

    psm.save_matched_data()

    print("\n" + "="*70)
    print("PSM分析完成!")
    print("="*70)


if __name__ == "__main__":
    main()
