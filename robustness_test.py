import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class RobustnessTest:
    """
    稳健性检验类
    包括安慰剂检验、样本调整、子样本分析等
    """
    
    def __init__(self, data):
        self.data = data.copy()
        self.results = {}
        
    def placebo_test(self, n_permutations=500):
        """
        安慰剂检验：随机分配处理组
        """
        print("="*70)
        print("安慰剂检验")
        print("="*70)
        
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
        
        original_result = self._estimate_did(self.data)
        original_effect = original_result['treatment_effect']
        
        print(f"\n原始处理效应: {original_effect:.4f}")
        
        placebo_effects = []
        
        print(f"\n开始置换检验（{n_permutations}次）...")
        
        for i in range(n_permutations):
            if i % 100 == 0 and i > 0:
                print(f"  已完成 {i} 次...")
            
            permuted_data = self.data.copy()
            
            unique_branches = permuted_data['branch_id'].unique()
            n_treatment = int(0.5 * len(unique_branches))
            permuted_treatment_branches = np.random.choice(unique_branches, n_treatment, replace=False)
            
            branch_to_treatment = {branch: 1 if branch in permuted_treatment_branches else 0 for branch in unique_branches}
            
            permuted_data['is_treatment'] = permuted_data['branch_id'].map(branch_to_treatment)
            permuted_data['treatment_post'] = permuted_data['is_treatment'] * permuted_data['is_post']
            
            try:
                result = self._estimate_did(permuted_data)
                placebo_effects.append(result['treatment_effect'])
            except:
                pass
        
        p_value = (np.abs(placebo_effects) >= np.abs(original_effect)).mean()
        
        print(f"\n安慰剂检验结果:")
        print(f"  原始处理效应: {original_effect:.4f}")
        print(f"  安慰剂效应均值: {np.mean(placebo_effects):.4f}")
        print(f"  安慰剂效应标准差: {np.std(placebo_effects):.4f}")
        print(f"  P值: {p_value:.4f}")
        print(f"  结果显著: {p_value < 0.05}")
        
        self.results['placebo'] = {
            'original_effect': original_effect,
            'placebo_effects': placebo_effects,
            'p_value': p_value
        }
        
        return self.results['placebo']
    
    def _estimate_did(self, data):
        """
        内部函数：估计DID模型
        """
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
        
        formula = "CEMI ~ treatment_post + C(branch_id) + C(month)"
        
        model = ols(formula, data=data).fit(cov_type='cluster',
                                            cov_kwds={'groups': data['branch_id']})
        
        return {
            'treatment_effect': model.params.get('treatment_post', np.nan),
            'std_error': model.bse.get('treatment_post', np.nan),
            'p_value': model.pvalues.get('treatment_post', np.nan)
        }
    
    def subset_test(self):
        """
        子样本稳健性检验：缩短时间窗口
        """
        print("\n" + "="*70)
        print("子样本稳健性检验")
        print("="*70)
        
        data = self.data.copy()
        data['month_date'] = pd.to_datetime(data['month'])
        
        subsets = [
            ('2022-06', '2024-12', '2022-06至2024-12'),
            ('2023-01', '2024-06', '2023-01至2024-06')
        ]
        
        results_list = []
        
        for start_date, end_date, label in subsets:
            print(f"\n子样本: {label}")
            
            subset_data = data[
                (data['month_date'] >= pd.to_datetime(start_date)) & 
                (data['month_date'] <= pd.to_datetime(end_date))
            ].copy()
            
            result = self._estimate_did(subset_data)
            
            results_list.append({
                'subset': label,
                'treatment_effect': result['treatment_effect'],
                'std_error': result['std_error'],
                'p_value': result['p_value'],
                'n_obs': len(subset_data)
            })
            
            print(f"  观测数: {len(subset_data)}")
            print(f"  处理效应: {result['treatment_effect']:.4f}")
            print(f"  P值: {result['p_value']:.4f}")
        
        results_df = pd.DataFrame(results_list)
        self.results['subset'] = results_df
        
        return results_df


def main():
    print("="*70)
    print("稳健性检验")
    print("="*70)
    
    print("\n读取数据...")
    df = pd.read_csv("d:\\trae_project\\did\\raw_data\\branch_panel_data.csv")
    
    rt = RobustnessTest(df)
    
    print("\n" + "-"*70)
    print("1. 安慰剂检验")
    rt.placebo_test(n_permutations=200)
    
    print("\n" + "-"*70)
    print("2. 子样本稳健性检验")
    rt.subset_test()
    
    print("\n" + "="*70)
    print("稳健性检验完成!")
    print("="*70)


if __name__ == "__main__":
    main()
