import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class HeterogeneityAnalysis:
    """
    异质性分析类
    包括分场景分析（智能客服、消费信贷、财富管理）
    """
    
    def __init__(self, data):
        self.data = data.copy()
        self.results = {}
        
    def analyze_by_scenario(self):
        """
        分场景异质性分析
        """
        print("="*70)
        print("分场景异质性分析")
        print("="*70)
        
        scenarios = {
            'service_resolution_rate': '智能客服场景',
            'auto_approval_rate': '消费信贷场景',
            'wealth_adoption_rate': '财富管理场景'
        }
        
        results_list = []
        
        for var_name, scenario_name in scenarios.items():
            print(f"\n{'='*70}")
            print(f"{scenario_name}")
            print(f"{'='*70}")
            
            result = self._estimate_did_for_outcome(var_name)
            
            results_list.append({
                'scenario': scenario_name,
                'outcome_var': var_name,
                'treatment_effect': result.get('treatment_effect', np.nan),
                'std_error': result.get('std_error', np.nan),
                'p_value': result.get('p_value', np.nan)
            })
        
        results_df = pd.DataFrame(results_list)
        
        print("\n" + "="*70)
        print("分场景结果汇总:")
        print("="*70)
        print(results_df)
        
        self.results['scenario'] = results_df
        
        return results_df
    
    def _estimate_did_for_outcome(self, outcome_var):
        """
        为特定结果变量估计DID模型
        """
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
        
        formula = f"{outcome_var} ~ treatment_post + C(branch_id) + C(month)"
        
        model = ols(formula, data=self.data).fit(cov_type='cluster',
                                                cov_kwds={'groups': self.data['branch_id']})
        
        treatment_effect = model.params.get('treatment_post', np.nan)
        std_error = model.bse.get('treatment_post', np.nan)
        p_value = model.pvalues.get('treatment_post', np.nan)
        
        print(f"处理效应: {treatment_effect:.4f}")
        print(f"标准误: {std_error:.4f}")
        print(f"P值: {p_value:.4f}")
        
        return {
            'model': model,
            'treatment_effect': treatment_effect,
            'std_error': std_error,
            'p_value': p_value
        }


def main():
    print("="*70)
    print("异质性分析")
    print("="*70)
    
    print("\n读取数据...")
    df = pd.read_csv("d:\\trae_project\\did\\raw_data\\branch_panel_data.csv")
    
    ha = HeterogeneityAnalysis(df)
    ha.analyze_by_scenario()
    
    print("\n" + "="*70)
    print("异质性分析完成!")
    print("="*70)


if __name__ == "__main__":
    main()
