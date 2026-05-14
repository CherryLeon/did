import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import warnings
warnings.filterwarnings('ignore')


class BaselineDID:
    """
    基准DID回归模型
    """
    
    def __init__(self, data):
        self.data = data.copy()
        self.results = {}
        
    def estimate_baseline(self, outcome_var='CEMI', control_vars=None):
        """
        基准DID模型估计
        Y_it = α + β * Treat_i × Post_t + γX_jt + δ_j + λ_t + ε_jt
        """
        print("="*70)
        print("基准DID模型估计")
        print("="*70)
        
        formula = f"{outcome_var} ~ treatment_post"
        
        if control_vars:
            formula += " + " + " + ".join(control_vars)
        
        formula += " + C(branch_id) + C(month)"
        
        print(f"\n模型设定: {formula}")
        
        model = ols(formula, data=self.data).fit(cov_type='cluster',
                                                cov_kwds={'groups': self.data['branch_id']})
        
        self.results['baseline'] = {
            'model': model,
            'treatment_effect': model.params.get('treatment_post', np.nan),
            'std_error': model.bse.get('treatment_post', np.nan),
            'p_value': model.pvalues.get('treatment_post', np.nan),
            'confidence_interval': model.conf_int().loc['treatment_post'].values if 'treatment_post' in model.conf_int().index else [np.nan, np.nan]
        }
        
        print("\n" + "="*70)
        print("回归结果摘要:")
        print("="*70)
        print(model.summary())
        
        print("\n" + "="*70)
        print("核心结果:")
        print("="*70)
        print(f"处理效应系数: {self.results['baseline']['treatment_effect']:.4f}")
        print(f"标准误: {self.results['baseline']['std_error']:.4f}")
        print(f"P值: {self.results['baseline']['p_value']:.4f}")
        print(f"95%置信区间: [{self.results['baseline']['confidence_interval'][0]:.4f}, {self.results['baseline']['confidence_interval'][1]:.4f}]")
        
        return self.results['baseline']
    
    def estimate_with_controls(self, outcome_var='CEMI'):
        """
        逐步加入控制变量的DID模型
        """
        print("\n" + "="*70)
        print("逐步加入控制变量的DID模型")
        print("="*70)
        
        control_sets = [
            [],
            ['ln_branch_size'],
            ['ln_branch_size', 'ln_city_gdp']
        ]
        
        results_list = []
        
        for i, controls in enumerate(control_sets):
            print(f"\n模型 {i+1}: {', '.join(controls) if controls else '无控制变量'}")
            
            result = self.estimate_baseline(outcome_var=outcome_var, control_vars=controls)
            
            results_list.append({
                'model': f'Model {i+1}',
                'controls': ', '.join(controls),
                'treatment_effect': result['treatment_effect'],
                'std_error': result['std_error'],
                'p_value': result['p_value']
            })
        
        results_df = pd.DataFrame(results_list)
        
        print("\n" + "="*70)
        print("逐步回归结果对比:")
        print("="*70)
        print(results_df)
        
        self.results['stepwise'] = results_df
        
        return results_df


def main():
    print("="*70)
    print("基准DID回归分析")
    print("="*70)
    
    print("\n读取数据...")
    df = pd.read_csv("d:\\trae_project\\did\\raw_data\\branch_panel_data.csv")
    
    did = BaselineDID(df)
    
    print("\n" + "-"*70)
    print("1. 基准模型（无控制变量）")
    did.estimate_baseline()
    
    print("\n" + "-"*70)
    print("2. 逐步加入控制变量")
    did.estimate_with_controls()
    
    print("\n" + "="*70)
    print("DID回归分析完成!")
    print("="*70)


if __name__ == "__main__":
    main()
