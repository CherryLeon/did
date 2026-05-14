import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import warnings
warnings.filterwarnings('ignore')


class ParallelTrendTest:
    """
    平行趋势检验类
    包括事件研究法和联合F检验
    """
    
    def __init__(self, data):
        self.data = data.copy()
        self.results = {}
        
    def prepare_relative_time(self, treatment_date='2023-03'):
        """
        准备相对时间变量
        """
        print("="*70)
        print("准备相对时间变量")
        print("="*70)
        
        self.data['month_date'] = pd.to_datetime(self.data['month'])
        treatment_date_dt = pd.to_datetime(treatment_date)
        
        self.data['relative_month'] = (
            (self.data['month_date'].dt.year - treatment_date_dt.year) * 12 + 
            (self.data['month_date'].dt.month - treatment_date_dt.month)
        )
        
        print(f"相对时间范围: {self.data['relative_month'].min()} 至 {self.data['relative_month'].max()}")
        
        return self.data
    
    def estimate_event_study(self, outcome_var='CEMI', pre_periods=12, post_periods=24):
        """
        事件研究法估计
        """
        print("\n" + "="*70)
        print("事件研究法估计")
        print("="*70)
        
        data = self.data.copy()
        
        min_rel_month = data['relative_month'].min()
        max_rel_month = data['relative_month'].max()
        
        relative_periods = list(range(-pre_periods, post_periods + 1))
        relative_periods = [p for p in relative_periods if min_rel_month <= p <= max_rel_month]
        
        for period in relative_periods:
            if period == -1:
                continue
            if period >= 0:
                data[f'rel_{period}'] = (data['relative_month'] == period) * data['is_treatment']
            else:
                data[f'rel_neg_{abs(period)}'] = (data['relative_month'] == period) * data['is_treatment']
        
        dummy_vars = []
        for period in relative_periods:
            if period == -1:
                continue
            if period >= 0:
                dummy_vars.append(f'rel_{period}')
            else:
                dummy_vars.append(f'rel_neg_{abs(period)}')
        
        formula = f"{outcome_var} ~ {' + '.join(dummy_vars)} + C(branch_id) + C(month)"
        
        model = ols(formula, data=data).fit(cov_type='cluster',
                                            cov_kwds={'groups': data['branch_id']})
        
        coefficients = {}
        for period in relative_periods:
            if period == -1:
                coefficients[period] = 0.0
            elif period >= 0:
                var_name = f'rel_{period}'
                coefficients[period] = model.params.get(var_name, np.nan)
            else:
                var_name = f'rel_neg_{abs(period)}'
                coefficients[period] = model.params.get(var_name, np.nan)
        
        self.results['event_study'] = {
            'model': model,
            'coefficients': coefficients,
            'relative_periods': relative_periods
        }
        
        print("\n政策前系数（应接近0）:")
        for period in sorted([p for p in coefficients.keys() if p < 0]):
            coef = coefficients[period]
            if period != -1 and not pd.isna(coef):
                print(f"  t{period}: {coef:.4f}")
        
        print("\n政策后系数（应显著为正）:")
        for period in sorted([p for p in coefficients.keys() if p >= 0]):
            coef = coefficients[period]
            if not pd.isna(coef):
                print(f"  t+{period}: {coef:.4f}")
        
        return self.results['event_study']
    
    def joint_f_test(self):
        """
        平行趋势联合F检验：检验政策前系数是否联合显著
        """
        print("\n" + "="*70)
        print("平行趋势联合F检验")
        print("="*70)
        
        if 'event_study' not in self.results:
            print("错误: 请先运行事件研究法估计")
            return None
        
        model = self.results['event_study']['model']
        coefficients = self.results['event_study']['coefficients']
        
        pre_coef_names = []
        for period in coefficients.keys():
            if period < -1:
                if period >= 0:
                    var_name = f'rel_{period}'
                else:
                    var_name = f'rel_neg_{abs(period)}'
                if var_name in model.params.index:
                    pre_coef_names.append(var_name)
        
        if len(pre_coef_names) < 2:
            print("政策前系数不足，无法进行联合F检验")
            return None
        
        f_test = model.f_test(pre_coef_names)
        
        print(f"\n联合F检验结果:")
        print(f"  F统计量: {f_test.statistic[0][0]:.4f}")
        print(f"  P值: {f_test.pvalue:.4f}")
        
        parallel_trend_holds = f_test.pvalue > 0.05
        print(f"  平行趋势假设成立: {parallel_trend_holds}")
        
        self.results['joint_f_test'] = {
            'f_statistic': f_test.statistic[0][0],
            'p_value': f_test.pvalue,
            'parallel_trend_holds': parallel_trend_holds
        }
        
        return self.results['joint_f_test']
    
    def test_parallel_trend_simple(self, outcome_var='CEMI'):
        """
        简化版平行趋势检验：时间趋势与处理组交互
        """
        print("\n" + "="*70)
        print("简化版平行趋势检验")
        print("="*70)
        
        data = self.data[self.data['is_post'] == 0].copy()
        
        data['time_trend'] = range(len(data))
        
        formula = f"{outcome_var} ~ is_treatment * time_trend + C(branch_id)"
        
        model = ols(formula, data=data).fit(cov_type='cluster',
                                            cov_kwds={'groups': data['branch_id']})
        
        interaction_coef = model.params.get('is_treatment:time_trend', None)
        interaction_pvalue = model.pvalues.get('is_treatment:time_trend', None)
        
        parallel_trend_holds = interaction_pvalue is None or interaction_pvalue > 0.05
        
        print(f"\n交互项系数: {interaction_coef:.4f}")
        print(f"交互项P值: {interaction_pvalue:.4f}")
        print(f"平行趋势假设成立: {parallel_trend_holds}")
        
        self.results['simple_test'] = {
            'model': model,
            'interaction_coef': interaction_coef,
            'interaction_pvalue': interaction_pvalue,
            'parallel_trend_holds': parallel_trend_holds
        }
        
        return self.results['simple_test']


def main():
    print("="*70)
    print("平行趋势检验（事件研究法）")
    print("="*70)
    
    print("\n读取数据...")
    df = pd.read_csv("d:\\trae_project\\did\\raw_data\\branch_panel_data.csv")
    
    pt_test = ParallelTrendTest(df)
    
    pt_test.prepare_relative_time()
    
    pt_test.estimate_event_study()
    
    pt_test.joint_f_test()
    
    pt_test.test_parallel_trend_simple()
    
    print("\n" + "="*70)
    print("平行趋势检验完成!")
    print("="*70)


if __name__ == "__main__":
    main()
