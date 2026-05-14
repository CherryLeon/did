"""
交错双重差分（Staggered DID）分析
适用于实验组分批实施政策的场景

场景描述：
- 实验组A：44个分行，2024年1-12月陆续实施政策
- 对照组B：45个分行，2025年开始实施政策
- 分析政策随时间的动态效果
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class StaggeredDID:
    """
    交错双重差分模型
    
    适用于实验组分批实施政策的场景
    """
    
    def __init__(self, data: pd.DataFrame, 
                 outcome: str, 
                 branch_id: str,
                 time_id: str,
                 treatment_start_date: str = 'treatment_start_date'):
        """
        初始化交错DID模型
        
        参数:
            data: 数据框
            outcome: 结果变量名
            branch_id: 分行标识变量名
            time_id: 时间标识变量名
            treatment_start_date: 政策开始日期变量名
        """
        self.data = data.copy() if data is not None else None
        self.outcome = outcome
        self.branch_id = branch_id
        self.time_id = time_id
        self.treatment_start_date = treatment_start_date
        self.results = {}
        
    def estimate_event_study(self, 
                        relative_periods: List[int] = None,
                        cluster_std: bool = True) -> dict:
        """
        事件研究设计：分析政策效应的动态演变
        
        参数:
            relative_periods: 相对政策期的期数列表，如[-6, -5, ..., 0, ..., 6]
            cluster_std: 是否在分行层面聚类标准误
            
        返回:
            包含事件研究结果的字典
        """
        if relative_periods is None:
            relative_periods = list(range(-12, 13))  # 政策前12期到政策后12期
        
        data = self.data.copy()
        
        # 创建相对时间变量
        data['relative_time'] = data.apply(
            lambda row: self._calculate_relative_time(row), axis=1)
        
        # 创建时间虚拟变量（使用Q前缀避免负号问题）
        for period in relative_periods:
            if period >= 0:
                data[f'rel_time_pos_{period}'] = (data['relative_time'] == period).astype(int)
            else:
                data[f'rel_time_neg_{abs(period)}'] = (data['relative_time'] == period).astype(int)
        
        # 构建事件研究模型
        # 使用相对时间=-1作为基准期
        time_dummies = []
        for p in relative_periods:
            if p == -1:
                continue  # 基准期
            elif p >= 0:
                time_dummies.append(f'rel_time_pos_{p}')
            else:
                time_dummies.append(f'rel_time_neg_{abs(p)}')
        
        formula = f"{self.outcome} ~ {' + '.join(time_dummies)}"
        
        # 添加控制变量（如果有）
        if 'branch_fe' in data.columns:
            formula += " + branch_fe"
        if 'time_fe' in data.columns:
            formula += " + time_fe"
        
        # 添加分行固定效应和时间固定效应
        formula += f" + C({self.branch_id}) + C({self.time_id})"
        
        model = ols(formula, data=data).fit(cov_type='cluster', 
                                            cov_kwds={'groups': data[self.branch_id]})
        
        # 提取系数
        coefficients = {}
        for period in relative_periods:
            if period == -1:
                coefficients[period] = 0  # 基准期
            else:
                # 根据正负期数使用不同的变量名格式
                if period >= 0:
                    coef_name = f'rel_time_pos_{period}'
                else:
                    coef_name = f'rel_time_neg_{abs(period)}'
                
                if coef_name in model.params.index:
                    coefficients[period] = model.params[coef_name]
                else:
                    coefficients[period] = np.nan
        
        self.results['event_study'] = {
            'model': model,
            'coefficients': coefficients,
            'summary': model.summary()
        }
        
        return self.results['event_study']
    
    def _calculate_relative_time(self, row):
        """
        计算相对政策期的时间
        """
        policy_date = row['treatment_start_date']
        current_date = row['date']
        
        if pd.isna(policy_date):
            return np.nan
        
        # 转换为datetime对象
        if isinstance(policy_date, str):
            policy_date = pd.to_datetime(policy_date)
        if isinstance(current_date, str):
            current_date = pd.to_datetime(current_date)
        
        # 计算月份差
        months_diff = (current_date.year - policy_date.year) * 12 + (current_date.month - policy_date.month)
        
        return months_diff
    
    def estimate_twfe(self, 
                   control_vars: Optional[List[str]] = None,
                   cluster_std: bool = True) -> dict:
        """
        双向固定效应模型（TWFE）
        
        参数:
            control_vars: 控制变量列表
            cluster_std: 是否在分行层面聚类标准误
            
        返回:
            包含TWFE结果的字典
        """
        data = self.data.copy()
        
        # 创建处理变量
        data['treated'] = data['is_treated'].astype(int)
        
        # 构建TWFE模型
        formula = f"{self.outcome} ~ treated"
        
        if control_vars:
            formula += " + " + " + ".join(control_vars)
        
        # 添加固定效应
        formula += f" + C({self.branch_id}) + C({self.time_id})"
        
        if cluster_std:
            model = ols(formula, data=data).fit(cov_type='cluster',
                                                cov_kwds={'groups': data[self.branch_id]})
        else:
            model = ols(formula, data=data).fit()
        
        self.results['twfe'] = {
            'model': model,
            'treatment_effect': model.params['treated'],
            'std_error': model.bse['treated'],
            'p_value': model.pvalues['treated'],
            'confidence_interval': model.conf_int().loc['treated'].values,
            'summary': model.summary()
        }
        
        return self.results['twfe']
    
    def plot_event_study(self, figsize: tuple = (14, 6), save_path: str = None):
        """
        绘制事件研究结果
        
        参数:
            figsize: 图表大小
            save_path: 保存路径（如果为None则显示图表）
        """
        
        if 'event_study' not in self.results:
            raise ValueError("请先运行estimate_event_study方法")
        
        coefficients = self.results['event_study']['coefficients']
        periods = sorted(coefficients.keys())
        coeffs = [coefficients[p] for p in periods]
        
        # 获取标准误
        model = self.results['event_study']['model']
        std_errors = []
        for p in periods:
            if p == -1:
                std_errors.append(0)
            else:
                # 根据正负期数使用不同的变量名格式
                if p >= 0:
                    coef_name = f'rel_time_pos_{p}'
                else:
                    coef_name = f'rel_time_neg_{abs(p)}'
                
                if coef_name in model.bse.index:
                    std_errors.append(model.bse[coef_name])
                else:
                    std_errors.append(0)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制系数和置信区间
        x_pos = range(len(periods))
        ax.errorbar(x_pos, coeffs, yerr=std_errors, 
                   fmt='o', capsize=5, capthick=2, linewidth=2,
                   label='政策效应')
        
        # 绘制零线
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
        
        # 标记政策实施期
        if 0 in periods:
            zero_idx = periods.index(0)
            ax.axvline(x=zero_idx, color='green', linestyle=':', alpha=0.7, linewidth=1.5,
                      label='政策实施期')
        
        # 标记基准期
        if -1 in periods:
            base_idx = periods.index(-1)
            ax.axvline(x=base_idx, color='orange', linestyle=':', alpha=0.7, linewidth=1.5,
                      label='基准期')
        
        ax.set_xlabel('相对政策期（月）', fontsize=12)
        ax.set_ylabel('政策效应', fontsize=12)
        ax.set_title('事件研究结果：政策效应的动态演变', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{p}' for p in periods], rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"事件研究结果图已保存到: {save_path}")
        else:
            plt.show()
    
    def plot_trends(self, figsize: tuple = (14, 6), save_path: str = None):
        """
        绘制实验组和对照组的趋势图
        
        参数:
            figsize: 图形大小
            save_path: 保存路径，如果提供则保存图片而不显示
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 左图：原始趋势
        treated_data = self.data[self.data['group'] == 'treated']
        control_data = self.data[self.data['group'] == 'control']
        
        treated_monthly = treated_data.groupby(['year', 'month'])['outcome'].mean().reset_index()
        control_monthly = control_data.groupby(['year', 'month'])['outcome'].mean().reset_index()
        
        treated_monthly['date'] = pd.to_datetime(
            treated_monthly['year'].astype(str) + '-' + 
            treated_monthly['month'].astype(str) + '-01')
        control_monthly['date'] = pd.to_datetime(
            control_monthly['year'].astype(str) + '-' + 
            control_monthly['month'].astype(str) + '-01')
        
        axes[0].plot(treated_monthly['date'], treated_monthly['outcome'], 
                     label='实验组（44家分行）', linewidth=2, marker='o', markersize=4)
        axes[0].plot(control_monthly['date'], control_monthly['outcome'], 
                     label='对照组（45家分行）', linewidth=2, marker='s', markersize=4)
        axes[0].axvline(pd.to_datetime('2024-01-01'), color='red', linestyle='--', 
                         alpha=0.5, label='实验组政策开始')
        axes[0].axvline(pd.to_datetime('2025-01-01'), color='green', linestyle='--', 
                         alpha=0.5, label='对照组政策开始')
        axes[0].set_xlabel('时间', fontsize=12)
        axes[0].set_ylabel('结果变量', fontsize=12)
        axes[0].set_title('实验组与对照组趋势对比', fontsize=13, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # 右图：分批实施效果
        treated_by_month = treated_data[treated_data['is_treated']].groupby(
            ['year', 'month'])['outcome'].mean().reset_index()
        treated_by_month['date'] = pd.to_datetime(
            treated_by_month['year'].astype(str) + '-' + 
            treated_by_month['month'].astype(str) + '-01')
        
        axes[1].plot(treated_by_month['date'], treated_by_month['outcome'], 
                     linewidth=2, marker='o', markersize=4, color='blue')
        axes[1].axvline(pd.to_datetime('2024-01-01'), color='red', linestyle='--', 
                         alpha=0.5, label='政策开始期')
        axes[1].set_xlabel('时间', fontsize=12)
        axes[1].set_ylabel('结果变量（已实施分行）', fontsize=12)
        axes[1].set_title('实验组已实施分行的效果演变', fontsize=13, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"趋势图已保存到: {save_path}")
        else:
            plt.show()
    
    def save_data_to_csv(self, filename: str = 'staggered_did_data.csv'):
        """
        将数据保存到CSV文件
        
        参数:
            filename: 保存的文件名
        """
        if self.data is None:
            raise ValueError("数据为空，无法保存")
        
        self.data.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"数据已保存到: {filename}")
        print(f"数据维度: {self.data.shape}")
        print(f"列名: {list(self.data.columns)}")
    
    def parallel_trend_test(self, pre_period_end: str = '2024-01-01') -> dict:
        """
        平行趋势检验
        
        参数:
            pre_period_end: 政策前的截止日期
            
        返回:
            包含平行趋势检验结果的字典
        """
        data = self.data.copy()
        data['date'] = pd.to_datetime(data['date'])
        
        # 只使用政策前数据
        pre_data = data[data['date'] < pd.to_datetime(pre_period_end)]
        
        # 创建时间趋势变量
        pre_data['time_trend'] = (pre_data['year'] - 2023) * 12 + pre_data['month']
        
        # 构建平行趋势检验模型
        formula = f"{self.outcome} ~ C(group) * time_trend"
        formula += f" + C({self.branch_id})"
        
        model = ols(formula, data=pre_data).fit(cov_type='cluster',
                                                cov_kwds={'groups': pre_data[self.branch_id]})
        
        # 检验交互项系数
        interaction_coef = model.params.get('C(group)[T.treated]:time_trend', None)
        interaction_pvalue = model.pvalues.get('C(group)[T.treated]:time_trend', None)
        
        parallel_trend_holds = interaction_pvalue is None or interaction_pvalue > 0.05
        
        self.results['parallel_trend'] = {
            'model': model,
            'interaction_coef': interaction_coef,
            'interaction_pvalue': interaction_pvalue,
            'parallel_trend_holds': parallel_trend_holds,
            'summary': model.summary()
        }
        
        return self.results['parallel_trend']
    
    def summary(self) -> str:
        """
        输出结果摘要
        """
        summary = f"交错双重差分模型结果\n"
        summary += f"{'='*60}\n"
        summary += f"样本量: {len(self.data)}\n"
        summary += f"实验组分行数: {self.data[self.data['group']=='treated'][self.branch_id].nunique()}\n"
        summary += f"对照组分行数: {self.data[self.data['group']=='control'][self.branch_id].nunique()}\n"
        summary += f"观测期: {self.data['date'].min()} 至 {self.data['date'].max()}\n\n"
        
        if 'twfe' in self.results:
            twfe = self.results['twfe']
            summary += f"TWFE模型结果:\n"
            summary += f"  处理效应: {twfe['treatment_effect']:.4f}\n"
            summary += f"  标准误: {twfe['std_error']:.4f}\n"
            summary += f"  P值: {twfe['p_value']:.4f}\n"
            summary += f"  95%置信区间: [{twfe['confidence_interval'][0]:.4f}, {twfe['confidence_interval'][1]:.4f}]\n\n"
        
        if 'parallel_trend' in self.results:
            pt = self.results['parallel_trend']
            summary += f"平行趋势检验:\n"
            summary += f"  交互项系数: {pt['interaction_coef']:.4f}\n"
            summary += f"  交互项P值: {pt['interaction_pvalue']:.4f}\n"
            summary += f"  平行趋势假设成立: {pt['parallel_trend_holds']}\n\n"
        
        return summary
    
    def placebo_test(self, n_permutations=1000):
        """
        安慰剂检验：随机分配政策实施时间，检验结果的稳健性
        
        参数:
            n_permutations: 置换次数
            
        返回:
            安慰剂检验结果
        """
        import numpy as np
        import warnings
        
        # 过滤statsmodels的警告
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            
            original_effect = self.results.get('twfe', {}).get('treatment_effect', 0)
            placebo_effects = []
            
            for _ in range(n_permutations):
                # 随机分配政策实施时间
                permuted_data = self.data.copy()
                # 随机打乱处理状态
                permuted_data['is_treated'] = np.random.permutation(permuted_data['is_treated'])
                
                # 临时替换数据并重新估计
                original_data = self.data
                self.data = permuted_data
                
                try:
                    # 重新估计TWFE
                    placebo_result = self.estimate_twfe()
                    placebo_effects.append(placebo_result['treatment_effect'])
                except:
                    pass
                
                # 恢复原始数据
                self.data = original_data
        
        # 计算p值
        p_value = (np.abs(placebo_effects) >= np.abs(original_effect)).mean()
        
        self.results['placebo_test'] = {
            'original_effect': original_effect,
            'placebo_effects': placebo_effects,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
        
        return self.results['placebo_test']
    
    def robustness_check(self):
        """
        稳健性检验：使用不同的模型设定
        
        返回:
            稳健性检验结果
        """
        results = {}
        
        # 原始模型
        results['original'] = self.estimate_twfe()
        
        # 不使用聚类标准误
        results['no_cluster'] = self.estimate_twfe(cluster_std=False)
        
        # 其他稳健性检验...
        
        self.results['robustness'] = results
        
        return results
    
    def heterogeneity_analysis(self, group_var):
        """
        异质性分析：按分组变量分析政策效应的差异
        
        参数:
            group_var: 分组变量名
            
        返回:
            异质性分析结果
        """
        results = {}
        
        if group_var not in self.data.columns:
            print(f"分组变量 {group_var} 不存在")
            return results
        
        for group in self.data[group_var].unique():
            group_data = self.data[self.data[group_var] == group]
            original_data = self.data
            self.data = group_data
            
            try:
                results[group] = self.estimate_twfe()
            except:
                pass
            
            self.data = original_data
        
        self.results['heterogeneity'] = results
        
        return results


def main():
    """
    主函数：演示交错DID方法的使用
    """
    print("="*70)
    print("交错双重差分实证")
    print("="*70)
    
    # 读取数据
    print("\n读取数据...")
    data = pd.read_csv('staggered_did_data_with_outcomes.csv')
    
    # 创建交错DID模型实例
    did = StaggeredDID(
        data=data,
        outcome='outcome',
        branch_id='branch_id',
        time_id='date'
    )
    
    print(f"数据读取完成:")
    print(f"  总观测数: {len(data)}")
    print(f"  实验组: {data[data['group']=='treated']['branch_id'].nunique()}家分行")
    print(f"  对照组: {data[data['group']=='control']['branch_id'].nunique()}家分行")
    print(f"  时间跨度: {data['date'].min()} 至 {data['date'].max()}")
    print(f"  包含的变量: {list(data.columns)}")
    
    # 计算合成的Treat、Post、outcome的统计量
    print("\n" + "="*70)
    print("合成变量统计量")
    print("="*70)
    
    # 计算Treat变量（处理组标识）
    data['Treat'] = (data['group'] == 'treated').astype(int)
    
    # 计算Post变量（时间标识）
    # 对于处理组，Post为1表示在处理时间之后；对于对照组，根据处理时间设置
    data['Post'] = 0
    for idx, row in data.iterrows():
        if pd.notna(row['treatment_start_date']):
            treatment_start = pd.to_datetime(row['treatment_start_date'])
            current_date = pd.to_datetime(row['date'])
            if current_date >= treatment_start:
                data.loc[idx, 'Post'] = 1
    
    # 计算Treat×Post交互项
    data['Treat_Post'] = data['Treat'] * data['Post']
    
    # 计算统计量
    stats = {
        'Treat': {
            '均值': data['Treat'].mean(),
            '标准差': data['Treat'].std(),
            '最小值': data['Treat'].min(),
            '最大值': data['Treat'].max()
        },
        'Post': {
            '均值': data['Post'].mean(),
            '标准差': data['Post'].std(),
            '最小值': data['Post'].min(),
            '最大值': data['Post'].max()
        },
        'Treat×Post': {
            '均值': data['Treat_Post'].mean(),
            '标准差': data['Treat_Post'].std(),
            '最小值': data['Treat_Post'].min(),
            '最大值': data['Treat_Post'].max()
        },
        'outcome': {
            '均值': data['outcome'].mean(),
            '标准差': data['outcome'].std(),
            '最小值': data['outcome'].min(),
            '最大值': data['outcome'].max()
        }
    }
    
    # 打印统计量
    print("变量\t\t均值\t\t标准差\t\t最小值\t\t最大值")
    print("-" * 70)
    for var, stat in stats.items():
        print(f"{var}\t\t{stat['均值']:.4f}\t\t{stat['标准差']:.4f}\t\t{stat['最小值']:.4f}\t\t{stat['最大值']:.4f}")
    
    # 保存统计量到CSV
    stats_df = pd.DataFrame(stats).T
    stats_df.to_csv('variable_statistics.csv', encoding='utf-8-sig')
    print(f"\n统计量已保存到: variable_statistics.csv")
    
    # 平行趋势检验
    print("\n" + "="*70)
    print("1. 平行趋势检验")
    print("="*70)
    pt_result = did.parallel_trend_test(pre_period_end='2024-01-01')
    print(f"交互项系数: {pt_result['interaction_coef']:.4f}")
    print(f"交互项P值: {pt_result['interaction_pvalue']:.4f}")
    print(f"平行趋势假设成立: {pt_result['parallel_trend_holds']}")
    
    # TWFE估计
    print("\n" + "="*70)
    print("2. 双向固定效应模型（TWFE）")
    print("="*70)
    twfe_result = did.estimate_twfe(cluster_std=True)
    print(f"处理效应: {twfe_result['treatment_effect']:.4f}")
    print(f"标准误: {twfe_result['std_error']:.4f}")
    print(f"P值: {twfe_result['p_value']:.4f}")
    print(f"95%置信区间: [{twfe_result['confidence_interval'][0]:.4f}, {twfe_result['confidence_interval'][1]:.4f}]")
    
    # 事件研究
    print("\n" + "="*70)
    print("3. 事件研究设计")
    print("="*70)
    event_result = did.estimate_event_study(relative_periods=list(range(-12, 13)))
    
    print("\n政策前效应（应接近0）:")
    for p in range(-6, 0):
        if p in event_result['coefficients']:
            print(f"  t{p}: {event_result['coefficients'][p]:.4f}")
    
    print("\n政策后效应（应显著）:")
    for p in range(0, 7):
        if p in event_result['coefficients']:
            print(f"  t+{p}: {event_result['coefficients'][p]:.4f}")
    
    # 绘制事件研究结果
    print("\n绘制事件研究结果...")
    did.plot_event_study(save_path='event_study.png')
    
    # 绘制趋势图并保存
    print("\n绘制趋势图...")
    did.plot_trends(save_path='trends_plot.png')
    
    # 输出摘要
    print("\n" + "="*70)
    print("分析完成！")
    print("="*70)
    print(did.summary())
    
    # 进行安慰剂检验
    print("\n" + "="*70)
    print("4. 安慰剂检验")
    print("="*70)
    placebo_result = did.placebo_test()
    print(f"原始处理效应: {placebo_result['original_effect']:.4f}")
    print(f"安慰剂检验p值: {placebo_result['p_value']:.4f}")
    print(f"结果稳健: {placebo_result['significant']}")
    
    # 进行稳健性检验
    print("\n" + "="*70)
    print("5. 稳健性检验")
    print("="*70)
    robustness_results = did.robustness_check()
    for name, result in robustness_results.items():
        print(f"{name}: 处理效应 = {result['treatment_effect']:.4f}, P值 = {result['p_value']:.4f}")


if __name__ == "__main__":
    main()
