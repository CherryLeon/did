import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class PropensityScoreMatching:
    """
    倾向得分匹配（PSM）类
    用于检验处理组与对照组的基线可比性
    """
    
    def __init__(self, data):
        """
        初始化PSM类
        
        参数:
            data: 包含处理组和对照组的数据框
        """
        self.data = data.copy()
        self.matched_data = None
        self.propensity_scores = None
        self.model = None
        
    def estimate_propensity_score(self, covariates, treatment_col='group'):
        """
        估计倾向得分
        
        参数:
            covariates: 用于估计倾向得分的协变量列表
            treatment_col: 处理组标识列名
            
        返回:
            包含倾向得分的数据框
        """
        print("="*70)
        print("步骤1: 估计倾向得分")
        print("="*70)
        
        # 创建处理变量（1=处理组，0=对照组）
        self.data['treatment'] = (self.data[treatment_col] == 'treated').astype(int)
        
        # 准备特征矩阵
        X = self.data[covariates].copy()
        y = self.data['treatment'].values
        
        # 标准化特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 拟合Logistic回归模型
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.model.fit(X_scaled, y)
        
        # 预测倾向得分
        self.propensity_scores = self.model.predict_proba(X_scaled)[:, 1]
        self.data['propensity_score'] = self.propensity_scores
        
        # 输出模型结果
        print(f"\nLogistic回归模型系数:")
        for i, var in enumerate(covariates):
            print(f"  {var}: {self.model.coef_[0][i]:.4f}")
        print(f"  截距: {self.model.intercept_[0]:.4f}")
        
        # 输出倾向得分分布
        print(f"\n倾向得分分布:")
        print(f"  处理组均值: {self.data[self.data['treatment']==1]['propensity_score'].mean():.4f}")
        print(f"  对照组均值: {self.data[self.data['treatment']==0]['propensity_score'].mean():.4f}")
        print(f"  处理组标准差: {self.data[self.data['treatment']==1]['propensity_score'].std():.4f}")
        print(f"  对照组标准差: {self.data[self.data['treatment']==0]['propensity_score'].std():.4f}")
        
        return self.data
    
    def perform_matching(self, method='nearest', ratio=1, caliper=0.05):
        """
        执行倾向得分匹配
        
        参数:
            method: 匹配方法 ('nearest'最近邻, 'caliper'卡尺匹配)
            ratio: 匹配比例（1:1, 1:2等）
            caliper: 卡尺宽度（倾向得分差异的最大允许值）
            
        返回:
            匹配后的数据框
        """
        print("\n" + "="*70)
        print("步骤2: 执行倾向得分匹配")
        print("="*70)
        
        # 分离处理组和对照组
        treated = self.data[self.data['treatment'] == 1].copy()
        control = self.data[self.data['treatment'] == 0].copy()
        
        print(f"\n匹配前样本量:")
        print(f"  处理组: {len(treated)} 个观测")
        print(f"  对照组: {len(control)} 个观测")
        
        # 获取唯一的分行（因为控制变量在分行层面固定）
        treated_branches = treated[['branch_id', 'propensity_score']].drop_duplicates()
        control_branches = control[['branch_id', 'propensity_score']].drop_duplicates()
        
        print(f"\n唯一分行数量:")
        print(f"  处理组: {len(treated_branches)} 家分行")
        print(f"  对照组: {len(control_branches)} 家分行")
        
        # 执行最近邻匹配
        matched_pairs = []
        used_control = set()
        
        for idx, treated_row in treated_branches.iterrows():
            treated_id = treated_row['branch_id']
            treated_ps = treated_row['propensity_score']
            
            # 计算与所有对照组的距离
            distances = np.abs(control_branches['propensity_score'].values - treated_ps)
            
            # 应用卡尺
            if method == 'caliper':
                valid_indices = np.where(distances <= caliper)[0]
                if len(valid_indices) == 0:
                    continue
                # 在有效范围内选择最近的
                min_idx = valid_indices[np.argmin(distances[valid_indices])]
            else:
                # 选择最近的
                min_idx = np.argmin(distances)
            
            control_id = control_branches.iloc[min_idx]['branch_id']
            
            # 检查是否已使用
            if control_id in used_control:
                # 寻找下一个最近的
                sorted_indices = np.argsort(distances)
                for idx in sorted_indices:
                    potential_control = control_branches.iloc[idx]['branch_id']
                    if potential_control not in used_control:
                        if method == 'caliper' and distances[idx] > caliper:
                            break
                        control_id = potential_control
                        min_idx = idx
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
        
        # 创建匹配后的数据
        matched_treated_ids = [pair['treated_branch'] for pair in matched_pairs]
        matched_control_ids = [pair['control_branch'] for pair in matched_pairs]
        
        matched_treated = treated[treated['branch_id'].isin(matched_treated_ids)]
        matched_control = control[control['branch_id'].isin(matched_control_ids)]
        
        self.matched_data = pd.concat([matched_treated, matched_control], ignore_index=True)
        
        print(f"\n匹配后样本量:")
        print(f"  成功匹配: {len(matched_pairs)} 对分行")
        print(f"  处理组: {len(matched_treated)} 个观测")
        print(f"  对照组: {len(matched_control)} 个观测")
        print(f"  匹配率: {len(matched_pairs)/len(treated_branches)*100:.2f}%")
        
        # 输出匹配对信息
        print(f"\n匹配对倾向得分差异统计:")
        distances = [pair['distance'] for pair in matched_pairs]
        print(f"  均值: {np.mean(distances):.4f}")
        print(f"  标准差: {np.std(distances):.4f}")
        print(f"  最小值: {np.min(distances):.4f}")
        print(f"  最大值: {np.max(distances):.4f}")
        
        return self.matched_data
    
    def assess_balance(self, covariates):
        """
        评估匹配后的平衡性
        
        参数:
            covariates: 需要评估平衡性的协变量列表
            
        返回:
            平衡性评估结果
        """
        print("\n" + "="*70)
        print("步骤3: 平衡性检验")
        print("="*70)
        
        if self.matched_data is None:
            print("错误: 请先执行匹配!")
            return None
        
        results = []
        
        print(f"\n{'变量':<25} {'匹配前标准化差异':<20} {'匹配后标准化差异':<20} {'改善幅度':<15}")
        print("-" * 80)
        
        for var in covariates:
            # 匹配前
            treated_before = self.data[self.data['treatment']==1][var]
            control_before = self.data[self.data['treatment']==0][var]
            
            mean_diff_before = treated_before.mean() - control_before.mean()
            pooled_std_before = np.sqrt((treated_before.var() + control_before.var()) / 2)
            std_diff_before = mean_diff_before / pooled_std_before if pooled_std_before > 0 else 0
            
            # 匹配后
            treated_after = self.matched_data[self.matched_data['treatment']==1][var]
            control_after = self.matched_data[self.matched_data['treatment']==0][var]
            
            mean_diff_after = treated_after.mean() - control_after.mean()
            pooled_std_after = np.sqrt((treated_after.var() + control_after.var()) / 2)
            std_diff_after = mean_diff_after / pooled_std_after if pooled_std_after > 0 else 0
            
            # 改善幅度
            improvement = abs(std_diff_before) - abs(std_diff_after)
            
            results.append({
                'variable': var,
                'std_diff_before': std_diff_before,
                'std_diff_after': std_diff_after,
                'improvement': improvement
            })
            
            print(f"{var:<25} {std_diff_before:>18.4f} {std_diff_after:>18.4f} {improvement:>13.4f}")
        
        # 总体评估
        print("\n" + "="*70)
        print("平衡性评估标准:")
        print("  - 标准化差异 < 0.1: 平衡性良好")
        print("  - 0.1 ≤ 标准化差异 < 0.25: 平衡性可接受")
        print("  - 标准化差异 ≥ 0.25: 平衡性较差")
        print("="*70)
        
        # 统计平衡性良好的变量数量
        balanced_vars = sum(1 for r in results if abs(r['std_diff_after']) < 0.1)
        print(f"\n匹配后平衡性良好的变量: {balanced_vars}/{len(covariates)} ({balanced_vars/len(covariates)*100:.1f}%)")
        
        return pd.DataFrame(results)
    
    def plot_propensity_score_distribution(self, save_path='psm_distribution.png'):
        """
        绘制倾向得分分布图
        
        参数:
            save_path: 保存路径
        """
        plt.figure(figsize=(12, 5))
        
        # 匹配前
        plt.subplot(1, 2, 1)
        treated_ps = self.data[self.data['treatment']==1]['propensity_score']
        control_ps = self.data[self.data['treatment']==0]['propensity_score']
        
        plt.hist(control_ps, bins=20, alpha=0.5, label='对照组', color='blue', density=True)
        plt.hist(treated_ps, bins=20, alpha=0.5, label='处理组', color='red', density=True)
        plt.xlabel('倾向得分')
        plt.ylabel('密度')
        plt.title('匹配前倾向得分分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 匹配后
        plt.subplot(1, 2, 2)
        if self.matched_data is not None:
            treated_ps_matched = self.matched_data[self.matched_data['treatment']==1]['propensity_score']
            control_ps_matched = self.matched_data[self.matched_data['treatment']==0]['propensity_score']
            
            plt.hist(control_ps_matched, bins=20, alpha=0.5, label='对照组', color='blue', density=True)
            plt.hist(treated_ps_matched, bins=20, alpha=0.5, label='处理组', color='red', density=True)
            plt.xlabel('倾向得分')
            plt.ylabel('密度')
            plt.title('匹配后倾向得分分布')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n倾向得分分布图已保存到: {save_path}")
        plt.close()
    
    def save_matched_data(self, output_path='matched_data.csv'):
        """
        保存匹配后的数据
        
        参数:
            output_path: 输出文件路径
        """
        if self.matched_data is not None:
            self.matched_data.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"\n匹配后的数据已保存到: {output_path}")
        else:
            print("错误: 没有匹配后的数据可保存!")


def main():
    """
    主函数：演示PSM的使用
    """
    print("="*70)
    print("倾向得分匹配（PSM）分析")
    print("="*70)
    
    # 读取数据
    print("\n读取数据...")
    data = pd.read_csv('staggered_did_data_with_outcomes.csv')
    
    # 只使用基线数据（2023年3月之前）进行匹配
    baseline_data = data[data['date'] < '2023-03-01'].copy()
    
    # 获取每个分行的基线特征（取平均值）
    branch_baseline = baseline_data.groupby('branch_id').agg({
        'group': 'first',
        'ln_customer_count': 'first',  # 控制变量，不随时间变化
        'ln_city_gdp': 'first',        # 控制变量，不随时间变化
        'outcome': 'mean',             # 基线结果变量
        'outcome1': 'mean',
        'outcome2': 'mean',
        'outcome3': 'mean',
        'outcome4': 'mean',
        'outcome5': 'mean'
    }).reset_index()
    
    print(f"\n基线数据:")
    print(f"  时间范围: 2022年3月 - 2023年2月")
    print(f"  分行数量: {len(branch_baseline)} 家")
    print(f"  处理组: {sum(branch_baseline['group']=='treated')} 家")
    print(f"  对照组: {sum(branch_baseline['group']=='control')} 家")
    
    # 创建PSM对象
    psm = PropensityScoreMatching(branch_baseline)
    
    # 定义用于匹配的协变量
    covariates = ['ln_customer_count', 'ln_city_gdp', 'outcome', 
                  'outcome1', 'outcome2', 'outcome3', 'outcome4', 'outcome5']
    
    # 步骤1: 估计倾向得分
    psm.estimate_propensity_score(covariates)
    
    # 步骤2: 执行匹配
    psm.perform_matching(method='nearest', ratio=1, caliper=0.05)
    
    # 步骤3: 平衡性检验
    balance_results = psm.assess_balance(covariates)
    
    # 绘制倾向得分分布图
    psm.plot_propensity_score_distribution('psm_distribution.png')
    
    # 保存匹配后的分行列表
    if psm.matched_data is not None:
        matched_branches = psm.matched_data['branch_id'].unique()
        
        # 从原始数据中提取匹配后的完整数据
        matched_full_data = data[data['branch_id'].isin(matched_branches)].copy()
        
        # 保存匹配后的完整数据
        matched_full_data.to_csv('staggered_did_data_matched.csv', 
                                  index=False, encoding='utf-8-sig')
        print(f"\n匹配后的完整数据已保存到: staggered_did_data_matched.csv")
        print(f"  包含观测数: {len(matched_full_data)}")
        print(f"  时间跨度: 2022年3月 - 2024年2月")
        
        # 保存匹配的分行列表
        matched_branch_list = pd.DataFrame({
            'branch_id': matched_branches,
            'group': [branch_baseline[branch_baseline['branch_id']==b]['group'].values[0] 
                     for b in matched_branches]
        })
        matched_branch_list.to_csv('matched_branch_list.csv', 
                                   index=False, encoding='utf-8-sig')
        print(f"  匹配的分行列表已保存到: matched_branch_list.csv")
    
    print("\n" + "="*70)
    print("PSM分析完成!")
    print("="*70)


if __name__ == "__main__":
    main()
