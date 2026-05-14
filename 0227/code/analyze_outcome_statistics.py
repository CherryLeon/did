import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def analyze_outcome_statistics(data_path, output_image='outcome_statistics.png'):
    """
    分析五个结果变量的统计趋势并保存图片
    
    参数:
        data_path: 数据文件路径
        output_image: 输出图片路径
    """
    # 读取数据
    print(f"读取数据: {data_path}")
    data = pd.read_csv(data_path)
    
    # 确保日期列是datetime类型
    data['date'] = pd.to_datetime(data['date'])
    
    # 定义结果变量
    outcome_vars = ['outcome1', 'outcome2', 'outcome3', 'outcome4', 'outcome5']
    
    # 计算统计量
    print("\n计算统计量...")
    statistics = {}
    
    for var in outcome_vars:
        var_data = data[var]
        statistics[var] = {
            'mean': var_data.mean(),
            'std': var_data.std(),
            'variance': var_data.var(),
            'min': var_data.min(),
            'max': var_data.max(),
            'median': var_data.median(),
            'skewness': stats.skew(var_data),
            'kurtosis': stats.kurtosis(var_data)
        }
    
    # 创建统计量表格
    stats_df = pd.DataFrame(statistics).T
    print("\n统计量汇总:")
    print(stats_df.round(4))
    
    # 计算按月份的统计趋势
    print("\n计算时间趋势...")
    monthly_stats = data.groupby(['year', 'month'])[outcome_vars].agg(['mean', 'std', 'var'])
    monthly_stats = monthly_stats.reset_index()
    monthly_stats['date'] = pd.to_datetime(
        monthly_stats['year'].astype(str) + '-' + 
        monthly_stats['month'].astype(str) + '-01'
    )
    
    # 计算按组别的统计量
    print("\n计算组别统计...")
    group_stats = data.groupby('group')[outcome_vars].agg(['mean', 'std', 'var'])
    print("\n按组别统计:")
    print(group_stats.round(4))
    
    # 创建图形
    print("\n生成图形...")
    fig = plt.figure(figsize=(20, 12))
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 1. 统计量柱状图
    ax1 = plt.subplot(2, 3, 1)
    stats_for_plot = stats_df[['mean', 'std', 'variance', 'min', 'max', 'median']]
    stats_for_plot.plot(kind='bar', ax=ax1, rot=45)
    ax1.set_title('五个结果变量的统计量对比', fontsize=12, fontweight='bold')
    ax1.set_xlabel('变量', fontsize=10)
    ax1.set_ylabel('统计量值', fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. 均值时间趋势
    ax2 = plt.subplot(2, 3, 2)
    for var in outcome_vars:
        ax2.plot(monthly_stats['date'], monthly_stats[(var, 'mean')], 
                 label=var, linewidth=2, marker='o', markersize=3)
    ax2.set_title('均值随时间变化趋势', fontsize=12, fontweight='bold')
    ax2.set_xlabel('时间', fontsize=10)
    ax2.set_ylabel('均值', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. 标准差时间趋势
    ax3 = plt.subplot(2, 3, 3)
    for var in outcome_vars:
        ax3.plot(monthly_stats['date'], monthly_stats[(var, 'std')], 
                 label=var, linewidth=2, marker='s', markersize=3)
    ax3.set_title('标准差随时间变化趋势', fontsize=12, fontweight='bold')
    ax3.set_xlabel('时间', fontsize=10)
    ax3.set_ylabel('标准差', fontsize=10)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. 五个变量随时间变化的趋势图（随机选择treat组的一个对象）
    ax4 = plt.subplot(2, 3, 4)
    
    # 随机选择1个treat组的分行
    treated_branches = data[data['group'] == 'treated']['branch_id'].unique()
    
    np.random.seed(42)
    selected_branch = np.random.choice(treated_branches, 1)[0]
    
    # 获取该分行的数据
    branch_data = data[data['branch_id'] == selected_branch].sort_values('date')
    
    # 绘制五个outcome变量和合成的outcome变量
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    labels = ['outcome1', 'outcome2', 'outcome3', 'outcome4', 'outcome5', 'outcome (合成)']
    
    for i, var in enumerate(['outcome1', 'outcome2', 'outcome3', 'outcome4', 'outcome5', 'outcome']):
        ax4.plot(branch_data['date'], branch_data[var], 
                 label=labels[i], linewidth=2, alpha=0.8, 
                 color=colors[i])
    
    ax4.set_title(f'五个变量随时间变化的趋势（treat组: {selected_branch}）', fontsize=12, fontweight='bold')
    ax4.set_xlabel('时间', fontsize=10)
    ax4.set_ylabel('值', fontsize=10)
    ax4.legend(fontsize=9, loc='best')
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    
    # 5. 箱线图对比
    ax5 = plt.subplot(2, 3, 5)
    data_melted = data.melt(value_vars=outcome_vars, var_name='variable', value_name='value')
    sns.boxplot(data=data_melted, x='variable', y='value', ax=ax5)
    ax5.set_title('五个变量的箱线图对比', fontsize=12, fontweight='bold')
    ax5.set_xlabel('变量', fontsize=10)
    ax5.set_ylabel('值', fontsize=10)
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. 密度图对比
    ax6 = plt.subplot(2, 3, 6)
    for var in outcome_vars:
        sns.kdeplot(data[var], ax=ax6, label=var, linewidth=2)
    ax6.set_title('五个变量的密度分布', fontsize=12, fontweight='bold')
    ax6.set_xlabel('值', fontsize=10)
    ax6.set_ylabel('密度', fontsize=10)
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    print(f"\n图形已保存到: {output_image}")
    plt.close()
    
    # 保存统计量到CSV
    stats_output = output_image.replace('.png', '_statistics.csv')
    stats_df.to_csv(stats_output, encoding='utf-8-sig')
    print(f"统计量已保存到: {stats_output}")
    
    return stats_df, monthly_stats, group_stats

if __name__ == "__main__":
    analyze_outcome_statistics(
        data_path='staggered_did_data_with_outcomes.csv',
        output_image='outcome_statistics.png'
    )
