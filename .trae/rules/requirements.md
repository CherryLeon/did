# Python 依赖包规范

## 数据生成依赖

| 包名 | 版本要求 | 用途 |
|------|---------|------|
| pandas | ≥ 1.5.0 | 数据处理与分析 |
| numpy | ≥ 1.23.0 | 数值计算与随机数生成 |
| python-dateutil | ≥ 2.8.0 | 日期时间处理 |

## 实证分析依赖

| 包名 | 版本要求 | 用途 |
|------|---------|------|
| pandas | ≥ 1.5.0 | 数据处理与分析 |
| numpy | ≥ 1.23.0 | 数值计算 |
| matplotlib | ≥ 3.6.0 | 数据可视化与图表生成 |
| seaborn | ≥ 0.12.0 | 统计图表绘制 |
| statsmodels | ≥ 0.13.0 | 回归分析与统计检验 |
| scikit-learn | ≥ 1.2.0 | 机器学习与PSM匹配 |
| scipy | ≥ 1.9.0 | 科学计算与统计分析 |

## 安装命令

```bash
pip install pandas>=1.5.0 numpy>=1.23.0 matplotlib>=3.6.0 seaborn>=0.12.0 statsmodels>=0.13.0 scikit-learn>=1.2.0 scipy>=1.9.0 python-dateutil>=2.8.0
```

## 快速安装

```bash
pip install -r requirements.txt
```

## 版本兼容性说明

- Python 版本要求：≥ 3.8
- Windows/Linux/macOS 均支持
- 建议使用虚拟环境管理依赖

## 包功能说明

### 数据生成
- `pandas`：面板数据创建、合并、导出
- `numpy`：随机数生成、数值计算
- `python-dateutil`：日期时间解析和计算

### 实证分析
- `pandas`：数据清洗、分组、描述性统计
- `numpy`：数组运算、随机数生成
- `matplotlib`：绘制回归系数图、效应趋势图
- `seaborn`：绘制分布图、热力图等统计图表
- `statsmodels`：DID回归、事件研究法、稳健性检验
- `scikit-learn`：PSM倾向得分匹配、Logit模型
- `scipy`：统计检验、分布分析