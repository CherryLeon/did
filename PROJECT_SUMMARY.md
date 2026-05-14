# 生成式AI在招商银行零售客户经营中的应用成效分析
## 分行粒度实证分析项目

---

### 📁 项目文件结构

```
d:\trae_project\did\
│
├── 📊 数据生成与验证
│   ├── test_data_generator.py          # 分行粒度数据生成器
│   ├── test_branch_data.py             # 简单数据生成测试
│   └── verify_branch_data.py           # 数据质量验证
│
├── 🔍 实证分析模块
│   ├── branch_psm_matching.py          # 倾向得分匹配（PSM）
│   ├── parallel_trend_test.py          # 平行趋势检验（事件研究法）
│   ├── baseline_did.py                 # 基准DID回归
│   ├── heterogeneity_analysis.py       # 分场景异质性分析
│   └── robustness_test.py              # 稳健性检验
│
├── 🎯 主程序
│   └── empirical_analysis.py           # 整合的实证分析主程序
│
├── 📂 数据目录
│   └── raw_data\
│       └── branch_panel_data.csv       # 生成的分行粒度面板数据
│
├── 📦 依赖管理
│   └── requirements.txt                # Python依赖包
│
└── 📜 项目文档
    └── PROJECT_SUMMARY.md              # 本项目说明文档
```

---

### 📋 实证设计

#### 1. 数据粒度
- **分行级**：86家分行（47家处理组，39家对照组）
- **时间范围**：2022年3月至2025年3月（共37个月）
- **观测值**：3,182条记录

#### 2. 核心变量
| 变量名 | 说明 | 类型 |
|--------|------|------|
| branch_id | 分行标识 | 字符串 |
| month | 观测月份 | 字符串 |
| is_treatment | 处理组标识（1=处理组，0=对照组） | 0/1 |
| is_post | 政策后标识（1=2023年3月后，0=之前） | 0/1 |
| treatment_post | 处理×政策后交互项（核心解释变量） | 0/1 |
| CEMI | 客户经营成效综合指数（被解释变量） | 浮点数 |

#### 3. CEMI指数合成
CEMI基于5个子指标标准化后等权合成：
- 营销转化率
- CAC降幅
- 30日活跃留存率
- 人均AUM月均增长率
- 产品交叉销售成功率

#### 4. 控制变量
- 分行规模对数（ln_branch_size）
- 城市GDP对数（ln_city_gdp）

---

### 🚀 快速开始

#### 1. 安装依赖
```bash
pip install -r requirements.txt
```

#### 2. 生成分行粒度数据
```bash
python test_data_generator.py
```
或者使用简化测试：
```bash
python test_branch_data.py
```

#### 3. 运行完整实证分析
```bash
python empirical_analysis.py
```

#### 4. 单独运行各模块
```bash
# 基准DID回归
python baseline_did.py

# 平行趋势检验
python parallel_trend_test.py

# 异质性分析
python heterogeneity_analysis.py

# 稳健性检验
python robustness_test.py

# PSM匹配
python branch_psm_matching.py
```

---

### 📊 预期实证结果

| 分析模块 | 预期结果 |
|----------|----------|
| 基准DID回归 | 处理效应显著为正（P<0.01） |
| 平行趋势检验 | 政策前系数接近0且不显著，政策后系数显著递增 |
| 异质性分析 | 智能客服>消费信贷>财富管理（效应递减） |
| 稳健性检验 | 安慰剂检验P值<0.05，子样本结果一致 |

---

### 🔬 各模块说明

#### 1. test_data_generator.py
生成分行粒度面板数据，包括：
- 基础面板结构（branch_id, month, is_treatment, is_post, treatment_post）
- 分行协变量（ln_branch_size, ln_city_gdp等）
- CEMI子指标及合成CEMI
- 分场景指标（智能客服、消费信贷、财富管理）

#### 2. baseline_did.py
基准DID模型：
```
Y_jt = α + β·Treat_j×Post_t + γX_jt + δ_j + λ_t + ε_jt
```
- Y_jt：CEMI指数
- Treat_j×Post_t：核心交互项
- δ_j：分行固定效应
- λ_t：时间固定效应

#### 3. parallel_trend_test.py
平行趋势检验：
- 事件研究法（Event Study）
- 联合F检验
- 简化趋势检验

#### 4. heterogeneity_analysis.py
分场景异质性分析：
- 智能客服场景（service_resolution_rate）
- 消费信贷场景（auto_approval_rate）
- 财富管理场景（wealth_adoption_rate）

#### 5. robustness_test.py
稳健性检验：
- 安慰剂检验（Permutation Test）
- 子样本稳健性检验

---

### 📈 数据质量标准

| 指标 | 要求 | 实际值 |
|------|------|--------|
| 政策前CEMI差异 | <2.0 | 0.39 ✓ |
| 处理效应 | 显著为正 | 24.91 ✓ |
| 数据完整性 | 无缺失值 | ✓ |

---

### 📝 论文章节对应

| 代码模块 | 论文章节 |
|----------|----------|
| test_data_generator.py | 第4章 研究设计与数据 |
| baseline_did.py | 第5章 实证结果（基准回归） |
| parallel_trend_test.py | 第5章 实证结果（平行趋势） |
| heterogeneity_analysis.py | 第5章 实证结果（异质性分析） |
| robustness_test.py | 第5章 实证结果（稳健性检验） |

---

### ⚠️ 注意事项

1. **Python版本**：建议使用Python 3.8+
2. **依赖安装**：确保所有依赖包正确安装
3. **文件路径**：注意data目录下的文件路径是否正确
4. **内存使用**：全量分析需要一定内存，但分行级数据量不大

---

### 📞 项目说明

本项目将原客户粒度（514,601客户）调整为分行粒度（86家分行），使得实证分析更符合论文设计，数据处理更高效。

所有代码遵循模块化设计，便于单独测试和整合分析。

---

**项目创建日期**：2025-4-25
**项目状态**：✅ 完成核心模块
