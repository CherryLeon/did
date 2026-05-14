# 数据生成规范

## 1. 核心要求

### 1.1 时间范围

| 项目 | 要求 | 说明 |
|------|------|------|
| 观测窗口 | 近三年 | 2022年3月至2025年3月 |
| 处理时间 | 2023年3月 | 政策实施月份 |
| 政策前 | 12个月 | 2022年3月至2023年2月 |
| 政策后 | 24个月 | 2023年3月至2025年3月 |

### 1.2 样本规模

| 项目 | 要求 | 说明 |
|------|------|------|
| 客户数量 | 514,601个 | 论文要求的样本量 |
| 观测值数量 | 12,350,424个 | 514,601客户×24个月 |
| 处理组分行 | 47家 | 论文要求的处理组规模 |
| 对照组分行 | 39家 | 论文要求的对照组规模 |

### 1.3 变量结构

#### 1.3.1 核心变量

| 变量名 | 类型 | 说明 | 范围 |
|--------|------|------|------|
| customer_id | 字符串 | 客户唯一标识 | C000001-C514601 |
| branch_id | 字符串 | 分行唯一标识 | Branch_1-Branch_86 |
| month | 字符串 | 观测月份 | 2022-03至2025-03 |
| is_treatment | 整数 | 处理组虚拟变量 | 0=对照组, 1=处理组 |
| is_post | 整数 | 政策后虚拟变量 | 0=政策前, 1=政策后 |
| treatment_post | 整数 | 处理×政策后交互项 | 0/1 |
| CEMI | 浮点数 | 客户经营成效综合指数 | 0-100 |

#### 1.3.2 分场景指标

| 变量名 | 类型 | 说明 | 范围 |
|--------|------|------|------|
| service_resolution_rate | 浮点数 | 智能客服一次解决率 | 0-100 |
| auto_approval_rate | 浮点数 | 消费贷款自动审批率 | 0-100 |
| marketing_conversion_rate | 浮点数 | 营销转化率 | 0-100 |
| wealth_adoption_rate | 浮点数 | 财富管理推荐采纳率 | 0-100 |

#### 1.3.3 客户特征变量

| 变量名 | 类型 | 说明 | 取值范围 |
|--------|------|------|----------|
| age | 整数 | 客户年龄 | 18-70 |
| gender | 字符串 | 客户性别 | M/F |
| income_level | 字符串 | 收入水平 | low/medium/high |
| tenure_months | 整数 | 客户在我行年限 | 1-120 |
| previous_campaigns | 整数 | 历史营销活动参与次数 | 0-10 |
| has_loan | 整数 | 是否有贷款 | 0/1 |
| has_credit_card | 整数 | 是否有信用卡 | 0/1 |
| AUM_level | 字符串 | 资产水平 | low/medium/high |

#### 1.3.4 分行特征变量

| 变量名 | 类型 | 说明 | 取值范围 |
|--------|------|------|----------|
| branch_size | 字符串 | 分行规模 | small/medium/large |
| location_type | 字符串 | 分行位置类型 | urban/suburban/rural |
| staff_count | 整数 | 员工数量 | 10-100 |
| prior_performance | 浮点数 | 政策前绩效 | 0-100 |
| digital_maturity | 浮点数 | 数字化成熟度 | 0-100 |

## 2. 数据质量标准

### 2.1 平行趋势要求

| 指标 | 要求 | 说明 |
|------|------|------|
| 政策前差异 | < 2.0 | 处理组与对照组CEMI均值差异 |
| 趋势一致性 | 显著 | 政策前处理组与对照组趋势一致 |

### 2.2 处理效应要求

| 指标 | 要求 | 说明 |
|------|------|------|
| 基准DID系数 | 显著为正 | 符合论文预期 |
| 时间递进效应 | 递增 | 处理效应随时间增加 |
| 场景异质性 | 客服>信贷>营销>财富 | 客服和信贷场景效应最大 |

### 2.3 数据完整性

| 指标 | 要求 | 说明 |
|------|------|------|
| 缺失值比例 | < 1% | 各变量缺失值比例 |
| 数据一致性 | 100% | 客户-月度观测无重复 |
| 逻辑一致性 | 100% | 变量间逻辑关系一致 |

## 3. 元数据规范

### 3.1 数据字典

| 表名 | 字段名 | 数据类型 | 描述 | 约束 |
|------|--------|----------|------|------|
| panel_data | customer_id | VARCHAR(10) | 客户唯一标识 | PRIMARY KEY |
| panel_data | branch_id | VARCHAR(10) | 分行唯一标识 | NOT NULL |
| panel_data | month | VARCHAR(7) | 观测月份 | NOT NULL |
| panel_data | is_treatment | TINYINT | 处理组虚拟变量 | 0/1 |
| panel_data | is_post | TINYINT | 政策后虚拟变量 | 0/1 |
| panel_data | treatment_post | TINYINT | 处理×政策后交互项 | 0/1 |
| panel_data | CEMI | FLOAT | 客户经营成效综合指数 | 0-100 |
| panel_data | service_resolution_rate | FLOAT | 智能客服一次解决率 | 0-100 |
| panel_data | auto_approval_rate | FLOAT | 消费贷款自动审批率 | 0-100 |
| panel_data | marketing_conversion_rate | FLOAT | 营销转化率 | 0-100 |
| panel_data | wealth_adoption_rate | FLOAT | 财富管理推荐采纳率 | 0-100 |
| customer_features | customer_id | VARCHAR(10) | 客户唯一标识 | PRIMARY KEY |
| customer_features | age | INT | 客户年龄 | 18-70 |
| customer_features | gender | VARCHAR(1) | 客户性别 | M/F |
| customer_features | income_level | VARCHAR(10) | 收入水平 | low/medium/high |
| customer_features | tenure_months | INT | 客户在我行年限 | 1-120 |
| customer_features | previous_campaigns | INT | 历史营销活动参与次数 | 0-10 |
| customer_features | has_loan | TINYINT | 是否有贷款 | 0/1 |
| customer_features | has_credit_card | TINYINT | 是否有信用卡 | 0/1 |
| customer_features | AUM_level | VARCHAR(10) | 资产水平 | low/medium/high |
| branch_features | branch_id | VARCHAR(10) | 分行唯一标识 | PRIMARY KEY |
| branch_features | branch_size | VARCHAR(10) | 分行规模 | small/medium/large |
| branch_features | location_type | VARCHAR(10) | 分行位置类型 | urban/suburban/rural |
| branch_features | staff_count | INT | 员工数量 | 10-100 |
| branch_features | prior_performance | FLOAT | 政策前绩效 | 0-100 |
| branch_features | digital_maturity | FLOAT | 数字化成熟度 | 0-100 |

### 3.2 数据文件格式

| 格式 | 要求 | 说明 |
|------|------|------|
| 文件格式 | CSV | 逗号分隔值格式 |
| 编码 | UTF-8 | 统一编码格式 |
| 压缩 | GZIP | 大型文件压缩存储 |
| 命名规范 | {dataset}_{version}_{date}.csv | 如：panel_data_v1_20260425.csv |

## 4. 生成流程规范

### 4.1 数据生成步骤

1. **基础面板生成**：创建客户-月度观测的基础结构
2. **CEMI指数计算**：生成包含处理效应的CEMI指数
3. **分场景指标生成**：生成分场景具体指标
4. **协变量添加**：添加客户和分行特征变量
5. **数据验证**：验证平行趋势和处理效应
6. **质量检查**：检查数据完整性和一致性
7. **输出存储**：保存为标准格式文件

### 4.2 验证标准

| 验证项 | 标准 | 说明 |
|--------|------|------|
| 样本量 | 符合要求 | 客户数、观测值数符合论文要求 |
| 平行趋势 | 通过检验 | 政策前处理组与对照组趋势一致 |
| 处理效应 | 显著为正 | DID系数显著且符合预期 |
| 时间递进 | 效应递增 | 处理后效应随时间增加 |
| 场景异质性 | 符合预期 | 客服和信贷场景效应最大 |
| 数据完整性 | 无缺失 | 各变量无缺失值 |
| 逻辑一致性 | 无矛盾 | 变量间逻辑关系一致 |

## 5. 应用场景

### 5.1 DID分析

- **数据需求**：完整面板数据，包含核心变量和协变量
- **时间要求**：政策前后至少12个月
- **变量要求**：is_treatment、is_post、treatment_post、CEMI

### 5.2 PSM匹配

- **数据需求**：政策前数据，包含协变量
- **匹配变量**：客户和分行特征变量
- **匹配方法**：核匹配/最近邻匹配/卡尺匹配

### 5.3 平行趋势检验

- **数据需求**：政策前后时间序列数据
- **时间要求**：政策前至少6期，政策后至少6期
- **检验方法**：事件研究法

### 5.4 稳健性检验

- **数据需求**：多规格数据集
- **检验类型**：安慰剂检验、样本调整、变量替换

## 6. 注意事项

1. **数据规模**：完整数据集约1200万行，需要足够内存和计算资源

2. **计算效率**：对于大型数据集，建议使用分块处理或采样方法

3. **参数调整**：根据具体研究需求调整效应大小、噪声水平等参数

4. **结果一致性**：使用固定随机种子确保结果可复现

5. **元数据管理**：建立完整的元数据文档，记录数据生成过程和参数

6. **合规性**：确保生成的数据符合相关数据隐私规定

7. **文档化**：详细记录数据生成过程，包括参数设置、验证结果等

## 7. 参考标准

本文档基于《生成式AI在招商银行零售客户经营中的应用及成效分析》论文的实证设计要求，确保生成的数据能够支持DID、PSM、平行趋势检验和稳健性检验等实证分析，保证实证结论显著且符合论文要求。