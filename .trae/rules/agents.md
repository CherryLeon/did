# 角色编排 — 银行经营管理实证分析体系

## 可用角色（斜杠命令）

| 命令 | 角色 | 触发场景 | 模型 |
|------|-----------|----------|------|
| `/empirical-research` | 实证研究员 | DID/PSM/平行趋势/稳健性检验设计与实施 | opus |
| `/data-generator` | 数据生成专家 | 基于论文背景和实证要求生成DID/PSM分析所需数据 | opus |
| `/thesis-review` | 金融专业硕士导师 | 评估论文是否满足硕士论文学术规范要求，点评结果以版本号格式输出为comments_vX.md | opus |
| `/thesis-revision` | 论文修改专家 | 根据评审意见修改论文、代码与数据，严格按照 `thesis-revision.md` 中的工作流规范执行。涉及文件：① `论文总结-DS模拟结果.md`（论文内容）② `test_data_generator.py`（数据生成代码）③ `empirical_analysis.py`（实证分析代码） | opus |

## 保留的通用角色

| 命令 | 用途 |
|------|------|
| `/plan` | 复杂任务的实施规划 |
| `/python-review` | Python 代码质量审查 |
| `/learn` | 从会话中提取可复用模式 |
| `/checkpoint` | 保存当前进度 |
| `/tdd` | 测试驱动开发 |
| `/update-docs` | 更新文档 |

## 角色使用原则

### 策略研究类任务的角色选择

```
用户说「设计DID模型」     → /empirical-research
用户说「做PSM匹配」      → /empirical-research
用户说「检验平行趋势」    → /empirical-research
用户说「稳健性检验」      → /empirical-research
用户说「生成实证数据」    → /data-generator
用户说「创建面板数据」    → /data-generator
用户说「生成CEMI指数」    → /data-generator
用户说「评估论文」        → /thesis-review
用户说「审查论文」        → /thesis-review
用户说「是否满足硕士论文」 → /thesis-review
用户说「修改论文」        → /thesis-revision
用户说「解决评审问题」     → /thesis-revision
```

### 多角色协作场景

**场景 1：学术实证研究**
1. `/data-generator` — 生成符合论文要求的实证数据
2. `/empirical-research` — 设计 DID 准自然实验
3. `/empirical-research` — 执行 PSM 匹配和平衡性检验
4. `/empirical-research` — 平行趋势检验和事件研究
5. `/empirical-research` — 稳健性检验体系

**场景 2：数据生成与验证**
1. `/data-generator` — 生成基础面板数据
2. `/data-generator` — 验证数据质量和平行趋势
3. `/empirical-research` — 初步分析验证处理效应
4. `/data-generator` — 根据分析结果调整数据参数
5. `/empirical-research` — 最终实证分析

## 并行执行

对独立的分析任务，始终使用并行任务执行：
```
# 好：并行执行不同的稳健性检验
同时启动 3 个 /empirical-research 执行不同的稳健性检验

# 好：并行分析不同子样本
同时启动 /empirical-research 分析不同行业或地区的子样本
```
