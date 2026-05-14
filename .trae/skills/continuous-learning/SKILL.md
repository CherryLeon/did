---
name: continuous-learning
description: 自动从会话中提炼可复用模式，并保存为可复用技能。
---

# 持续学习（Continuous Learning）

在会话结束时自动评估并提炼可复用模式，保存为 learned skills。

## 工作方式

在每次会话结束时（Stop hook）运行：

1. 会话评估：检查会话长度是否足够（默认 10+ 条用户消息）
2. 模式识别：提炼可复用模式
3. 保存：写入 `skills/learned/`

## Configuration

编辑 `config.json`：

```json
{
  "min_session_length": 10,
  "extraction_threshold": "medium",
  "auto_approve": false,
  "learned_skills_path": "skills/learned/",
  "patterns_to_detect": [
    "error_resolution",
    "user_corrections",
    "workarounds",
    "debugging_techniques",
    "project_specific"
  ],
  "ignore_patterns": [
    "simple_typos",
    "one_time_fixes",
    "external_api_issues"
  ]
}
```

## 模式类型

| 模式 | 含义 |
|------|------|
| `error_resolution` | 错误定位与修复套路 |
| `user_corrections` | 用户纠错中提炼的项目偏好/约束 |
| `workarounds` | 框架/库的坑与规避方案 |
| `debugging_techniques` | 高效调试方法 |
| `project_specific` | 项目特定惯例 |

## Hook 配置

加入 `settings.json`：

```json
{
  "hooks": {
    "Stop": [{
      "matcher": "*",
      "hooks": [{
        "type": "command",
        "command": "skills/continuous-learning/evaluate-session.sh"
      }]
    }]
  }
}
```

## 为什么用 Stop hook

- 轻量：每次会话只跑一次
- 不影响交互：不会给每条消息增加开销
- 上下文完整：能读到整段会话

## 相关

- `/learn`：手动提炼

---
