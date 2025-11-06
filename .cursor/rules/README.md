# LiveTalking Cursor 规则说明

本目录包含 LiveTalking 项目的 Cursor AI 规则文件，用于帮助 AI 助手更好地理解和协助项目开发。

## 规则文件列表

### 始终应用的规则

这些规则会自动应用到所有请求：

- **project-structure.mdc** - 项目结构和核心文件说明

### 按文件类型应用的规则

这些规则会根据文件类型自动应用：

- **python-style.mdc** (*.py) - Python 代码风格规范
- **frontend-guidelines.mdc** (web/*.html, web/*.js) - 前端开发指南

### 可按需获取的规则

这些规则可以通过描述手动获取：

- **model-integration.mdc** - 数字人模型集成和扩展指南
- **webrtc-guidelines.mdc** - WebRTC 实现和调试指南
- **asr-tts.mdc** - ASR 和 TTS 模块开发指南
- **llm-integration.mdc** - 大语言模型集成指南
- **deployment.mdc** - 部署和运维指南
- **avatar-management.mdc** - 数字人形象管理和制作指南
- **testing-debugging.mdc** - 测试和调试指南
- **security-best-practices.mdc** - 安全最佳实践和注意事项
- **contribution-guidelines.mdc** - 贡献指南和开发流程
- **performance-optimization.mdc** - 性能优化技巧和最佳实践
- **troubleshooting-faq.mdc** - 常见问题和故障排除

## 规则使用说明

### 对于开发者

这些规则文件主要供 Cursor AI 使用，但开发者也可以阅读这些文件来了解项目的最佳实践和开发规范。

### 对于 AI 助手

- 始终应用的规则会自动加载
- 按文件类型的规则会在编辑相应文件时自动加载
- 可按需获取的规则可以通过 `fetch_rules` 工具获取

## 规则文件格式

所有规则文件使用 Markdown 格式，带有 YAML frontmatter：

```markdown
---
alwaysApply: true  # 或 false
description: "规则描述"  # 用于手动获取
globs: "*.py,*.txt"  # 文件匹配模式
---

# 规则内容

规则的详细说明...
```

## 更新规则

如果需要更新或添加新规则：

1. 在 `.cursor/rules/` 目录下创建或编辑 `.mdc` 文件
2. 添加适当的 frontmatter 元数据
3. 使用 Markdown 格式编写规则内容
4. 使用 `[filename](mdc:filename)` 格式引用项目文件

## 规则覆盖范围

### 核心功能
- ✅ 项目结构和架构
- ✅ 代码风格和规范
- ✅ 模型集成
- ✅ WebRTC 实现
- ✅ ASR/TTS 集成
- ✅ LLM 集成

### 开发流程
- ✅ 测试和调试
- ✅ 性能优化
- ✅ 安全最佳实践
- ✅ 部署和运维

### 用户支持
- ✅ 常见问题解答
- ✅ 故障排除
- ✅ Avatar 管理

### 社区
- ✅ 贡献指南
- ✅ 开发流程

## 反馈和改进

如果您发现规则有误或需要改进，欢迎：

1. 提交 Issue
2. 提交 Pull Request
3. 在社区讨论

---

**注意**: 这些规则文件是为了辅助开发，不会影响项目的实际运行。

