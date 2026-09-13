# MatResearcher

面向固态电池材料的文献调研多智能体系统。

## 架构

```
工作流编排 (LangGraph, 12 节点 + 2 个条件分支)
  ├── 任务规划 Agent      — 科学问题理解、任务拆解、检索策略生成
  ├── 文献检索 Agent       — Sciverse API 语义检索、查询扩展
  ├── 检索覆盖度核验  — 证据Agent检验检索覆盖度，不足则回退修正
  ├── LLM 三分类预筛 Agent — relevant/partial/irrelevant 三分类，降低精排负载
  ├── 文献筛选 Agent       — 去重、Reranker 精排、阈值过滤
  ├── PDF 解析 Agent       — MinerU 全文结构化解析（含表格/图注结构化入库）
  ├── 知识抽取 Agent       — LLM 抽取成分/结构/性能/工艺/实验条件
  ├── 数据质量核验    — 证据Agent检验单篇抽取合理性，异常分区存储
  ├── 跨文献融合 Agent     — 实体规范化、单位统一、冲突检测
  ├── Gap 识别 Agent      — 缺失检测、矛盾归纳、Gap 生成与评分
  ├── 证据核验 Agent       — 原文回溯、引用验证、事实核查
  ├── 假设交叉核验 (MP/Sci-Base) — 可选：Gap 假设外部/离线交叉验证（需配置）
  ├── 构效关系分析 Agent    — 可选：归一化记录的定量结构-性能相关性（Step 8.5）
  └── 报告生成 Agent       — 结构化调研报告（区分事实/推论/假设）
```

## 快速开始

```bash
# 安装
pip install -e .

# 配置环境变量
cp .env.example .env
# 编辑 .env，填入 Sciverse API Key、LLM 配置等

# 运行文献调研
cd src
python -m matresearcher.main survey "你的研究话题"

# 评估
python scripts/evaluate.py --config config/workflow.yaml
```


## 项目结构

```
matresearcher/
├── config/              # YAML 配置与提示词模板
├── src/matresearcher/
│   ├── models/          # Pydantic 数据模型
│   ├── state.py         # 工作流状态定义
│   ├── agents/          # 角色 Agent（任务规划/检索/预筛/筛选/解析/抽取/融合/Gap/核验/报告）
│   ├── tools/           # Sciverse/MinerU/LLM/Embedding/Reranker
│   ├── knowledge_base/  # 向量库 + 关系库
│   ├── rules/           # 单位转换/化学式规范化/冲突检测
│   ├── workflow/        # LangGraph 工作流引擎
│   └── main.py          # CLI 入口
└── scripts/             # 运行与评估脚本
```

赛题背景：AI for Research 赛道 · 方向三，材料科学文献驱动的科学发现智能体

## 许可证

Apache-2.0
