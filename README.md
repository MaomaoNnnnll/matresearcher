# MatResearcher

面向固态电池材料的文献调研多智能体系统。

## 架构

```
工作流编排 (LangGraph)
  ├── 任务规划 Agent      — 科学问题理解、任务拆解、检索策略生成
  ├── 文献检索 Agent       — Sciverse API 语义检索、查询扩展
  ├── [4a] 检索覆盖度核验  — 证据Agent检验检索覆盖度，不足则回退修正
  ├── 文献筛选 Agent       — 去重、Reranker 精排、阈值过滤
  ├── PDF 解析 Agent       — MinerU 全文结构化解析
  ├── 知识抽取 Agent       — LLM 抽取成分/结构/性能/工艺/实验条件
  ├── [7a] 数据质量核验    — 证据Agent检验单篇抽取合理性，异常分区存储
  ├── 跨文献融合 Agent     — 实体规范化、单位统一、冲突检测
  ├── Gap 识别 Agent      — 缺失检测、矛盾归纳、Gap 生成与评分
  ├── 证据核验 Agent       — 原文回溯、引用验证、事实核查
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
matresearcher survey "硫化物固态电解质室温离子导电率的提升策略"

# 评估
python scripts/evaluate.py --config config/workflow.yaml
```

## Docker 部署

```bash
docker-compose up -d
docker exec matresearcher matresearcher survey "你的科学问题"
```

## 项目结构

```
matresearcher/
├── config/              # YAML 配置与提示词模板
├── src/matresearcher/
│   ├── models/          # Pydantic 数据模型
│   ├── state.py         # 工作流状态定义
│   ├── agents/          # 8 个角色 Agent
│   ├── tools/           # Sciverse/MinerU/LLM/Embedding/Reranker
│   ├── knowledge_base/  # 向量库 + 关系库
│   ├── rules/           # 单位转换/化学式规范化/冲突检测
│   ├── workflow/        # LangGraph 工作流引擎
│   └── main.py          # CLI 入口
├── scripts/             # 运行与评估脚本
├── Dockerfile
└── docker-compose.yml
```

## 许可证

Apache-2.0
