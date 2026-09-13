# MatResearcher — 智能体架构说明

面向**固态电池材料**的「文献驱动科学发现」多智能体系统。给定一句自然语言研究问题，
系统通过 12 个核心步骤 + 2 个可选扩展节点的 LangGraph 工作流，自动完成
**任务规划 → 文献检索 → 解析 → 知识抽取/融合 → Gap 识别 → 证据核验 → 报告生成 → 事实核查闭环**，
最终产出带「构效关系分析」「参考文献清单」「证据溯源清单」的文献调研报告，或参赛方案文档。

> 本文档聚焦**智能体（Agent）架构**与数据流，是对 `README.md` 的架构补充版本。
> 安装/环境变量等快速上手仍以下方「快速开始」为准。

---

## 1. 系统总览

| 维度 | 说明 |
|------|------|
| 编排框架 | [LangGraph](https://github.com/langchain-ai/langgraph) `StateGraph`（`WorkflowState` 共享状态在节点间流转） |
| 核心步骤 | 12 步线性主链 |
| 可选扩展 | Step 8.5 构效关系分析、Step 10.5 假设外部/离线交叉核验（均按 `config/workflow.yaml` 开关挂载） |
| 条件分支 | 2 个：`coverage_check`（不足则回退补检索）、`fact_check`（有问题时回到报告生成重写） |
| 状态管理 | `WorkflowState`（`TypedDict, total=False`），所有通道显式声明 |
| 缓存/续跑 | `RunCache`（节点 pkl + 每篇文献 pkl + `MANIFEST.json`），支持 `--resume` |
| 双输出模式 | `output_mode: survey`（文献调研报告）/ `submission`（参赛方案文档） |

---

## 2. 智能体工作流拓扑

```
                         ┌─────────────────────────────┐
                         │   raw_question (用户输入)     │
                         └───────────────┬─────────────┘
                                         ▼
                            (1) task_planning ───────────── 子任务 + 检索策略
                                         │
                                         ▼
                            (2) literature_search ──────── Sciverse 检索 + DOI 补全
                                         │
                                         ▼
                            (3) coverage_check ─────────── 检索覆盖度核验
                                  │            │
              retry_search ◄──────┘            └─────► continue
                    │                                      │
                    └──────── (回到 2) ◄───────────────────┘
                                         │
                                         ▼
                            (4) llm_prefilter ─────────── 三分类预筛 (relevant/partial/irrelevant)
                                         │
                                         ▼
                            (5) literature_filter ──────── 去重 + Reranker 精排 + 阈值过滤
                                         │
                                         ▼
                            (6) pdf_parsing ───────────── MinerU 结构化解析 (表格/图注)
                                         │
                                         ▼
                       (7) knowledge_extraction ──────── 知识抽取 + 数据质量核验 (PASS/REVIEW/FAIL)
                                         │
                                         ▼
                         (8) knowledge_fusion ────────── 规范化 + 双库存储 + 融合 + 冲突检测
                                  │            │
              [structure_property   │            └───── 直接
               enabled=true]        │                    │
                    │               │                    │
                    ▼               │                    │
            (8.5) structure_property ┘──► (9) gap_generation ◄┘
                                         │
                                         ▼
                       (10) evidence_verification ───── 证据溯源核验 (回溯原文段落)
                                  │            │
              [hypothesis_crosscheck            └───── 直接
               enabled=true]                    │
                    │                           │
                    ▼                           │
          (10.5) hypothesis_crosscheck ────────┴──► (11) report_generation
                                                                 │
                                                                 ▼
                                                       (12) fact_check ─ 事实核查
                                                              │        │
                                          revise ◄────────────┘        └──► done (END)
                                             │
                                             └──────── (回到 11) ◄─────────┘
```

**关键路径说明**
- `coverage_check` 失败且重试未用尽 → 路由回 `literature_search`，并刷新检索策略（补充语义查询 / 放宽年份）。
- `fact_check` 发现需修订且未达上限 → 路由回 `report_generation` 做**就地重写**（闭环），最多 `max_revisions` 轮。
- `structure_property`（Step 8.5）、`hypothesis_crosscheck`（Step 10.5）仅在对应配置开关开启时装配进图。

---

## 3. 智能体（Agent）清单

每个节点由 `workflow/nodes.py` 的 `create_all_nodes()` 装配一个共享 agent 实例；所有 agent 复用同一 `LLMClient`、
同一 `TokenCounter`、同一套工具与知识库实例。

| Step | 节点名 | Agent 类 | 核心职责 |
|------|--------|----------|----------|
| 1 | `task_planning` | `TaskPlanningAgent` | 解析科学问题、分解为子任务、生成检索策略（`semantic_queries` / `primary_keywords` / `filters`） |
| 2 | `literature_search` | `LiteratureSearchAgent` | Sciverse 语义检索 + 查询扩展；**DOI 三层补全**（Crossref → Sciverse meta → doc_id 兜底链接）；按归一化标题去重，质量序：有 DOI > 来源优先级 > 相关度 |
| 3 | `coverage_check` | `EvidenceVerificationAgent.check_coverage` | 检索覆盖度核验（子任务覆盖、年份分布等），输出 `passed` + `issues`；不足触发补检索 |
| 4 | `llm_prefilter` | `LLMPrefilterAgent` | 三分类 `relevant/partial/irrelevant`，降低后续精排负载 |
| 5 | `literature_filter` | `LiteratureFilterAgent` | 去重 + Reranker 精排（BAAI/bge-reranker-v2-m3）+ 阈值过滤；结果写关系库 |
| 6 | `pdf_parsing` | `PDFParsingAgent` | MinerU 全文结构化解析（表格/图注）；Sciverse OA fallback 取 PDF（Unpaywall 用 `UNPAYWALL_EMAIL`）；表格结构化入库 |
| 7 | `knowledge_extraction` | `KnowledgeExtractionAgent` | LLM 抽取 成分/结构/性能/工艺/实验条件；内嵌数据质量核验 **PASS/REVIEW/FAIL** 三分法，异常分区至 `anomaly_records`（`quality_report.json`） |
| 8 | `knowledge_fusion` | `KnowledgeFusionAgent` | 实体规范化（单位统一、化学式规范化）、向量库+关系库双存储、跨文献融合、冲突检测；产出 `normalized_records` / `fused_table` / `conflicts` |
| 8.5 | `structure_property` | `rules.structure_property` | 定量构效关系分析（归一化记录的属性相关性统计），渲染为 `## 6. 构效关系定量分析` 章节 |
| 9 | `gap_generation` | `GapIdentificationAgent` | 缺失检测、矛盾归纳、Gap 生成与评分（`ResearchGap.score.total_score`），产出排序后的 `scored_gaps` |
| 10 | `evidence_verification` | `EvidenceVerificationAgent.verify_gaps` | 证据溯源核验：将每条 Gap 主张回溯到原文段落（doc_id/DOI/url + 引用验证），标记 `verification_status` |
| 10.5 | `hypothesis_crosscheck` | `tools.materials_project.cross_check_gap` | 用 Materials Project API + Sci-Base 离线语料交叉验证 Gap 假设，给出 `corroborated/refuted/unknown`  verdict |
| 11 | `report_generation` | `ReportGenerationAgent` | 结构化调研报告；区分**文献事实 / 跨文献推论 / 待验证假设**；追加构效关系、假设交叉核验、证据溯源清单；双输出（见 §6） |
| 12 | `fact_check` | `EvidenceVerificationAgent.fact_check_report` | 最终事实核查：结构化检查 + 修订建议；`needs_revision` 触发回到 Step 11 重写（闭环） |

> **闭环重写的保证**：Step 11 返回的 `draft_report` 始终是**调研报告本体**，`submission_report` 才是参赛方案（独立变量，不原地改写 `draft`）。
> 这样事实核查闭环始终修订「调研报告」，且 `survey_report.md` 不被 `submission` 文档覆盖（修复自 2026-09-11 的覆盖 bug）。

---

## 4. 工具层（`src/matresearcher/tools/`）

统一通过 `nodes.create_all_nodes` 实例化并注入各 agent，避免重复连接：

| 工具 | 文件 | 作用 |
|------|------|------|
| `LLMClient` | `llm.py` | 统一 LLM 调用；共享 `TokenCounter` 做分步骤 token/成本统计 |
| `SciverseClient` | `sciverse.py` | Sciverse 语义检索 API + `/meta-search`（DOI 补全次级源） |
| `MinerUParser` | `mineru.py` | MinerU 全文结构化解析（表格/图注），支持 API / CLI 模式 |
| `EmbeddingModel` | `embedding.py` | 向量化（默认 `BAAI/bge-m3`） |
| `RerankerModel` | `reranker.py` | 重排序（默认 `BAAI/bge-reranker-v2-m3`） |
| `MaterialsProjectClient` / `SciBaseFallback` | `materials_project.py` | 外部（MP API）/ 离线（Sci-Base）假设交叉核验；缺 key 时降级离线 |
| `DOIEnricher` | `doi_enrichment.py` | 标题→DOI 三层补全（Crossref 主查 `≥0.72` 相似度护栏 + Sciverse meta 次查 + doc_id 兜底 landing URL） |

---

## 5. 知识库与规则引擎

### 5.1 知识库（`src/matresearcher/knowledge_base/`）
- **`VectorStore`**（`vector_store.py`）：基于 [Chroma](https://www.trychroma.com/)，承载归一化记录的向量检索。
- **`RelationalStore`**（`relational.py`）：基于 SQLAlchemy + SQLite，结构化存储文献元数据、解析表格、融合表，供评分/追溯查询。

### 5.2 规则引擎（`src/matresearcher/rules/`）
- `unit_conversion.py` — 物理量单位统一
- `formula_normalizer.py` — 化学式规范化（实体对齐前提）
- `material_config.py` — 材料领域配置
- `structure_property.py` — 定量构效关系分析（`analyze_structure_property` / `format_structure_property_md`）
- `conflict_detector.py` — 跨文献数据冲突检测

### 5.3 数据模型（`src/matresearcher/models/`）
- `literature.py` — `Literature` / `LiteratureMetadata`（含 `doi`、`url` 兜底、`doc_id`）
- `knowledge.py` — `KnowledgeRecord` / `NormalizedRecord` / `FusedKnowledgeTable`
- `gap.py` — `ResearchGap` / `ConflictItem` / `MissingItem`

---

## 6. 状态与双输出数据流（`state.py`）

`WorkflowState`（`TypedDict, total=False`）是贯穿全流程的共享状态，所有通道必须显式声明，
否则 LangGraph 会**静默丢弃**未声明键（历史 bug：`_structure_property_md`、`fact_check_*` 曾因漏声明而丢失）。

### 双输出模式（`config.workflow.yaml → agents.report_generation.output_mode`）
- **`survey`**：`survey_report.md` = 文献调研报告本体（含 `## 6. 构效关系定量分析` → `## 7. 方法局限性` → `## 参考文献清单` → `## 附录 A. 证据溯源清单`）。此时 `submission_report` 为空。
- **`submission`**：`survey_report.md` 仍为上述调研报告（交付物之一）；`report.md` 由 `submission_report`（参赛方案文档，六章模板）产出。

> 报告尾部章节由 `_assemble_survey_report()` **按规范顺序重排**，且对修订轮幂等（不重复、不乱序）。

---

## 7. 闭环与重试机制

| 机制 | 触发点 | 行为 | 上限 |
|------|--------|------|------|
| 检索覆盖回退 | `coverage_check` 未通过 | 路由回 `literature_search` 并重刷策略，失效下游缓存 | `workflow.coverage_check.max_retries`（默认 2） |
| 事实核查重写 | `fact_check` 标 `needs_revision` | 路由回 `report_generation` 做就地重写 | `agents.evidence_verification.max_revisions` |
| 循环节点缓存豁免 | `report_generation` / `fact_check` | 这两个节点**不命中** RunCache，确保每轮重跑而非重放旧 patch | — |

`recursion_limit=150` 作为兜底，避免任何意外环空转。

---

## 8. 配置（`config/workflow.yaml` + 环境变量）

- **全局开关**：`workflow.structure_property_analysis.enabled`、`materials_project.enabled`、`workflow.data_quality_check.enabled`、`workflow.coverage_check.max_retries`。
- **Agent 配置**：`agents.default_llm`（api_base/api_key/model 支持 `${ENV_VAR}` 占位符）+ 各步骤子段（如 `literature_search.doi_enrichment`、`pdf_parsing.unpaywall_email=${UNPAYWALL_EMAIL}`、`report_generation.output_mode`）。
- **工具端点**：`sciverse_api_base` / `sciverse_api_key`、`mineru_api_url`、`embedding_model`、`reranker_model`、`vector_persist_dir`、`database_url`。
- **环境变量加载**：`env_loader.load_env_files()` 依次加载项目 `.env`（非敏感）与 `~/.matresearcher/secrets.env`（密钥），`engine._load_config` 再解析 YAML 中的 `${VAR}`。
  - 个人邮箱 `UNPAYWALL_EMAIL` 建议写入用户级 `secrets.env`，**不要硬编码进仓库**。

---

## 9. 快速开始

### 9.1 安装依赖
> ⚠️ 推荐使用**已建好的虚拟环境**直接运行，避免重复安装一大堆依赖：
```bash
cd matresearcher
# 仅在首次（或依赖变更）时安装一次：
.venv/Scripts/python.exe -m pip install -e .
```
（若改用 `uv run`，uv 会在首次/`uv.lock` 变更时重新解析并安装全部依赖，可能触发"先装一堆包"的现象。）

### 9.2 配置环境变量
```bash
cp .env.example .env
# 编辑 .env：填入 SCIVERSE_API_KEY、LLM_API_KEY、LLM_API_BASE、LLM_MODEL 等
# 可选：在 ~/.matresearcher/secrets.env 写入 UNPAYWALL_EMAIL、MATERIALS_PROJECT_API_KEY
```

### 9.3 运行文献调研
```bash
# 方式 A：已建好的 venv（推荐）
.venv/Scripts/python.exe -m matresearcher.main survey "硫化物固态电解质室温离子导电率的提升策略"

# 方式 B：控制台脚本（需先 pip install -e .）
matresearcher survey "..."

# 从失败/中断处续跑（复用已完成节点的缓存，不重复消耗 token）
matresearcher survey "..." --resume 2026-09-13_181425
```
产出位于 `outputs/logs/<run_id>/`：`survey_report.md`（调研报告）、`report.md`（submission 模式下的参赛方案）、`token_usage.json`、`quality_report.json` 等。

### 9.4 评估
```bash
python scripts/evaluate.py --config config/workflow.yaml
# 或
matresearcher evaluate --config config/workflow.yaml
```

---

## 10. 项目结构

```
matresearcher/
├── config/workflow.yaml        # 工作流 + Agent + 工具配置（${ENV_VAR} 占位符）
├── src/matresearcher/
│   ├── main.py                 # Typer CLI 入口（survey / serve / evaluate）
│   ├── state.py                # WorkflowState（共享状态 schema）
│   ├── models/                 # Pydantic 数据模型（literature/knowledge/gap）
│   ├── agents/                 # 角色 Agent（§3 表格）
│   │   ├── base.py             # Agent 基类（load_config / 日志 / LLM 调用）
│   │   └── ...
│   ├── tools/                  # LLM / Sciverse / MinerU / Embedding / Reranker / MP / DOI 补全
│   ├── knowledge_base/         # 向量库 (Chroma) + 关系库 (SQLAlchemy/SQLite)
│   ├── rules/                  # 单位转换 / 化学式规范化 / 构效关系 / 冲突检测
│   ├── workflow/               # LangGraph 编排
│   │   ├── engine.py           # MatResearcherWorkflow（compile/run/resume）
│   │   ├── nodes.py            # create_all_nodes 装配所有 Agent + 共享工具
│   │   └── run_cache.py        # 节点 / 每篇文献 / MANIFEST 缓存
│   ├── evaluation/             # 评估基线（baselines.py）
│   ├── env_loader.py           # .env / secrets.env 加载
│   └── config_check.py         # 配置静态检查
├── tests/                      # pytest 套件（asyncio_mode=auto）
├── scripts/                    # 运行 / 评估 / 质量门禁脚本
└── outputs/logs/<run_id>/      # 每次运行的报告与中间产物
```

---

## 11. 测试

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
```
覆盖：DOI 三层补全、事实核查闭环（含就地重写）、报告双输出分离、`survey_report.md` 章节顺序（构效关系→方法局限→参考文献→附录A）与修订幂等、检索覆盖回退等。

---

## 12. 许可证

Apache-2.0
