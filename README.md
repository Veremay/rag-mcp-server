# Modular RAG MCP Server

> 一个可插拔、可观测的模块化 RAG (检索增强生成) 服务框架

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)

---

## 🔧 项目功能

本项目是一个**企业级智能问答与知识检索系统**，可应用于以下场景：

- **📖 文档问答**：支持 PDF、Markdown、代码文件等多格式文档的智能问答，快速从海量文档中提取精准答案
- **🔍 语义搜索**：基于混合检索技术，提供比传统关键词搜索更智能的语义理解能力
- **💡 知识库构建**：将企业内部文档、技术资料转化为可检索的知识库，提升团队协作效率
- **🤖 AI 助手集成**：通过 MCP (Model Context Protocol) 协议，可无缝对接 Claude、GitHub Copilot 等 AI 助手
- **🎯 个性化应用**：可扩展为客服机器人、技术文档助手、代码搜索引擎等垂直领域应用

> 💼 **面试利器**：本项目的模块化设计和完整实现，可直接作为简历项目展示，涵盖 RAG 技术栈的核心知识点，是大模型/AI 工程师面试的绝佳项目案例。

---

## 🎯 项目特点

### 1️⃣ **可插拔架构 (Pluggable Architecture)**
- **LLM 后端灵活切换**：支持 Azure OpenAI、OpenAI API、本地模型（Ollama/vLLM）等多种后端，通过配置文件一键切换，零代码修改
- **模型组件自由替换**：Embedding 模型、Rerank 模型、文档解析器、切分策略等核心组件均采用抽象接口设计，支持"乐高积木式"组合
- **检索策略可配置**：支持纯向量检索、BM25 关键词检索、混合检索（Hybrid Search）等多种模式动态配置

### 2️⃣ **全链路可观测 (Observable)**
- **结构化日志追踪**：每个模块输出详细的结构化日志，便于问题定位与性能分析
- **评测指标体系**：集成 Ragas/DeepEval 等 RAG 评测框架，量化检索质量与生成效果
- **监控与调试友好**：提供完整的请求链路追踪，支持实时性能监控

### 3️⃣ **MCP 协议集成 (Model Context Protocol)**
- **标准化接口**：完整实现 MCP 协议规范，可无缝对接 Claude Desktop、GitHub Copilot 等支持 MCP 的 AI 助手
- **开箱即用**：通过 MCP Server 暴露 RAG 能力，让 AI 助手直接调用项目的检索和问答功能，无需额外开发
- **工具化封装**：将 RAG 流程封装为 MCP Tools，AI 可以自主决策何时调用检索、如何组合多个工具完成复杂任务
- **上下文增强**：为 AI 对话提供实时的知识库支持，让通用大模型具备领域专业知识

### 4️⃣ **工程化 RAG 实践**
- **智能分块策略**：语义感知的文档切分，保留完整语义单元
- **混合检索 (Hybrid Search)**：BM25 + Dense Embedding 融合，平衡查准率与查全率
- **两段式精排**：粗排召回 → Rerank 精排，在性能与精度间取得最优平衡

---

## 📚 AI 驱动开发：让 AI 成为你的协作伙伴

### 💡 核心理念

> **"文档即规范，实现交给 AI"**

本项目采用创新的 **AI 协作开发模式**，让您专注于架构设计与业务逻辑，将代码实现高效委托给 AI：

#### ✨ 项目特色
- **完整的 Skills 体系**：通过精心设计的 Markdown 技能文件（`.trae/skills/`），AI 可以理解项目规范、遵循最佳实践，自动化完成代码实现
- **VibeCoding 实践**：掌握最新的 AI 协作开发技巧（VibeCoding），通过自然语言描述需求，让 AI 自动生成符合规范的代码
- **规范驱动开发**：`DEV_SPEC.md` 作为项目的"宪法"，定义架构、模块设计、技术选型等核心规范，AI 严格遵循文档完成编码
- **零背景快速上手**：即使您不熟悉 RAG 技术栈，只需理解文档、修改需求描述，AI 会自动将您的想法转化为生产级代码

#### 🚀 工作流程

```
1. 📝 理解文档 (DEV_SPEC.md)  → 掌握项目设计理念与技术架构
2. ✏️ 修改规范文档             → 根据需求调整设计方案或新增模块
3. 🤖 调用 Skills 交给 AI      → 使用 dev-workflow、implement 等技能让 AI 完成编码
4. ✅ 验证与迭代               → Review 代码、运行测试，持续优化
```

#### 📖 配套资源

本项目提供**三位一体**的学习资源，帮助您快速掌握 AI 协作开发模式：

| 资源类型 | 内容说明 |
|---------|---------|
| 📄 **详尽的技术文档** | `DEV_SPEC.md` 提供完整的架构设计、技术选型、模块详解 |
| 💻 **Skills 工作流** | `.trae/skills/` 包含 spec-sync、implement、testing 等 AI 技能，指导 AI 完成开发任务 |
| 🎬 **视频教程** | 从环境搭建到核心模块实现，全程实战演示 |

> 💡 **提示**：详细的设计理念、技术选型与模块设计请参考 [DEV_SPEC.md](DEV_SPEC.md)

### 🎁 你将收获什么

通过学习和实践本项目，你将掌握：

#### 🔥 **最新的 AI 协作技能**
- **Skills 工程化**：学会构建可复用的 AI 技能库，让 AI 成为你的"编程助手"
- **VibeCoding 技巧**：掌握与 AI 高效协作的开发模式，提升 10 倍开发效率
- **文档驱动开发**：理解如何通过规范文档指导 AI 完成复杂工程项目

#### 🎯 **RAG 技术全栈能力**
- **深入每个细节**：从文档解析、智能分块、向量化、混合检索到 Rerank 重排，逐一掌握 RAG 链路的每个环节
- **工程化实践**：不仅是理论，更有生产级代码实现，理解如何将论文技术落地到实际项目
- **性能优化**：学习如何平衡检索速度与精度，优化 Embedding 策略，调优 Rerank 模型

#### 💼 **面试竞争力提升**
- **简历项目加分**：本项目涵盖大模型/AI 工程师岗位的核心技术栈，可直接写入简历作为亮点项目
- **面试问题应对**：配套的面试题库帮你应对"RAG 如何优化召回率"、"Embedding 模型如何选择"等高频问题
- **技术深度展示**：模块化设计、可插拔架构等工程实践，展现你的系统设计能力

---

## 🎓 学习资源与社区

### 📺 配套视频教程

本项目提供**全面的视频讲解**，涵盖：
- ✅ **RAG 核心技术深度剖析**：从分块策略、混合检索到 Rerank 重排，全面解构 RAG 技术细节
- ✅ **代码实战逐行讲解**：环境配置、模块实现、性能优化，手把手带你完成项目
- ✅ **大模型面试真题解析**：精选大厂面试题，结合项目实战讲解答题思路
- ✅ **转行求职指南**：简历撰写技巧、面试准备策略、职业规划建议


### 🎁 额外福利

持续更新中的内容：
- 📝 **基于本项目的简历模板**：如何将技术亮点写进简历
- 🎤 **常见面试问题集锦**：针对 RAG 项目的高频面试题及参考答案
- 💼 **求职经验分享**：从技术学习到拿到 Offer 的完整路径

---

## 🚀 快速开始

```bash
# 1. 克隆项目
git clone https://github.com/yourusername/Modular-RAG-MCP-Server.git
cd Modular-RAG-MCP-Server

# 2. 创建并激活虚拟环境（Python 3.10+）
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .\.venv\Scripts\Activate.ps1  # Windows

# 3. 安装依赖
python -m pip install -U pip
python -m pip install -e ".[dev]"
# 若需 DeepDoc PDF 解析（pdfplumber + OCR + 版面 + 表格），请加装：pip install -e ".[deepdoc]"

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入您的 API Keys (如 ALIYUN_API_KEY 等)

# 5. 首次数据摄取（测试）
python scripts/ingest.py --path tests/fixtures/sample_documents/ --collection test

# 6. 运行 MCP Server
python src/main.py
```

### 使用 uv（无需单独安装 Python）

若本机未安装 Python 或希望用 [uv](https://docs.astral.sh/uv/) 统一管理解释器与依赖，可按以下步骤操作（uv 会自动下载符合 `pyproject.toml` 中 `requires-python` 的 Python）：

```bash
# 1. 安装 uv（Windows PowerShell）
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. 进入项目目录后，用 uv 安装 Python 3.11 并创建虚拟环境（无需从 python.org 下载）
cd path/to/rag-mcp-server
uv python install 3.11
uv venv --python 3.11

# 3. 激活虚拟环境（Windows PowerShell）
.\.venv\Scripts\Activate.ps1

# 4. 按 pyproject.toml 安装依赖（含 dev）
uv pip install -e ".[dev]"
# 若需 DeepDoc PDF 解析，请加装：uv pip install -e ".[deepdoc]"

# 5. 后续步骤同上：配置 .env、摄取数据、运行 python src/main.py
```

> 说明：`pyproject.toml` 只定义项目与依赖，不能替代 Python 解释器；用 uv 时由 uv 负责下载并管理解释器，再根据 `pyproject.toml` 安装依赖。

---

## ⚙️ 配置说明

核心配置文件位于 `config/settings.yaml`，支持热重载。主要配置项说明：

| 配置模块 | 关键字段 | 说明 |
|---------|---------|------|
| **LLM** | `llm.provider` | 支持 `openai`, `azure`, `ollama`。使用 `base_url` 可对接 DeepSeek/Moonshot 等兼容接口。 |
| **Embedding** | `embedding.provider` | 向量模型提供商，支持 `openai`, `ollama`, `local`。 |
| **Vector Store** | `vector_store.backend` | 向量库后端，目前支持 `chroma` (本地持久化)。 |
| **Ingestion** | `ingestion.splitter` | 文档切分策略，推荐 `recursive`。 |
| **Ingestion** | `ingestion.loader.pdf_parser` | PDF 解析方式：`original`（默认，pypdf+图片）或 `deepdoc`（本仓库内 DeepDoc 解析）。详见 `docs/deepdoc-migration-guide.md`。 |
| **Retrieval** | `retrieval.fusion_algorithm` | 混合检索融合算法，默认 `rrf` (Reciprocal Rank Fusion)。 |
| **Rerank** | `rerank.backend` | 精排模型，支持 `cross_encoder` (本地) 或 `llm` (云端)。 |

---

## 🔌 MCP 配置示例

将本项目作为 MCP Server 集成到您的 AI 助手：

### 1. Claude Desktop
编辑 `~/Library/Application Support/Claude/claude_desktop_config.json`：

```json
{
  "mcpServers": {
    "modular-rag": {
      "command": "/absolute/path/to/project/.venv/bin/python",
      "args": ["/absolute/path/to/project/src/main.py"],
      "env": {
        "ALIYUN_API_KEY": "sk-xxx",
        "ALIYUN_BASE_URL": "https://..."
      }
    }
  }
}
```

### 2. GitHub Copilot / Cursor
编辑项目根目录下的 `.vscode/settings.json` 或工具特定的 MCP 配置文件：

```json
{
  "mcp.servers": {
    "modular-rag": {
      "command": "python",
      "args": ["src/main.py"],
      "env": { ... }
    }
  }
}
```

---

## 📊 Dashboard 使用指南

本项目包含一个基于 Streamlit 的可视化管理平台。

**启动 Dashboard：**
```bash
streamlit run src/observability/dashboard/app.py
```

**功能模块：**
1.  **Overview (总览)**：查看系统运行状态、文档总量、最近活动。
2.  **Data Browser (数据浏览器)**：浏览已摄入的文档库，查看 Chunk 切分详情与 Metadata。
3.  **Ingestion Manager (摄取管理)**：上传文件触发处理流程，支持进度实时追踪。
4.  **Ingestion Traces (摄取追踪)**：查看历史摄取任务的耗时瀑布图，定位性能瓶颈。
5.  **Query Traces (查询追踪)**：可视化查询全链路（检索 -> 排序 -> 生成），对比 Dense/Sparse 结果。
6.  **Evaluation (评估面板)**：运行 Golden Test Set，查看 RAG 性能指标 (Hit Rate, MRR)。

---

## 📋 正式使用指南（含监测）

从「测试用 test collection」切换到正式使用时，按以下步骤操作即可。

### 1. 确定正式知识库名称

- 在 `config/settings.yaml` 中，默认 `vector_store.collection_name` 为 **`knowledge_hub`**，可直接作为正式库名。
- 若需多个知识库（如按业务拆分），可后续通过 `--collection 名称` 或 Dashboard 的「Target Collection」分别摄取与查询。

### 2. 摄入正式文档

```bash
# 激活虚拟环境后，将正式文档目录或单文件摄入到默认库 knowledge_hub
python scripts/ingest.py --path /path/to/your/docs/ --collection knowledge_hub

# 或指定其他库名
python scripts/ingest.py --path /path/to/your/docs/ --collection my_production
```

- 也可在 **Dashboard → Ingestion Manager** 中上传文件、选择 Target Collection，并勾选「Force re-ingest」按需强制重跑。

### 3. 启动监测（Dashboard + 追踪日志）

- **追踪日志**：`config/settings.yaml` 中 `observability.log_file` 默认为 `./logs/traces.jsonl`，摄取与查询的 trace 会写入该文件，无需额外配置。
- **Dashboard**：在项目根目录执行：
  ```bash
  streamlit run src/observability/dashboard/app.py
  ```
  浏览器打开提示的地址（默认 `http://localhost:8501`），即可使用：
  - **Overview**：当前配置的 collection、持久化路径、文档量等；
  - **Data Browser**：按 collection 浏览已摄入文档与 Chunk；
  - **Ingestion Manager**：上传新文件、选择 collection、查看/删除已摄入文档；
  - **Ingestion Traces**：历史摄取任务耗时与阶段；
  - **Query Traces**：查询链路的检索与生成记录（需先有查询发生）；
  - **Evaluation**：使用 Golden Test Set 跑 RAG 指标（可选）。

### 4. 连接 MCP 供 AI 助手使用

- 在 Cursor / Claude Desktop 等工具中配置 MCP Server，指向本项目的 `python src/main.py`，并传入 `.env` 中的 API 等环境变量（见上文「MCP 配置示例」）。
- 配置完成后，AI 助手即可通过 MCP 调用 `query_knowledge_hub` 等工具查询你摄入的正式库。

### 5. 可选：与测试数据隔离

- **保留 test**：不删 `test` collection 时，正式使用 `knowledge_hub`（或自定义库名）即可，两者互不影响。
- **不再需要 test**：若使用 jsonl 后端，可删除 `data/db/jsonl/test.jsonl`；若使用 chroma，可在 Chroma 中删除对应 collection。  
  增量历史：若希望正式库不受测试文件影响，可保留 `data/cache/ingestion_history.json`（按 hash 跳过未改文件）；若希望「正式库完全独立」，可新建一份 ingestion 历史或清空该文件后只对正式文档做 ingest。

### 6. 可选：正式环境下的评估

- 默认 `evaluation.golden_test_set` 指向 `./tests/fixtures/golden_test_set.json`。
- 支持 **ragas + custom** 组合：在 `config/settings.yaml` 中设置 `evaluation.backends: [ragas, custom]`，即可同时得到 Ragas 指标（如 faithfulness、answer_relevancy）与自定义指标（Hit Rate、MRR 等）；命令行临时指定可运行：`python scripts/evaluate.py --backends ragas,custom`。
- 正式使用时可在配置中改为自己的 Golden Test Set 路径，并在 **Dashboard → Evaluation** 中运行，用于监控检索与回答质量。

---

## 🧪 运行测试

本项目包含完整的测试金字塔：

```bash
# 运行所有测试
pytest

# 仅运行单元测试 (Unit Tests)
pytest tests/unit

# 仅运行集成测试 (Integration Tests)
pytest tests/integration

# 仅运行端到端测试 (E2E Tests)
pytest tests/e2e

# 运行 Dashboard 冒烟测试
pytest tests/e2e/test_dashboard_smoke.py
```

---

## ❓ 常见问题 (FAQ)

**Q: 安装依赖时报错 `ModuleNotFoundError`？**
A: 请确保已激活虚拟环境 (`source .venv/bin/activate`) 且 pip 版本已更新。

**Q: 启动 Server 提示 `API Key not found`？**
A: 检查 `.env` 文件是否存在且已填入 Key，或者直接在 `config/settings.yaml` 中硬编码（不推荐）。

**Q: Dashboard 无法连接到 Server？**
A: Dashboard 直接读取本地数据库与日志文件，无需 Server 进程启动即可查看历史数据。但"查询追踪"需要先产生查询记录。

**Q: 如何切换到本地模型？**
A: 修改 `config/settings.yaml`，设置 `llm.provider: ollama` 并指定 `base_url` (如 `http://localhost:11434`)。

---

## 📂 项目结构

以下为项目目录与主要文件说明，便于快速定位与维护。

### 树形结构概览

```
.
├── README.md
├── DEV_SPEC.md              # 核心设计文档
├── pyproject.toml
├── .env.example
├── .trae/
│   ├── rules/
│   │   └── project_rules.md
│   └── skills/
│       ├── spec-sync/
│       ├── implement/
│       ├── testing-stage/
│       ├── dev-workflow/
│       ├── stage-summary/
│       ├── problem-recorder/
│       ├── progress-tracker/
│       ├── checkpoint/
│       └── interview-assistant/
├── config/
│   └── settings.yaml
├── scripts/
│   ├── ingest.py
│   ├── query.py
│   ├── evaluate.py
│   ├── rebuild_bm25.py
│   ├── start_dashboard.py
│   ├── verify_connectivity.py
│   ├── verify_trace_data.py
│   └── test_ingestion_transforms.py
├── src/
│   ├── main.py
│   ├── core/
│   │   ├── settings.py
│   │   ├── trace/           # trace_context, trace_collector
│   │   ├── query_engine/    # dense/sparse/hybrid, fusion, reranker, query_processor
│   │   └── response/       # response_builder, citation_generator, multimodal_assembler
│   ├── ingestion/
│   │   ├── pipeline.py
│   │   ├── models.py
│   │   ├── document_manager.py
│   │   ├── embedding/       # dense_encoder, sparse_encoder, batch_processor
│   │   ├── storage/        # vector_upserter, bm25_indexer, image_storage
│   │   └── transform/      # metadata_enricher, chunk_refiner, image_captioner
│   ├── libs/
│   │   ├── llm/            # base, factory, openai/azure/ollama/deepseek
│   │   ├── embedding/      # base, factory, openai, local
│   │   ├── vector_store/   # base, chroma, jsonl, factory
│   │   ├── splitter/       # base, recursive_splitter, factory
│   │   ├── reranker/       # base, cross_encoder, llm_reranker, factory
│   │   ├── loader/         # base, pdf_loader, file_integrity
│   │   └── evaluator/      # base, custom, factory
│   ├── mcp_server/
│   │   ├── server.py
│   │   ├── protocol_handler.py
│   │   └── tools/          # query_knowledge_hub, list_collections, get_document_summary
│   └── observability/
│       ├── logger.py
│       ├── evaluation/     # eval_runner, ragas_evaluator, composite_evaluator
│       └── dashboard/
│           ├── app.py
│           ├── pages/      # overview, data_browser, ingestion_manager, ingestion_traces, query_traces, evaluation_panel
│           └── services/   # app_context, config_service, trace_service
├── tests/
│   ├── conftest.py
│   ├── fixtures/           # golden_test_set.json, sample_documents/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── stage_summaries/
├── problem_summaries/
├── data/                   # 运行时生成：向量库、BM25、缓存、图片
└── logs/                   # 运行时生成：traces.jsonl
```

### 根目录

| 文件/目录 | 作用 |
|----------|------|
| `README.md` | 项目说明、快速开始、配置与使用指南。 |
| `DEV_SPEC.md` | 核心设计文档（项目「宪法」）：架构、模块设计、技术选型与开发规范。 |
| `pyproject.toml` | Python 项目与依赖定义（含 `[dev]` 可选依赖），支持 `pip install -e ".[dev]"`。 |
| `.env.example` | 环境变量模板，复制为 `.env` 后填入 API Key 等敏感配置。 |
| `.trae/` | AI 协作开发配置目录。 |
| `config/` | 应用配置。 |
| `scripts/` | 命令行脚本（摄取、查询、评估、Dashboard 启动等）。 |
| `src/` | 主源码包。 |
| `tests/` | 单元 / 集成 / E2E 测试及 fixtures。 |
| `stage_summaries/` | 阶段总结文档。 |
| `problem_summaries/` | 问题记录文档。 |
| `data/` | 运行时数据（向量库、BM25 索引、缓存、图片等），可随运行生成。 |
| `logs/` | 追踪日志（如 `traces.jsonl`），供 Dashboard 与排查使用。 |

### config/

| 文件 | 作用 |
|------|------|
| `settings.yaml` | 主配置文件：LLM/Embedding/Rerank/向量库/摄取/检索/评测等，支持热重载。 |

### scripts/

| 文件 | 作用 |
|------|------|
| `ingest.py` | 文档摄取入口：指定 `--path` 与 `--collection`，将本地文件解析、切块、向量化并写入向量库与 BM25。 |
| `query.py` | 命令行查询：对指定 collection 做检索与问答，用于快速验证。 |
| `evaluate.py` | 使用 Golden Test Set 跑 RAG 评测（Hit Rate、MRR 等）。 |
| `rebuild_bm25.py` | 按现有向量库/JSONL 数据重建 BM25 索引，用于索引损坏或策略变更后。 |
| `start_dashboard.py` | 启动 Streamlit Dashboard 的封装脚本（可确保项目根在 `sys.path`）。 |
| `verify_connectivity.py` | 校验与外部服务（如 LLM/Embedding API）的连通性。 |
| `verify_trace_data.py` | 校验追踪日志数据是否可读、格式是否正确。 |
| `test_ingestion_transforms.py` | 对摄取阶段各类 transform（enricher、captioner 等）做脚本级测试。 |

### src/ — 主源码

| 文件/目录 | 作用 |
|----------|------|
| `main.py` | 应用入口：将项目根加入 `sys.path`、加载配置；实际 MCP 服务由 `mcp_server.server` 提供。 |
| `__init__.py` | 包标识。 |

### src/core/ — 核心配置与查询链路

| 文件/目录 | 作用 |
|----------|------|
| `settings.py` | 读取 `config/settings.yaml` 与环境变量，暴露 `Settings` 与 `load_settings()`，供全局使用。 |
| `trace/` | 追踪上下文与收集：请求级 trace、span，与 observability 日志对接。 |
| `trace/trace_context.py` | 当前请求的 trace 上下文（如 trace_id、span 栈）。 |
| `trace/trace_collector.py` | 收集 span、写入结构化 trace 日志。 |
| `query_engine/` | 检索与排序：稠密/稀疏检索、融合、精排、元数据过滤、查询处理。 |
| `query_engine/dense_retriever.py` | 基于向量库的稠密检索。 |
| `query_engine/sparse_retriever.py` | 基于 BM25 的稀疏检索。 |
| `query_engine/hybrid_search.py` | 混合检索编排：稠密 + 稀疏 + 融合算法（如 RRF）。 |
| `query_engine/fusion.py` | 多路检索结果的融合逻辑（如 RRF 实现）。 |
| `query_engine/reranker.py` | 精排：调用 Rerank 模型对候选做重排。 |
| `query_engine/metadata_filter.py` | 按元数据过滤检索条件。 |
| `query_engine/query_processor.py` | 查询预处理与检索流程的协调。 |
| `response/` | 回答与引用：组装多模态回复、生成引用。 |
| `response/response_builder.py` | 根据检索结果与 LLM 输出构建最终回答结构。 |
| `response/citation_generator.py` | 根据命中 chunk 生成引用信息。 |
| `response/multimodal_assembler.py` | 多模态内容（文本+图片等）的组装。 |

### src/ingestion/ — 文档摄取

| 文件/目录 | 作用 |
|----------|------|
| `pipeline.py` | 摄取主流程：加载 → 切分 → 可选 transform → 向量化 + BM25 索引 → 写入向量库与图片存储。 |
| `models.py` | 摄取领域模型：Document、Chunk 等数据结构。 |
| `document_manager.py` | 文档库管理：列表、删除、按 collection 管理，供 Dashboard 与 API 使用。 |
| `embedding/` | 摄取侧向量化与批处理。 |
| `embedding/dense_encoder.py` | 使用配置的 Embedding 模型对文本做稠密向量化。 |
| `embedding/sparse_encoder.py` | 为 BM25 生成稀疏表示（如词袋/统计）。 |
| `embedding/batch_processor.py` | 批量调用 Dense/Sparse 编码，控制并发与批大小。 |
| `storage/` | 写入向量库、BM25 索引与图片存储。 |
| `storage/vector_upserter.py` | 将 chunk 向量与元数据 upsert 到向量库。 |
| `storage/bm25_indexer.py` | 构建与更新 BM25 索引（与 sparse 编码配合）。 |
| `storage/image_storage.py` | 图片及索引的持久化（如截图、示意图）。 |
| `transform/` | 摄取阶段的可选变换。 |
| `transform/metadata_enricher.py` | 丰富 chunk 元数据（如来源、页码）。 |
| `transform/chunk_refiner.py` | 对 chunk 做后处理（如去重、截断）。 |
| `transform/image_captioner.py` | 为图片生成描述，便于检索与展示。 |
| `transform/base_transform.py` | Transform 抽象基类。 |

### src/libs/ — 可插拔组件（LLM / Embedding / 向量库 / 分块 / Rerank / 评测）

| 文件/目录 | 作用 |
|----------|------|
| `llm/` | LLM 调用抽象与多实现。 |
| `llm/base_llm.py` | LLM 接口定义。 |
| `llm/llm_factory.py` | 根据配置创建 OpenAI / Azure / Ollama / DeepSeek 等实现。 |
| `llm/openai_llm.py` | OpenAI 兼容 API 的 LLM 实现。 |
| `llm/azure_llm.py` | Azure OpenAI 实现。 |
| `llm/ollama_llm.py` | 本地 Ollama 实现。 |
| `llm/deepseek_llm.py` | DeepSeek 等兼容接口实现。 |
| `embedding/` | 文本向量化抽象与多实现。 |
| `embedding/base_embedding.py` | Embedding 接口定义。 |
| `embedding/embedding_factory.py` | 根据配置创建 OpenAI / 本地等实现。 |
| `embedding/openai_embedding.py` | OpenAI Embedding API 实现。 |
| `embedding/local_embedding.py` | 本地/内网 Embedding 模型实现。 |
| `vector_store/` | 向量存储抽象与多后端。 |
| `vector_store/base_vector_store.py` | 向量库接口。 |
| `vector_store/chroma_store.py` | Chroma 持久化实现。 |
| `vector_store/jsonl_store.py` | JSONL 文件型实现（便于调试/轻量）。 |
| `vector_store/vector_store_factory.py` | 根据配置创建对应后端。 |
| `splitter/` | 文档切分策略。 |
| `splitter/base_splitter.py` | 分块器接口。 |
| `splitter/recursive_splitter.py` | 按层级（段落/句）递归切分。 |
| `splitter/splitter_factory.py` | 根据配置创建分块器。 |
| `reranker/` | 精排模型抽象与实现。 |
| `reranker/base_reranker.py` | Reranker 接口。 |
| `reranker/cross_encoder_reranker.py` | 本地 Cross-Encoder 精排。 |
| `reranker/llm_reranker.py` | 基于 LLM 的精排。 |
| `reranker/reranker_factory.py` | 根据配置创建 Reranker。 |
| `loader/` | 文档加载。 |
| `loader/base_loader.py` | Loader 接口。 |
| `loader/pdf_loader.py` | PDF 解析与文本提取。 |
| `loader/file_integrity.py` | 文件完整性/去重（如基于 hash 的跳过）。 |
| `evaluator/` | RAG 评测抽象与实现。 |
| `evaluator/base_evaluator.py` | 评测器接口。 |
| `evaluator/custom_evaluator.py` | 自定义指标（如 Hit Rate、MRR）。 |
| `evaluator/evaluator_factory.py` | 根据配置创建评测器。 |

### src/mcp_server/ — MCP 协议与工具

| 文件/目录 | 作用 |
|----------|------|
| `server.py` | MCP 服务入口：stdio 协议、注册 Tools、将请求路由到对应 tool 实现。 |
| `protocol_handler.py` | 解析与处理 MCP 协议消息（如 tools/call）。 |
| `tools/` | MCP 暴露的 RAG 能力。 |
| `tools/query_knowledge_hub.py` | 工具：对指定 collection 做检索与问答。 |
| `tools/list_collections.py` | 工具：列出已有 collections。 |
| `tools/get_document_summary.py` | 工具：获取某文档/collection 的摘要信息。 |
| `tools/__init__.py` | 导出工具列表供 server 注册。 |

### src/observability/ — 可观测与 Dashboard

| 文件/目录 | 作用 |
|----------|------|
| `logger.py` | 结构化 trace 写入（如 JSONL），供 Dashboard 与分析使用。 |
| `evaluation/` | 评测执行与报告。 |
| `evaluation/eval_runner.py` | 加载 Golden Set、跑检索/生成、汇总指标。 |
| `evaluation/ragas_evaluator.py` | Ragas 指标集成。 |
| `evaluation/composite_evaluator.py` | 组合多种评测器。 |
| `dashboard/` | Streamlit 可视化管理端。 |
| `dashboard/app.py` | Dashboard 入口：多页导航（Overview、Data Browser、Ingestion、Traces、Evaluation 等）。 |
| `dashboard/pages/overview.py` | 总览页：系统状态、文档量、最近活动。 |
| `dashboard/pages/data_browser.py` | 数据浏览器：按 collection 查看文档与 Chunk。 |
| `dashboard/pages/ingestion_manager.py` | 摄取管理：上传、选择 collection、查看/删除已摄入文档。 |
| `dashboard/pages/ingestion_traces.py` | 摄取追踪：历史任务耗时与阶段。 |
| `dashboard/pages/query_traces.py` | 查询追踪：检索→精排→生成链路可视化。 |
| `dashboard/pages/evaluation_panel.py` | 评估面板：运行 Golden Test、查看 RAG 指标。 |
| `dashboard/services/app_context.py` | Dashboard 共享上下文（如 DocumentManager、配置）。 |
| `dashboard/services/config_service.py` | 向 Dashboard 提供配置读取。 |
| `dashboard/services/trace_service.py` | 读取 trace 日志供各页展示。 |

### tests/

| 文件/目录 | 作用 |
|----------|------|
| `conftest.py` | 公共 pytest fixtures（如临时配置、测试用 collection）。 |
| `fixtures/` | 测试数据：如 `golden_test_set.json`、`sample_documents/`。 |
| `unit/` | 单元测试：各模块独立测试（factory、retriever、reranker、embedding 等）。 |
| `integration/` | 集成测试：如摄取流水线、Chroma  roundtrip、MCP 服务、混合检索。 |
| `e2e/` | 端到端测试：数据摄取、召回、MCP 客户端、Dashboard 冒烟。 |

### .trae/skills/

| 目录/文件 | 作用 |
|----------|------|
| `spec-sync/` | 规范同步：将 DEV_SPEC 与实现对齐的 AI 技能。 |
| `implement/` | 按规范实现代码的 AI 技能。 |
| `testing-stage/` | 测试阶段与用例编写的 AI 技能。 |
| `dev-workflow/` | 整体开发流程的 AI 技能。 |
| `stage-summary/` | 阶段总结的 AI 技能。 |
| `problem-recorder/` | 问题记录的 AI 技能。 |
| `progress-tracker/` | 进度跟踪的 AI 技能。 |
| `checkpoint/` | 检查点/里程碑的 AI 技能。 |
| `interview-assistant/` | 面试辅助的 AI 技能。 |
| `project_rules.md`（在 `.trae/rules/`） | 项目级 AI 规则与约定。 |

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！在贡献代码前，请：
1. 阅读 [DEV_SPEC.md](DEV_SPEC.md) 了解项目架构与设计理念
2. 遵循项目的代码规范（见 `DEV_SPEC.md` 中的"开发规范"章节）
3. 确保测试通过（`pytest tests/`）

---

## 📄 License

[MIT License](LICENSE)

---

## 🌟 Star History

如果这个项目对您有帮助，欢迎 Star ⭐️ 支持！

---
