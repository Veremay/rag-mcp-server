---
name: stage-summary
description: 在大阶段完成时，询问是否生成阶段总结；若需要，则生成包含功能概览、完成项、文件改动与 Markdown 流程图（ASCII/Unicode）的 Markdown 文件。
metadata:
  category: review
  triggers: "阶段总结, 阶段复盘, 大阶段总结, phase summary, stage summary"
allowed-tools: Read SearchCodebase Grep LS RunCommand apply_patch
---

# 阶段总结生成器（Stage Summary）

## 目标

当用户完成一个大阶段（阶段 A/B/C/...）后：

1. 主动询问是否需要对该阶段进行总结
2. 若用户确认需要，总结该阶段：
   - 做了什么、实现了什么（基于 DEV_SPEC.md 的阶段目标/任务项 + 实际代码结构）
   - 改动在哪些文件（优先用 git 推断；无 git 则回退到 DEV_SPEC.md 中“修改文件”汇总）
   - 生成对应流程/架构图（Markdown 代码块中的 ASCII/Unicode 流程图）
3. 将总结写入项目根目录下的 `stage_summaries/` 目录（如不存在则创建），按阶段分文件存档

## 交互与止损规则（必须遵守）

1. 本技能至少 2 轮对话：先询问是否总结，再生成并落盘
2. 只要提出问题（例如“是否生成总结”），必须立刻结束本轮输出，等待用户回复
3. 不模拟用户确认；必须等待用户真实回复“需要/是/yes”或“跳过/否/no”
4. 不修改任何业务逻辑代码；只允许新增/更新 `stage_summaries/*.md`
5. 阶段总结落盘后，自动激活 `interview-assistant` 技能：由其询问是否生成面试问题并追加到同一文件末尾。

## Turn 1：识别阶段并询问

### 1) 推断“可能需要总结的阶段”

按以下优先级确定候选阶段：

1. 若用户明确说“总结阶段 C/阶段C/Phase C”，则候选阶段 = 用户指定
2. 否则读取 `DEV_SPEC.md` 的“总体进度/阶段进度”信息，找到最近一个 `100%` 的阶段作为候选阶段

### 2) 询问模板（必须原样输出并停止）

输出：

```text
检测到你可能刚完成了【阶段 <X>】。
是否需要生成该阶段总结（写入 stage_summaries/ 目录）？

回复：
  - 需要 / 是 / yes：生成总结
  - 跳过 / 否 / no：不生成
  - 指定阶段：例如 “阶段 D”
```

然后立刻停止，等待用户回复。

## Turn 2：生成阶段总结并写入文件

触发条件：用户回复“需要/是/yes”或指定阶段字母。

### 1) 采集信息（按顺序，尽量自动化）

#### 1.1 从 DEV_SPEC.md 提取阶段信息

- 阶段名称与目标：来自“阶段总览（大阶段 → 目的）”
- 阶段任务完成情况：来自“进度跟踪表”的对应阶段表格
- 阶段详细说明（若存在）：来自对应“阶段 X：...”章节内容（包含各子任务的“修改文件/验收标准/测试方法”）

#### 1.2 汇总文件改动清单（两种来源，尽量同时提供）

- Git 推断（优先）：
  - 若处于 git 仓库中，尝试用阶段任务表里的最小/最大“完成日期”作为时间窗，运行：
    - `git log --since <min_date> --until <max_date> --name-only --pretty=format:`
  - 去重后得到 `files_changed_by_git`
- Spec 推断（回退/补充）：
  - 在 DEV_SPEC.md 的阶段 X 详细章节中，提取每个子任务下 `- **修改文件**：` 小节的文件路径
  - 去重后得到 `files_changed_by_spec`

输出文件清单时，必须标注来源（Git 推断 / Spec 推断），避免误导。

#### 1.3 形成流程/架构图（Markdown ASCII/Unicode）

按阶段字母选择模板（可根据实际文件/模块名称做最小调整）：

- 阶段 A：工程骨架与测试基座
- 阶段 B：Libs 可插拔层
- 阶段 C：Ingestion Pipeline
- 阶段 D：Retrieval
- 阶段 E：MCP Server 与 Tools
- 阶段 F：Observability + Evaluation
- 阶段 G：E2E + 文档收口

输出必须是 Markdown 代码块（建议使用 ```text），风格参考 DEV_SPEC.md 中的“框线流程图”。例如：

```text
用户查询 (via MCP Client)
      │
      ▼
┌─────────────────┐
│  MCP Server     │  JSON-RPC 解析，工具路由
│ (Stdio Transport)│
└────────┬────────┘
         │ query + params
         ▼
┌─────────────────┐
│ Query Processor │  关键词提取 + 同义词扩展 + Metadata 解析
└────────┬────────┘
         │ processed_query + filters
         ▼
┌─────────────────────────────────────────────┐
│              Hybrid Search                  │
│  ┌─────────────┐          ┌─────────────┐   │
│  │Dense Retrieval│  并行   │Sparse Retrieval│   │
│  │ (Embedding)  │◄───────►│  (BM25)     │   │
│  └──────┬──────┘          └──────┬──────┘   │
│         │                        │          │
│         └────────┬───────────────┘          │
│                  ▼                          │
│         ┌─────────────┐                     │
│         │   Fusion    │  RRF 融合           │
│         │   (RRF)     │                     │
│         └──────┬──────┘                     │
└────────────────┼────────────────────────────┘
                 │ Top-M 候选
                 ▼
┌─────────────────┐
│    Reranker     │  CrossEncoder / LLM / None
│   (Optional)    │
└────────┬────────┘
         │ Top-K 精排结果
         ▼
┌─────────────────┐
│ Response Builder│  引用生成 + 图片 Base64 编码 + MCP 格式化
└────────┬────────┘
         │ MCP Response (TextContent + ImageContent)
         ▼
返回给 MCP Client (Copilot / Claude Desktop)
```

若无法匹配具体阶段模板，输出一个“输入 → 处理 → 输出”的通用框线流程图，并在正文标注“需人工复核”。

### 2) 生成 Markdown 内容结构（固定）

总结文件必须包含以下段落（顺序固定）：

1. 标题：`# 阶段 <X> 总结`
2. 元信息：生成日期、阶段时间窗（若可推断）、对应 DEV_SPEC 章节引用
3. 阶段目标（从 DEV_SPEC 提取）
4. 完成内容清单：
   - 已完成任务列表（含任务编号与名称）
   - 关键实现点（基于任务名 + 实际代码结构提炼，避免编造）
5. 代码改动影响面：
   - 文件变更（分 Created/Modified 若 git 能提供；否则仅列清单）
   - 关键模块入口（例如脚本入口、管线入口、工厂入口）
6. 架构/流程图（Markdown ASCII/Unicode）
7. 验收/回归建议（从 DEV_SPEC 的测试方法提取或最小可运行命令）
8. 人工复核点（必须包含）：
   - 文件清单是否完整
   - ASCII/Unicode 流程图是否与实际一致

### 3) 落盘规则

- 输出目录：项目根目录 `stage_summaries/`
- 文件名：`stage_<X>_<min_date>_to_<max_date>.md`
  - 若日期不可推断：`stage_<X>_<YYYY-MM-DD>.md`
- 若同名文件存在：在文件顶部追加一行 `> Updated: <timestamp>`，并合并更新内容（不新增第二份同名文件）

### 4) 完成回执（写入后输出）

写入成功后，输出：

- 生成文件路径（绝对路径）
- 本次总结覆盖的阶段与时间窗
- ASCII/Unicode 流程图的标题

随后自动进入 `interview-assistant` 技能，并把“本次生成的阶段总结文件路径”作为上下文。由 `interview-assistant` 负责弹出询问并在用户确认后将面试问题追加到文件末尾（本技能不再额外提问）。

## 资源消耗提示

- 主要消耗来自 `git log` 与代码检索：通常为轻量 IO（秒级），随仓库规模线性增长
- 不执行任何网络调用，不会触发外部 API 费用
