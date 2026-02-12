# Problem Summaries (Vol. 1)

## Problem 1
**Time:** 2026-02-12

### Description
Keyword Search（关键词搜索）未能召回中文查询结果。例如查询 "刘泽鹏的个人网站地址是什么" 时，Keyword Search 结果为 0，尽管相关文档已正确索引。

### Cause
中文分词与索引粒度不匹配。索引阶段文档被切分为单字或短语（如 "刘", "泽", "鹏", "个人网站"），但查询阶段 `QueryProcessor` 缺乏中文分词能力，将整句 "刘泽鹏的个人网站地址是什么" 视为一个长 Token，导致精确匹配失败。

### Solution
优化 `src/core/query_engine/query_processor.py`，实现轻量级中文分词策略：
1.  新增中文停用词库（包括虚词和疑问词）。
2.  利用停用词将长句切分为短语（如 "刘泽鹏", "个人网站"）。
3.  采用混合粒度策略：同时保留短语和生成的单字（Unigrams），确保能匹配索引中的不同粒度 Token。

---

## Problem 2
**Time:** 2026-02-12

### Description
Sparse Retrieval (BM25) 再次出现命中数为 0 的情况，导致 Trace 日志中 Sparse 阶段无数据。

### Cause
`BM25Indexer.build()` 方法逻辑为全量重建。每次 Ingestion Pipeline 运行时（即使只处理少量文档），都会用当前批次的数据覆盖整个索引文件，导致之前的历史索引数据丢失。

### Solution
1.  **实现增量更新**：在 `src/ingestion/storage/bm25_indexer.py` 中新增 `upsert()` 方法，支持加载现有索引并在其基础上更新词频和文档统计。
2.  **管道集成**：修改 `src/ingestion/pipeline.py`，将默认索引构建行为从 `build` 改为 `upsert`。
3.  **数据恢复**：编写并运行 `scripts/rebuild_bm25.py`，从 Vector Store (Chroma/Jsonl) 中读取所有 Chunk 重新生成完整的 BM25 索引。

---

## Problem 3
**Time:** 2026-02-12 11:35:00

### Description
关于 Ingestion Transforms (`ChunkRefiner`, `MetadataEnricher`, `ImageCaptioner`) 在关闭 LLM 增强 (`enable_llm: false`) 时的降级逻辑与行为确认。

### Cause
用户需要了解在无 LLM 资源或出于成本/速度考虑关闭 LLM 时，系统如何处理数据清洗与元数据生成，以及功能会受到何种程度的影响。

### Solution
明确了各组件的 Rule-based Fallback 实现：
1.  **ChunkRefiner**: 仅进行正则清洗（Regex Cleaning）。移除页码标识（如 `Page X of Y`）、孤立数字行及首尾空白，不进行任何语义层面的文本润色或断句修复。
2.  **MetadataEnricher**: 采用统计与截断策略。
    *   **Title**: 提取第一行非空文本。
    *   **Summary**: 直接截取文本前 N 个字符（由 `max_summary_chars` 控制）。
    *   **Tags**: 基于词频统计（Frequency-based），取前 800 字符中出现频率最高的词（过滤停用词）。
3.  **ImageCaptioner**: **完全失效**。若未启用 Vision LLM，该步骤直接跳过，不做任何处理，图片内容将无法被索引。

#### 功能对比表
| 组件 | 开启 LLM (Enable) | 关闭 LLM (Disable) | 差异 |
| :--- | :--- | :--- | :--- |
| **ChunkRefiner** | 智能重写，修复断句、指代不清，优化语义 | **仅正则清洗**，去除页码和多余空行 | 语义质量大幅下降 |
| **MetadataEnricher** | 深度理解内容，生成精准摘要、标题和主题标签 | **机械截取**首行和前 N 字符，基于**词频**生成标签 | 元数据准确性和概括性下降 |
| **ImageCaptioner** | 视觉模型识别图片内容并生成描述文本 | **完全跳过**，不做任何处理 | 失去图片检索能力 |


#### 总结

如果不开启 LLM， TRANSFORM 阶段本质上退化为一个**“极其轻量的去空格工具 + 简单的元数据提取器”**。如果您的源文档本身比较干净（没有页眉页脚噪音），那么 Split 输出 = Transform 输出 是完全符合预期
---
