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
