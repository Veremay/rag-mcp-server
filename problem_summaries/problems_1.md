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
