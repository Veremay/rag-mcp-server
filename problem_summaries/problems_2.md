# Problem Summaries (Vol. 2)

## Problem 1
**Time:** 2026-03-18 00:00:00

### Description
新增DeepDoc PDF解析功能，提供高级PDF处理能力，包括OCR、版面识别、表格提取和图像处理。

### Cause
为了提升PDF文档的解析质量和准确性，特别是对于包含复杂布局、表格和图片的学术论文和专业文档。

### Solution
实现了完整的DeepDoc PDF解析系统，包括：

**核心组件：**
- `RAGFlowPdfParser`：完整的PDF解析器，支持OCR、版面分析、表格识别
- `DeepDocPdfLoader`：加载器，与现有系统集成，提供回退机制
- `PlainParser`：纯文本解析器，作为回退方案

**主要特性：**
1. **多阶段解析流程**：pdfplumber转图 → OCR识别 → 版面分析 → 表格提取
2. **智能版面识别**：识别文本、表格、图表等不同元素类型
3. **高级表格处理**：提取表格结构并转换为HTML格式
4. **图像提取**：自动识别并保存文档中的图片
5. **多列文本处理**：智能识别和处理多列布局
6. **容错机制**：当完整解析失败时回退到纯文本解析
7. **兼容性**：与现有PdfLoader接口保持兼容

**技术实现：**
- 使用pdfplumber进行PDF转图像
- 集成OCR引擎进行文本识别
- 采用XGBoost模型进行文本行合并判断
- 使用KMeans聚类进行列数识别
- 实现表格结构识别和HTML转换
- 支持多设备并行处理（通过asyncio Semaphore）

**集成方式：**
- 在配置中设置`pdf_parser=deepdoc`启用
- 输出与现有Document模型兼容，包含text、metadata和image_refs
- 表格数据单独存储在metadata["tables"]中，便于后续处理

**文件结构：**
- `src/libs/deepdoc/`：核心解析库
- `src/libs/loader/deepdoc_pdf_loader.py`：加载器实现
- `tests/unit/test_deepdoc_*.py`：单元测试

此功能显著提升了系统处理复杂PDF文档的能力，特别是对于包含表格、多列布局和图像的学术论文。

---