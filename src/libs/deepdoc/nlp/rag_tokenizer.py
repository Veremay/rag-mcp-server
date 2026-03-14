"""
最小 rag_tokenizer 实现：供 table_structure_recognizer / pdf_parser 的 tokenize/tag/is_chinese，
不依赖 infinity；表格/版面逻辑仅需简单分词与中文判断。
"""
import re


def is_chinese(text):
    """判断字符串是否主要为中文（用于版面/表格块类型）。"""
    if not text or not isinstance(text, str):
        return False
    chinese = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return chinese / len(text) > 0.2


def _simple_tokenize(text):
    """简单按空白与标点切分，返回空格连接的 token 串（与 RAGFlow tokenize 行为接近）。"""
    if not text or not isinstance(text, str):
        return ""
    # 按空白、中英文标点切分，保留连续字母数字
    tokens = re.findall(r"[^\s\W]+|[^\s]", text)
    return " ".join(t for t in tokens if t.strip())


def _tag(token):
    """简单词性占位：表格 blockType 仅用 tag(tks[0])=='nr'；无词性库时返回空或通用标记。"""
    if not token:
        return ""
    if len(token) <= 2 and (token.isdigit() or is_chinese(token)):
        return "nr"
    return "n"


class _RagTokenizer:
    def tokenize(self, line: str) -> str:
        return _simple_tokenize(line)

    def fine_grained_tokenize(self, tks: str) -> str:
        return tks

    def tag(self, token: str) -> str:
        return _tag(token)


rag_tokenizer = _RagTokenizer()
tokenize = rag_tokenizer.tokenize
fine_grained_tokenize = rag_tokenizer.fine_grained_tokenize
tag = _tag
