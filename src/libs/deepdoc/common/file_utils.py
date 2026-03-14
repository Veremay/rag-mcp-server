"""
项目根路径：供 deepdoc 加载 ONNX/资源用，优先环境变量 DEEPDOC_RES_DIR 或包内 res。
"""
import os

# 默认指向本包内 res，或环境变量 DEEPDOC_RES_DIR
_PROJECT_BASE = os.getenv("DEEPDOC_RES_DIR")


def get_project_base_directory(*args):
    global _PROJECT_BASE
    if _PROJECT_BASE is None:
        _PROJECT_BASE = os.path.abspath(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "res")
        )
    if args:
        return os.path.join(_PROJECT_BASE, *args)
    return _PROJECT_BASE


def traversal_files(base):
    """遍历目录下所有文件路径，供 init_in_out 等使用。"""
    for root, _ds, fs in os.walk(base):
        for f in fs:
            yield os.path.join(root, f)
