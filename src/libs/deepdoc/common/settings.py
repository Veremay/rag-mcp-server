"""
占位：仅提供 PARALLEL_DEVICES 供 OCR/pdf_parser 多 GPU 并行；默认 0（单设备）。
"""
import logging
import os

# 多 GPU 数量，0 表示单设备；可由 init_parallel_devices() 根据 torch.cuda 更新
PARALLEL_DEVICES = int(os.environ.get("DEEPDOC_PARALLEL_DEVICES", "0"))


def init_parallel_devices():
    """若已安装 torch 且存在 CUDA，则设置 PARALLEL_DEVICES 为 GPU 数量。"""
    global PARALLEL_DEVICES
    try:
        from src.libs.deepdoc.common import misc_utils
        misc_utils.pip_install_torch()
        import torch
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            PARALLEL_DEVICES = count
            logging.info("deepdoc: found %s gpus", count)
    except Exception:
        pass
