"""
占位：pip_install_torch 供 OCR 加载 GPU 时可选安装 torch；本仓库默认 CPU 可跳过。
"""
import logging
import os
import subprocess
import sys

_ran = False


def pip_install_torch():
    """若 DEVICE=gpu 可尝试安装 torch，否则 no-op，便于 OCR 使用 GPU。"""
    global _ran
    if _ran:
        return
    _ran = True
    device = os.getenv("DEVICE", "cpu")
    if device != "cpu":
        try:
            logging.info("Installing pytorch for GPU")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "torch>=2.5.0,<3.0.0"],
                timeout=300,
            )
        except Exception as e:
            logging.warning("pip_install_torch failed: %s", e)
