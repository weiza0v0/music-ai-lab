"""Stage A — environment audit. Writes docs/environment_report.md and prints a summary."""
from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "environment_report.md"


def _mem() -> str:
    lines = sh(["free", "-h"]).splitlines()
    return lines[1].strip() if len(lines) > 1 else "n/a"


def sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"<error: {exc}>"


def main() -> None:
    import importlib.metadata as md

    lines = ["# 环境审计报告（阶段 A）", "",
             f"- 生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
             f"- 操作系统：{platform.platform()}",
             f"- Python：{sys.version.split()[0]}（{sys.executable}）",
             f"- CPU：{platform.processor() or 'unknown'} / {shutil.os.cpu_count()} 逻辑核",
             f"- 内存：{_mem()}",
             f"- 磁盘：\n```\n{sh(['df', '-h', str(ROOT)])}\n```", ""]

    gpu = sh(["nvidia-smi"])
    lines += ["## GPU", ""]
    if "Failed to initialize NVML" in gpu or not gpu:
        lines += ["- WSL 内 `nvidia-smi` 失败：`Failed to initialize NVML: GPU access blocked by the operating system`。",
                  "- 典型原因：当前进程的文件/设备沙箱没有放行 `/dev/dxg`（`open('/dev/dxg')` 返回 EACCES），",
                  "  NVML 于是报告 GPU 被系统阻断，CUDA `cuInit(0)` 返回 100（CUDA_ERROR_NO_DEVICE）。",
                  "- 处理办法：在 DSH 中把文件策略切到 `danger-full-access` 后重试；Windows 侧 `nvidia-smi.exe` 应始终正常。",
                  "- **本次实验若在无 GPU 的会话中运行，则退化为 CPU 训练**。"]
    else:
        torch_note = ""
        try:
            import torch
            if torch.cuda.is_available():
                torch_note = (f"- torch 可见 {torch.cuda.device_count()} 块 GPU："
                              f"{torch.cuda.get_device_name(0)}，算力 "
                              f"{torch.cuda.get_device_capability(0)}。")
        except Exception:  # noqa: BLE001
            pass
        lines += ["- `nvidia-smi` 正常：", "```", gpu[:1500], "```", torch_note]
    lines += [""]

    try:
        import torch
        lines += ["## PyTorch", "",
                  f"- torch {torch.__version__}，CUDA 可用：{torch.cuda.is_available()}，"
                  f"设备数：{torch.cuda.device_count()}", ""]
    except Exception as exc:  # noqa: BLE001
        lines += ["## PyTorch", "", f"- 导入失败：{exc}", ""]

    lines += ["## 关键依赖", "", "| 包 | 版本 |", "|---|---|"]
    for pkg in ["torch", "numpy", "pandas", "scikit-learn", "librosa", "soundfile",
                "matplotlib", "streamlit", "pytest"]:
        try:
            lines.append(f"| {pkg} | {md.version(pkg)} |")
        except md.PackageNotFoundError:
            lines.append(f"| {pkg} | 缺失 |")
    lines += ["", "## 音频工具链", "",
              f"- ffmpeg: {sh(['bash', '-lc', 'ffmpeg -version 2>/dev/null | head -1'])}",
              f"- ffprobe: {sh(['bash', '-lc', 'which ffprobe'])}", ""]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[env] wrote {OUT}")


if __name__ == "__main__":
    main()
