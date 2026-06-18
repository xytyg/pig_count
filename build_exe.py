#!/usr/bin/env python3
"""
打包 count_bot.py 为可执行 exe 程序
用法: python build_exe.py
"""

import os
import sys
import subprocess
import shutil

APP_NAME = "猪只计数监控"
SCRIPT = "count_bot.py"
ICON = "src/1.ico"
OUTPUT_DIR = "dist"

# 需要打包进去的数据文件（相对路径 → 在 exe 中的路径）
DATA_FILES = [
    ("src/best.pt", "src"),
    ("src/1.ico", "src"),
    ("src/2.mp4", "src"),
    ("botsort.yaml", "."),
]


def main():
    # 检查 pyinstaller
    try:
        import PyInstaller
    except ImportError:
        print("[构建] 正在安装 PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller", "-q"])

    # 清理旧构建
    for d in ["build", OUTPUT_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)
    for f in [f"{APP_NAME}.spec"]:
        if os.path.exists(f):
            os.remove(f)

    # 组装命令
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--onedir",
        "--name", APP_NAME,
        "--icon", ICON,
        # 显式添加数据文件（不打包整个 src/，避免 ~2.5GB 测试视频）
        "--add-data", f"src/best.pt{os.pathsep}src",
        "--add-data", f"src/1.ico{os.pathsep}src",
        "--add-data", f"botsort.yaml{os.pathsep}.",
        "--hidden-import", "PIL",
        "--hidden-import", "PIL._tkinter_finder",
        "--hidden-import", "cv2",
        "--hidden-import", "ultralytics",
        "--hidden-import", "ultralytics.nn.tasks",
        "--hidden-import", "tkinter",
        "--hidden-import", "tkinter.filedialog",
        "--hidden-import", "tkinter.messagebox",
        SCRIPT,
    ]

    print(f"[构建] 开始打包 {APP_NAME}...")
    print(f"[构建] 命令: {' '.join(cmd)}")
    subprocess.check_call(cmd)

    # 构建完成
    exe_path = os.path.join(OUTPUT_DIR, APP_NAME, f"{APP_NAME}.exe")
    if os.path.exists(exe_path):
        print(f"\n[完成] 打包成功！")
        print(f"[完成] 可执行文件: {os.path.abspath(exe_path)}")
        print(f"[完成] 整个程序目录: {os.path.abspath(os.path.join(OUTPUT_DIR, APP_NAME))}")
        print(f"[完成] 直接运行 {APP_NAME}.exe 即可启动")
    else:
        print(f"\n[错误] 打包失败，未生成 exe 文件")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
