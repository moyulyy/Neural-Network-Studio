"""NeuroStudio 启动入口。

用法：
    conda activate neurostudio
    python run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover - 安装提示
        print("未找到 PyTorch，请先安装 CUDA 版本：", file=sys.stderr)
        print("  python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121", file=sys.stderr)
        print(f"\n原始错误：{exc}", file=sys.stderr)
        return 1

    try:
        from PySide6.QtCore import QSettings
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:  # pragma: no cover
        print("未找到 PySide6，请执行：python -m pip install PySide6", file=sys.stderr)
        print(f"原始错误：{exc}", file=sys.stderr)
        return 1

    app = QApplication(sys.argv)
    app.setApplicationName("NeuroStudio")
    app.setApplicationDisplayName("NeuroStudio")
    app.setOrganizationName("NeuroStudio")
    app.setStyle("Fusion")

    from neurostudio.ui.theme import theme

    settings = QSettings("NeuroStudio", "NeuroStudio")
    saved_theme = str(settings.value("theme", "light"))
    theme.set_mode(saved_theme, notify=False)

    from neurostudio.ui.main_window import MainWindow

    window = MainWindow()
    window.show()
    theme.changed.connect(lambda: settings.setValue("theme", theme.mode))
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
