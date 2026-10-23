"""参数帮助：可点击的「!」按钮与 HTML 说明弹窗。"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, QUrl
from PySide6.QtGui import (
    QColor,
    QDesktopServices,
    QFont,
    QPainter,
    QPen,
)
from PySide6.QtWidgets import (
    QAbstractButton,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from .theme import theme

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HELP_PATH = PROJECT_ROOT / "docs" / "parameters.html"


class HelpButton(QAbstractButton):
    """参数旁的小圆形感叹号。"""

    def __init__(self, topic: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.topic = topic
        self.setFixedSize(17, 17)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setToolTip("查看参数说明")
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self._hover = False
        self.clicked.connect(lambda: show_help(self.topic, self))
        theme.changed.connect(self.update)

    def enterEvent(self, event) -> None:  # noqa: N802
        self._hover = True
        self.update()

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._hover = False
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        color = QColor(theme.palette.accent if self._hover else theme.palette.tertiary)
        painter.setPen(QPen(color, 1.15))
        painter.setBrush(
            theme.rgba("accent", 30) if self._hover else Qt.BrushStyle.NoBrush
        )
        painter.drawEllipse(QRectF(1.2, 1.2, 14.6, 14.6))
        font = QFont(self.font())
        font.setPixelSize(10)
        font.setWeight(QFont.Weight.Bold)
        painter.setFont(font)
        painter.setPen(color)
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "!")


def help_label(text: str, topic: str) -> QWidget:
    """创建一个「标签 + !」组合，用于表单左侧。"""
    holder = QWidget()
    layout = QHBoxLayout(holder)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)
    title = QLabel(text)
    title.setProperty("role", "muted")
    title.style().unpolish(title)
    title.style().polish(title)
    layout.addWidget(title)
    layout.addWidget(HelpButton(topic))
    layout.addStretch(1)
    return holder


class HelpDialog(QDialog):
    def __init__(self, topic: str = "index", parent: QWidget | None = None):
        super().__init__(parent)
        self.topic = topic
        self.setWindowTitle("NeuroStudio · 参数说明")
        self.resize(740, 660)
        self.setMinimumSize(520, 420)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(True)
        self.browser.setFrameShape(QFrame.Shape.NoFrame)
        layout.addWidget(self.browser, 1)

        bottom = QHBoxLayout()
        bottom.setSpacing(8)
        bottom.addStretch(1)
        self.open_btn = QPushButton("在浏览器中打开")
        self.open_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.open_btn.clicked.connect(self._open_in_browser)
        bottom.addWidget(self.open_btn)
        close_btn = QPushButton("关闭")
        close_btn.setProperty("variant", "primary")
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.clicked.connect(self.accept)
        bottom.addWidget(close_btn)
        layout.addLayout(bottom)

        self._load_document()

    def _load_document(self) -> None:
        if not HELP_PATH.exists():
            self.browser.setPlainText(
                f"未找到参数说明文件：\n{HELP_PATH}\n\n请确认 docs/parameters.html 存在。"
            )
            self.open_btn.setEnabled(False)
            return
        raw = HELP_PATH.read_text(encoding="utf-8")
        palette = theme.palette
        injected = (
            "<style>"
            f"body {{ color: {palette.text}; }}"
            f"h2, td, th {{ border-bottom-color: {palette.separator}; }}"
            f".note {{ color: {palette.secondary}; }}"
            f"a {{ color: {palette.accent}; }}"
            f"code {{ color: {palette.accent}; }}"
            "</style>"
        )
        html = raw.replace("</head>", injected + "</head>")
        self.browser.setHtml(html)
        QTimer.singleShot(0, lambda: self.browser.scrollToAnchor(self.topic))

    def _open_in_browser(self) -> None:
        if HELP_PATH.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(HELP_PATH)))


def show_help(topic: str = "index", parent: QWidget | None = None) -> None:
    window = parent.window() if parent is not None else None
    dialog = HelpDialog(topic, window)
    dialog.exec()
