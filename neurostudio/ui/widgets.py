"""iOS 风格通用控件。"""

from __future__ import annotations

from typing import Any, Callable

from PySide6.QtCore import (
    QEasingCurve,
    QEvent,
    QPoint,
    QPointF,
    QPropertyAnimation,
    QRectF,
    QSize,
    Qt,
    QTimer,
    Property,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QAbstractButton,
    QButtonGroup,
    QComboBox,
    QFrame,
    QGraphicsOpacityEffect,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .icons import icon, icon_pixmap
from .theme import theme


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------


def repolish(widget: QWidget) -> None:
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


def set_role(widget: QWidget, role: str) -> None:
    widget.setProperty("role", role)
    repolish(widget)


def label(text: str = "", role: str | None = None, parent: QWidget | None = None) -> QLabel:
    widget = QLabel(text, parent)
    if role:
        set_role(widget, role)
    return widget


def make_icon_badge(name: str, tone: str = "accent", size: int = 30) -> QPixmap:
    scale = 2
    pixmap = QPixmap(size * scale, size * scale)
    pixmap.fill(Qt.GlobalColor.transparent)
    pixmap.setDevicePixelRatio(scale)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    background = theme.rgba(f"{tone}_soft" if tone != "gray" else "card_alt", 255)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(background)
    painter.drawRoundedRect(QRectF(0, 0, size, size), size * 0.3, size * 0.3)
    color = theme.palette.accent if tone in {"accent", "gray"} else getattr(theme.palette, tone)
    inner = max(14, int(size * 0.55))
    painter.drawPixmap(
        int((size - inner) / 2),
        int((size - inner) / 2),
        icon_pixmap(name, color, inner),
    )
    painter.end()
    return pixmap


def icon_button(name: str, tooltip: str = "", size: int = 30, tone: str = "secondary") -> QToolButton:
    button = QToolButton()
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    button.setFixedSize(size, size)
    button.setIconSize(QSize(17, 17))
    button.setToolTip(tooltip)
    button._icon_name = name  # type: ignore[attr-defined]
    button._icon_tone = tone  # type: ignore[attr-defined]
    button._icon_size = 17  # type: ignore[attr-defined]
    button.setIcon(icon(name, getattr(theme.palette, tone), 17))
    return button


def primary_button(text: str, icon_name: str | None = None, height: int = 36) -> QPushButton:
    button = QPushButton(text)
    button.setProperty("variant", "primary")
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    button.setMinimumHeight(height)
    if icon_name:
        button.setIcon(icon(icon_name, "#FFFFFF", 16))
        button.setIconSize(QSize(16, 16))
        button._icon_name = icon_name  # type: ignore[attr-defined]
        button._icon_tone = "#FFFFFF"  # type: ignore[attr-defined]
        button._icon_size = 16  # type: ignore[attr-defined]
    return button


def secondary_button(text: str, icon_name: str | None = None, height: int = 36) -> QPushButton:
    button = QPushButton(text)
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    button.setMinimumHeight(height)
    if icon_name:
        button.setIcon(icon(icon_name, theme.palette.secondary, 16))
        button.setIconSize(QSize(16, 16))
        button._icon_name = icon_name  # type: ignore[attr-defined]
        button._icon_tone = "secondary"  # type: ignore[attr-defined]
        button._icon_size = 16  # type: ignore[attr-defined]
    return button


def quiet_button(text: str, icon_name: str | None = None) -> QPushButton:
    button = QPushButton(text)
    button.setProperty("variant", "quiet")
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    if icon_name:
        button.setIcon(icon(icon_name, theme.palette.accent, 15))
        button.setIconSize(QSize(15, 15))
        button._icon_name = icon_name  # type: ignore[attr-defined]
        button._icon_tone = "accent"  # type: ignore[attr-defined]
        button._icon_size = 15  # type: ignore[attr-defined]
    return button


def refresh_theme_recursive(root: QWidget) -> None:
    """主题切换后刷新自绘控件与图标。"""
    widgets = [root] + root.findChildren(QWidget)
    seen: set[int] = set()
    for widget in widgets:
        if id(widget) in seen:
            continue
        seen.add(id(widget))
        refresh = getattr(widget, "refresh_theme", None)
        if callable(refresh):
            try:
                refresh()
            except Exception:
                pass
        name = getattr(widget, "_icon_name", None)
        if name:
            tone = getattr(widget, "_icon_tone", "secondary")
            color = tone if str(tone).startswith("#") else getattr(theme.palette, tone)
            size = int(getattr(widget, "_icon_size", 16))
            if hasattr(widget, "setIcon"):
                widget.setIcon(icon(name, color, size))


# ---------------------------------------------------------------------------
# 容器
# ---------------------------------------------------------------------------


class Card(QFrame):
    def __init__(self, padding: int = 20, spacing: int = 14, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("Card")
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(padding, padding, padding, padding)
        self.body.setSpacing(spacing)

    def add(self, widget: QWidget, stretch: int = 0) -> QWidget:
        self.body.addWidget(widget, stretch)
        return widget

    def add_layout(self, layout, stretch: int = 0):
        self.body.addLayout(layout, stretch)
        return layout

    def add_stretch(self, stretch: int = 1) -> None:
        self.body.addStretch(stretch)


class InsetCard(QFrame):
    def __init__(self, padding: int = 14, spacing: int = 8, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("InsetCard")
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(padding, padding, padding, padding)
        self.body.setSpacing(spacing)


class Separator(QFrame):
    def __init__(self, vertical: bool = False, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("Separator")
        if vertical:
            self.setFixedWidth(1)
            self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        else:
            self.setFixedHeight(1)
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)


class CardHeader(QWidget):
    def __init__(
        self,
        title: str,
        description: str = "",
        right: QWidget | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)
        text_col = QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(3)
        text_col.addWidget(label(title, "card-title"))
        if description:
            text_col.addWidget(label(description, "card-desc"))
        row.addLayout(text_col, 1)
        if right is not None:
            row.addWidget(right, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.right = right


class PageHeader(QWidget):
    def __init__(self, title: str, subtitle: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)
        self.title_label = label(title, "title")
        row.addWidget(self.title_label)
        row.addStretch(1)
        self.actions = QHBoxLayout()
        self.actions.setSpacing(8)
        row.addLayout(self.actions)
        outer.addLayout(row)
        self.subtitle_label = label(subtitle, "subtitle")
        self.subtitle_label.setWordWrap(True)
        outer.addWidget(self.subtitle_label)

    def add_action(self, widget: QWidget) -> QWidget:
        self.actions.addWidget(widget)
        return widget

    def set_subtitle(self, text: str) -> None:
        self.subtitle_label.setText(text)


class SectionLabel(QLabel):
    def __init__(self, text: str, parent: QWidget | None = None):
        super().__init__(text.upper(), parent)
        set_role(self, "section")


class EmptyState(QWidget):
    def __init__(
        self,
        title: str,
        description: str = "",
        icon_name: str = "sparkles",
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 30, 20, 30)
        layout.setSpacing(8)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.icon_label = QLabel()
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.icon_label)
        layout.addWidget(label(title, "card-title"), 0, Qt.AlignmentFlag.AlignCenter)
        if description:
            desc = label(description, "muted")
            desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
            desc.setWordWrap(True)
            layout.addWidget(desc, 0, Qt.AlignmentFlag.AlignCenter)
        self._icon_name = icon_name
        self.refresh_theme()

    def refresh_theme(self) -> None:
        self.icon_label.setPixmap(icon_pixmap(self._icon_name, theme.palette.tertiary, 34))


# ---------------------------------------------------------------------------
# 标签 / 统计
# ---------------------------------------------------------------------------


class Pill(QWidget):
    TONES = {"accent", "green", "orange", "red", "purple", "teal", "gray"}

    def __init__(self, text: str = "", tone: str = "gray", parent: QWidget | None = None):
        super().__init__(parent)
        self._text = text
        self._tone = tone if tone in self.TONES else "gray"
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        theme.changed.connect(self.update)

    def setText(self, text: str) -> None:
        self._text = text
        self.updateGeometry()
        self.setMinimumWidth(self.sizeHint().width())
        self.update()

    def text(self) -> str:
        return self._text

    def set_tone(self, tone: str) -> None:
        self._tone = tone if tone in self.TONES else "gray"
        self.update()

    def sizeHint(self) -> QSize:
        metrics = self.fontMetrics()
        return QSize(metrics.horizontalAdvance(self._text) + 20, 22)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        background = (
            theme.rgba(self._tone if self._tone != "gray" else "secondary", 26)
            if self._tone != "gray"
            else theme.rgba("secondary", 22)
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(background)
        painter.drawRoundedRect(QRectF(0, 1, self.width(), self.height() - 2), 9, 9)
        color = (
            getattr(theme.palette, self._tone)
            if self._tone != "gray"
            else theme.palette.secondary
        )
        painter.setPen(QColor(color))
        font = self.font()
        font.setPixelSize(11)
        font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._text)


class StatTile(QFrame):
    def __init__(
        self,
        title: str,
        value: str = "—",
        subtitle: str = "",
        tone: str = "accent",
        icon_name: str = "activity",
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setObjectName("InsetCard")
        self._tone = tone
        self._icon_name = icon_name
        self._title = title
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(6)
        top = QHBoxLayout()
        top.setSpacing(8)
        self.icon_label = QLabel()
        self.icon_label.setFixedSize(26, 26)
        top.addWidget(self.icon_label)
        self.title_label = label(title, "muted")
        top.addWidget(self.title_label, 1)
        layout.addLayout(top)
        self.value_label = label(value, "value")
        layout.addWidget(self.value_label)
        self.sub_label = label(subtitle, "tiny")
        self.sub_label.setWordWrap(True)
        layout.addWidget(self.sub_label)
        self._refresh_icon()

    def _refresh_icon(self) -> None:
        self.icon_label.setPixmap(make_icon_badge(self._icon_name, self._tone, 26))

    def set_value(self, value: str, subtitle: str | None = None) -> None:
        self.value_label.setText(value)
        if subtitle is not None:
            self.sub_label.setText(subtitle)

    def set_title(self, text: str) -> None:
        self.title_label.setText(text)

    def set_tone(self, tone: str, icon_name: str | None = None) -> None:
        self._tone = tone
        if icon_name:
            self._icon_name = icon_name
        self._refresh_icon()

    def refresh_theme(self) -> None:
        self._refresh_icon()


# ---------------------------------------------------------------------------
# 交互控件
# ---------------------------------------------------------------------------


class Switch(QAbstractButton):
    """iOS 胶囊开关。"""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(44, 26)
        self._offset = 0.0
        self._animation = QPropertyAnimation(self, b"offset", self)
        self._animation.setDuration(170)
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.toggled.connect(self._animate)
        theme.changed.connect(self.update)

    def _get_offset(self) -> float:
        return self._offset

    def _set_offset(self, value: float) -> None:
        self._offset = float(value)
        self.update()

    offset = Property(float, _get_offset, _set_offset)

    def _animate(self, checked: bool) -> None:
        self._animation.stop()
        self._animation.setStartValue(self._offset)
        self._animation.setEndValue(1.0 if checked else 0.0)
        self._animation.start()

    def setChecked(self, checked: bool) -> None:  # noqa: N802
        super().setChecked(checked)
        self._offset = 1.0 if checked else 0.0
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        track = QRectF(1, 2, self.width() - 2, self.height() - 4)
        if self.isChecked():
            painter.setBrush(QColor(theme.palette.accent))
            painter.setPen(Qt.PenStyle.NoPen)
        else:
            painter.setBrush(QColor(theme.palette.field))
            painter.setPen(QPen(QColor(theme.palette.border), 1))
        painter.drawRoundedRect(track, track.height() / 2, track.height() / 2)
        radius = (self.height() - 8) / 2
        cx = 3 + radius + self._offset * (self.width() - 6 - 2 * radius)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#FFFFFF"))
        painter.drawEllipse(QRectF(cx - radius, self.height() / 2 - radius, radius * 2, radius * 2))


class SegmentedControl(QFrame):
    changed = Signal(int)

    def __init__(self, items: list[str], current: int = 0, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("Segmented")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(2)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: list[QPushButton] = []
        for index, text in enumerate(items):
            button = QPushButton(text)
            button.setProperty("variant", "segment")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            button.setMinimumHeight(27)
            self._group.addButton(button, index)
            layout.addWidget(button)
            self._buttons.append(button)
        self._group.idClicked.connect(self._on_clicked)
        if self._buttons:
            self._buttons[max(0, min(current, len(self._buttons) - 1))].setChecked(True)

    def _on_clicked(self, index: int) -> None:
        self.changed.emit(index)

    def current_index(self) -> int:
        return self._group.checkedId()

    def set_current_index(self, index: int, emit: bool = False) -> None:
        if 0 <= index < len(self._buttons):
            self._buttons[index].setChecked(True)
            if emit:
                self.changed.emit(index)


class NavButton(QAbstractButton):
    """侧边栏导航项（自绘图标 + 文字）。"""

    def __init__(
        self, text: str, icon_name: str, shortcut_hint: str = "", parent: QWidget | None = None
    ):
        super().__init__(parent)
        self.setText(text)
        self.icon_name = icon_name
        self.shortcut_hint = shortcut_hint
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(42)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._hover = False
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        theme.changed.connect(self.update)

    def enterEvent(self, event: QEvent) -> None:  # noqa: N802
        self._hover = True
        self.update()

    def leaveEvent(self, event: QEvent) -> None:  # noqa: N802
        self._hover = False
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(190, 42)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(0, 0, self.width(), self.height())
        checked = self.isChecked()
        if checked:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(theme.rgba("accent", 32))
            painter.drawRoundedRect(rect.adjusted(0, 1, -2, -1), 10, 10)
        elif self._hover:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(theme.rgba("text", 12))
            painter.drawRoundedRect(rect.adjusted(0, 1, -2, -1), 10, 10)

        color = theme.palette.accent if checked else theme.palette.secondary
        painter.drawPixmap(13, int((self.height() - 17) / 2), icon_pixmap(self.icon_name, color, 17))
        font = self.font()
        font.setPixelSize(13)
        font.setWeight(QFont.Weight.DemiBold if checked else QFont.Weight.Medium)
        painter.setFont(font)
        painter.setPen(QColor(theme.palette.accent if checked else theme.palette.text))
        painter.drawText(
            QRectF(42, 0, self.width() - 58, self.height()),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            self.text(),
        )
        if checked:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(theme.palette.accent))
            painter.drawEllipse(QRectF(self.width() - 15, self.height() / 2 - 2.5, 5, 5))


class ChevronComboBox(QComboBox):
    """带自绘 chevron 的下拉框，保证深浅色主题下箭头都清晰可见。"""

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor(theme.palette.secondary), 1.6)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        cx = self.width() - 14.0
        cy = self.height() / 2.0
        painter.drawLine(QPointF(cx - 3.6, cy - 1.8), QPointF(cx, cy + 1.8))
        painter.drawLine(QPointF(cx, cy + 1.8), QPointF(cx + 3.6, cy - 1.8))


class WindowDot(QAbstractButton):
    """窗口左上角的红黄绿控制点。"""

    SYMBOLS = {"close": "×", "minimize": "–", "maximize": "+"}

    def __init__(self, color: str, action: str, parent: QWidget | None = None):
        super().__init__(parent)
        self._color = color
        self._action = action
        self.setFixedSize(16, 16)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._hover = False
        self.setToolTip({"close": "关闭", "minimize": "最小化", "maximize": "最大化"}[action])
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

    def enterEvent(self, event: QEvent) -> None:  # noqa: N802
        self._hover = True
        self.update()

    def leaveEvent(self, event: QEvent) -> None:  # noqa: N802
        self._hover = False
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(self._color))
        painter.drawEllipse(QRectF(2, 2, 12, 12))
        if self._hover:
            painter.setPen(QColor(0, 0, 0, 150))
            font = painter.font()
            font.setPixelSize(10)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.SYMBOLS[self._action])


class DropZone(QFrame):
    clicked = Signal()
    fileDropped = Signal(str)

    def __init__(
        self,
        title: str = "拖入 CSV / Excel 文件",
        subtitle: str = "或点击选择文件",
        icon_name: str = "upload",
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setObjectName("DropZone")
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(96)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(4)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.icon_label = QLabel()
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.icon_label)
        self.title_label = label(title, "card-title")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title_label)
        self.subtitle_label = label(subtitle, "muted")
        self.subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.subtitle_label)
        self._icon_name = icon_name
        self._dragging = False
        self.refresh_theme()

    def set_text(self, title: str, subtitle: str = "") -> None:
        self.title_label.setText(title)
        self.subtitle_label.setText(subtitle)
        self.subtitle_label.setVisible(bool(subtitle))

    def refresh_theme(self) -> None:
        self.icon_label.setPixmap(icon_pixmap(self._icon_name, theme.palette.accent, 26))

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._dragging = True
            self.setProperty("dragging", True)
            repolish(self)
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:  # noqa: N802
        self._dragging = False
        self.setProperty("dragging", False)
        repolish(self)
        super().dragLeaveEvent(event)

    def dropEvent(self, event) -> None:  # noqa: N802
        self._dragging = False
        self.setProperty("dragging", False)
        repolish(self)
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path:
                self.fileDropped.emit(path)
                event.acceptProposedAction()
                return
        event.ignore()


class DataTable(QTableWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.setAlternatingRowColors(False)
        self.setShowGrid(False)
        self.setWordWrap(False)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.verticalHeader().setVisible(False)
        self.verticalHeader().setDefaultSectionSize(30)
        self.horizontalHeader().setHighlightSections(False)
        self.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.horizontalHeader().setMinimumSectionSize(70)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.viewport().setAutoFillBackground(False)
        self.setHorizontalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)
        self.setVerticalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)

    def load_dataframe(
        self,
        frame,
        max_rows: int = 200,
        formatter: Callable[[str, Any], str] | None = None,
    ) -> None:
        columns = [str(c) for c in frame.columns]
        subset = frame.head(max_rows)
        self.clear()
        self.setColumnCount(len(columns))
        self.setHorizontalHeaderLabels(columns)
        self.setRowCount(len(subset))
        for row_index, (_, row) in enumerate(subset.iterrows()):
            for col_index, column in enumerate(frame.columns):
                value = row[column]
                if formatter is not None:
                    text = formatter(str(column), value)
                else:
                    text = _format_cell(value)
                item = QTableWidgetItem(text)
                item.setToolTip(text)
                self.setItem(row_index, col_index, item)
        self.resizeRowsToContents()
        for row_index in range(self.rowCount()):
            self.setRowHeight(row_index, max(28, min(36, self.rowHeight(row_index))))

    def set_message(self, text: str, rows: int = 3, columns: int = 1) -> None:
        self.clear()
        self.setColumnCount(columns)
        self.setRowCount(rows)
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setItem(0, 0, item)
        self.setSpan(0, 0, rows, columns)


def _format_cell(value: Any) -> str:
    if value is None:
        return "—"
    try:
        import pandas as pd

        if pd.isna(value):
            return "—"
    except (TypeError, ValueError):
        pass
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        if abs(value) >= 1000:
            return f"{value:,.2f}"
        return f"{value:.4g}"
    return str(value)


# ---------------------------------------------------------------------------
# 轻提示
# ---------------------------------------------------------------------------

_active_toasts: list["Toast"] = []


class Toast(QFrame):
    def __init__(self, parent: QWidget, text: str, tone: str = "success"):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setStyleSheet(
            f"QFrame {{ background: {theme.palette.card}; border: 1px solid {theme.palette.border};"
            f" border-radius: 12px; }}"
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 9, 14, 9)
        layout.setSpacing(8)
        icon_label = QLabel()
        color = {
            "success": theme.palette.green,
            "error": theme.palette.red,
            "info": theme.palette.accent,
            "warning": theme.palette.orange,
        }.get(tone, theme.palette.accent)
        icon_label.setPixmap(icon_pixmap("check" if tone == "success" else "info", color, 16))
        layout.addWidget(icon_label)
        text_label = QLabel(text)
        text_label.setStyleSheet(f"color: {theme.palette.text}; font-size: 12px; font-weight: 500;")
        layout.addWidget(text_label)
        self.adjustSize()
        self._effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._effect)
        self._effect.setOpacity(0.0)
        self._closed = False

    @classmethod
    def show_message(cls, parent: QWidget, text: str, tone: str = "success", duration: int = 2200) -> None:
        window = parent.window()
        for old in list(_active_toasts):
            old._cleanup()
        toast = cls(window, text, tone)
        toast.adjustSize()
        x = max(20, (window.width() - toast.width()) // 2)
        y = 10  # 顶部标题栏空白区域，不遮挡页面操作
        toast.move(x, y)
        toast.raise_()
        toast.show()
        _active_toasts.append(toast)
        fade_in = QPropertyAnimation(toast._effect, b"opacity", toast)
        fade_in.setDuration(160)
        fade_in.setStartValue(0.0)
        fade_in.setEndValue(1.0)
        fade_in.start()
        toast._fade_in = fade_in  # type: ignore[attr-defined]

        def fade_out() -> None:
            if toast._closed:
                return
            animation = QPropertyAnimation(toast._effect, b"opacity", toast)
            animation.setDuration(220)
            animation.setStartValue(1.0)
            animation.setEndValue(0.0)
            animation.setEasingCurve(QEasingCurve.Type.InCubic)
            animation.finished.connect(toast._cleanup)
            animation.start()
            toast._fade_out = animation  # type: ignore[attr-defined]

        QTimer.singleShot(duration, fade_out)

    def _cleanup(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self in _active_toasts:
            _active_toasts.remove(self)
        self.hide()
        self.deleteLater()


# ---------------------------------------------------------------------------
# 表单辅助
# ---------------------------------------------------------------------------


def form_row(
    grid: QGridLayout,
    row: int,
    text: str,
    control: QWidget,
    hint: str = "",
) -> QWidget:
    """在 :class:`QGridLayout` 中放入一行「标签 + 控件」。"""
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(10)
    title = label(text, "muted")
    title.setMinimumWidth(92)
    title.setMaximumWidth(112)
    layout.addWidget(title, 0, Qt.AlignmentFlag.AlignVCenter)
    layout.addWidget(control, 1)
    if hint:
        hint_label = label(hint, "tiny")
        hint_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(hint_label, 0)
    grid.addWidget(container, row, 0)
    return container


def h_spacer() -> QWidget:
    spacer = QWidget()
    spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
    return spacer
