"""主窗口：无边框圆角窗口、侧边导航与页面调度。"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from PySide6.QtCore import QEvent, QPoint, QRectF, QSize, Qt, QTimer
from PySide6.QtGui import (
    QColor,
    QFont,
    QIcon,
    QKeySequence,
    QLinearGradient,
    QPainter,
    QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSizeGrip,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..core.model import ModelConfig
from ..state import AppState
from .help import show_help
from .icons import icon, icon_pixmap
from .pages.data_page import DataPage
from .pages.model_page import ModelPage
from .pages.test_page import TestPage
from .pages.train_page import TrainPage
from .theme import theme
from .charts import GradientLogo
from .widgets import (
    InsetCard,
    NavButton,
    Pill,
    SectionLabel,
    Toast,
    WindowDot,
    label,
    icon_button,
    make_icon_badge,
    refresh_theme_recursive,
    repolish,
)

APP_NAME = "NeuroStudio"
APP_SUBTITLE = "神经网络工作台"
WINDOW_RADIUS = 20


class DragArea(QWidget):
    """可拖动窗口的空白标题区域。"""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._press_global: QPoint | None = None
        self._window_pos: QPoint | None = None
        self.setMouseTracking(True)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_global = event.globalPosition().toPoint()
            self._window_pos = self.window().frameGeometry().topLeft()
            handle = self.window().windowHandle()
            if handle is not None:
                handle.startSystemMove()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if (
            self._press_global is not None
            and self._window_pos is not None
            and event.buttons() & Qt.MouseButton.LeftButton
            and not self.window().isMaximized()
        ):
            delta = event.globalPosition().toPoint() - self._press_global
            self.window().move(self._window_pos + delta)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._press_global = None
        self._window_pos = None

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            window = self.window()
            if hasattr(window, "toggle_maximize"):
                window.toggle_maximize()


class TitleBar(DragArea):
    def __init__(
        self,
        theme_button: QWidget,
        help_button: QWidget,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setFixedHeight(50)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(26, 0, 18, 0)
        layout.setSpacing(8)
        layout.addStretch(1)
        hint = label("本地运行 · 数据不离开设备", "tiny")
        layout.addWidget(hint)
        layout.addWidget(help_button)
        layout.addWidget(theme_button)


class Sidebar(QFrame):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("Sidebar")
        self.setFixedWidth(242)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(4)

        self.drag_area = DragArea()
        self.drag_area.setFixedHeight(22)
        dots = QHBoxLayout(self.drag_area)
        dots.setContentsMargins(2, 2, 0, 2)
        dots.setSpacing(8)
        self.close_dot = WindowDot("#FF5F57", "close")
        self.min_dot = WindowDot("#FEBC2E", "minimize")
        self.max_dot = WindowDot("#28C840", "maximize")
        dots.addWidget(self.close_dot)
        dots.addWidget(self.min_dot)
        dots.addWidget(self.max_dot)
        dots.addStretch(1)
        layout.addWidget(self.drag_area)

        logo_row = QWidget()
        logo_layout = QHBoxLayout(logo_row)
        logo_layout.setContentsMargins(2, 6, 0, 0)
        logo_layout.setSpacing(10)
        self.logo = GradientLogo(36)
        logo_layout.addWidget(self.logo)
        logo_text = QVBoxLayout()
        logo_text.setSpacing(0)
        title = label(APP_NAME, "card-title")
        title_font = QFont(title.font())
        title_font.setPixelSize(16)
        title_font.setWeight(QFont.Weight.Bold)
        title.setFont(title_font)
        logo_text.addWidget(title)
        logo_text.addWidget(label(APP_SUBTITLE, "tiny"))
        logo_layout.addLayout(logo_text, 1)
        layout.addWidget(logo_row)
        layout.addSpacing(22)

        layout.addWidget(SectionLabel("工作区"))
        layout.addSpacing(4)
        nav_items = [
            ("数据", "database"),
            ("模型", "layers"),
            ("训练", "activity"),
            ("评估", "target"),
        ]
        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        self.nav_buttons: list[NavButton] = []
        for index, (text, icon_name) in enumerate(nav_items):
            button = NavButton(text, icon_name, f"Ctrl+{index + 1}")
            self.nav_group.addButton(button, index)
            layout.addWidget(button)
            self.nav_buttons.append(button)
        layout.addSpacing(14)

        model_row = QWidget()
        model_layout = QHBoxLayout(model_row)
        model_layout.setContentsMargins(6, 0, 2, 0)
        model_layout.setSpacing(6)
        model_layout.addWidget(label("当前模型", "tiny"))
        model_layout.addStretch(1)
        self.model_pill = Pill("尚无模型", "gray")
        model_layout.addWidget(self.model_pill)
        layout.addWidget(model_row)

        layout.addStretch(1)

        self.device_card = InsetCard(padding=11, spacing=6)
        device_top = QHBoxLayout()
        device_top.setSpacing(9)
        self.device_icon = QLabel()
        self.device_icon.setFixedSize(28, 28)
        device_top.addWidget(self.device_icon)
        device_text = QVBoxLayout()
        device_text.setSpacing(0)
        self.device_title = label("计算设备", "card-title")
        self.device_title_font = QFont(self.device_title.font())
        self.device_title_font.setPixelSize(12)
        self.device_title_font.setWeight(QFont.Weight.DemiBold)
        self.device_title.setFont(self.device_title_font)
        device_text.addWidget(self.device_title)
        self.device_sub = label("", "tiny")
        self.device_sub.setWordWrap(False)
        device_text.addWidget(self.device_sub)
        device_top.addLayout(device_text, 1)
        self.device_card.body.addLayout(device_top)
        self.torch_label = label("", "tiny")
        self.device_card.body.addWidget(self.torch_label)
        layout.addWidget(self.device_card)
        layout.addSpacing(4)
        self.version_label = label("NeuroStudio v1.0.0", "tiny")
        self.version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.version_label)

    def refresh_theme(self) -> None:
        self.logo.update()
        self.device_icon.setPixmap(make_icon_badge("cpu", "green" if torch.cuda.is_available() else "gray", 28))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        _warmup_torch()
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(app_icon())
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setMinimumSize(1180, 780)
        self._margin = 10
        self.state = AppState()
        self._build_ui()
        self._apply_device_info()
        self._center()
        theme.changed.connect(self._on_theme_changed)
        self.state.active_model_changed.connect(self._on_active_model_changed)
        self._on_theme_changed()
        self._navigate(0)

    # -- 构建 ---------------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("WindowRoot")
        self.setCentralWidget(central)
        self.outer_layout = QVBoxLayout(central)
        self.outer_layout.setContentsMargins(self._margin, self._margin, self._margin, self._margin)
        self.outer_layout.setSpacing(0)

        self.shell = QFrame()
        self.shell.setObjectName("WindowShell")
        self.shell.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.outer_layout.addWidget(self.shell)
        shell_layout = QHBoxLayout(self.shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)

        self.sidebar = Sidebar()
        self.sidebar.close_dot.clicked.connect(self.close)
        self.sidebar.min_dot.clicked.connect(self.showMinimized)
        self.sidebar.max_dot.clicked.connect(self.toggle_maximize)
        shell_layout.addWidget(self.sidebar)

        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(0)
        shell_layout.addLayout(right, 1)

        self.theme_btn = self._make_theme_button()
        self.help_btn = icon_button("info", "参数说明", 30)
        self.help_btn.clicked.connect(lambda: show_help("index", self))
        self.title_bar = TitleBar(self.theme_btn, self.help_btn)
        right.addWidget(self.title_bar)

        page_host = QWidget()
        page_layout = QVBoxLayout(page_host)
        page_layout.setContentsMargins(24, 6, 24, 18)
        page_layout.setSpacing(0)
        self.stack = QStackedWidget()
        page_layout.addWidget(self.stack)
        right.addWidget(page_host, 1)

        self.data_page = DataPage(self.state)
        self.model_page = ModelPage(self.state)
        self.train_page = TrainPage(self.state)
        self.test_page = TestPage(self.state)
        self.pages = [self.data_page, self.model_page, self.train_page, self.test_page]
        for page in self.pages:
            self.stack.addWidget(page)

        self.data_page.continue_requested.connect(lambda: self._navigate(1))
        self.model_page.continue_requested.connect(lambda: self._navigate(2))
        self.model_page.back_requested.connect(lambda: self._navigate(0))
        self.train_page.evaluate_requested.connect(lambda: self._navigate(3))
        self.sidebar.nav_group.idClicked.connect(self._navigate)

        self.size_grip = QSizeGrip(self.shell)
        self.size_grip.setFixedSize(16, 16)
        self.size_grip.raise_()

        self._create_shortcuts()

    def _make_theme_button(self):
        button = icon_button("moon", "切换深色 / 浅色主题", 30)
        button.clicked.connect(theme.toggle)
        return button

    def _create_shortcuts(self) -> None:
        for index in range(4):
            shortcut = QShortcut(QKeySequence(f"Ctrl+{index + 1}"), self)
            shortcut.activated.connect(lambda i=index: self._navigate(i))
        open_shortcut = QShortcut(QKeySequence("Ctrl+O"), self)
        open_shortcut.activated.connect(self._open_file)
        save_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        save_shortcut.activated.connect(self._export_model)
        theme_shortcut = QShortcut(QKeySequence("Ctrl+D"), self)
        theme_shortcut.activated.connect(theme.toggle)

    # -- 导航 ---------------------------------------------------------------
    def _navigate(self, index: int) -> None:
        if not 0 <= index < len(self.pages):
            return
        if index == 1 and self.state.frame is not None and self.state.prepared is None:
            try:
                self.state.prepare_current()
            except Exception:  # noqa: BLE001
                pass
        self.stack.setCurrentIndex(index)
        if index < len(self.sidebar.nav_buttons):
            self.sidebar.nav_buttons[index].setChecked(True)
        page = self.pages[index]
        on_show = getattr(page, "on_show", None)
        if callable(on_show):
            on_show()

    def _open_file(self) -> None:
        self._navigate(0)
        self.data_page._choose_file()

    def _export_model(self) -> None:
        if self.state.active is None:
            Toast.show_message(self, "尚无可用模型，请先训练或加载模型包", "warning")
            return
        self.train_page._export_model()

    # -- 设备与状态 ---------------------------------------------------------
    def _apply_device_info(self) -> None:
        cuda = torch.cuda.is_available()
        if cuda:
            try:
                gpu_name = torch.cuda.get_device_name(0)
            except Exception:  # noqa: BLE001
                gpu_name = "CUDA GPU"
            self.sidebar.device_title.setText("CUDA 就绪")
            self.sidebar.device_title.setStyleSheet(f"color: {theme.palette.green}; font-weight: 600;")
            if len(gpu_name) > 26:
                gpu_name = gpu_name[:25] + "…"
            self.sidebar.device_sub.setText(gpu_name)
            self.sidebar.device_sub.setToolTip(torch.cuda.get_device_name(0))
            self.sidebar.device_icon.setPixmap(make_icon_badge("cpu", "green", 28))
        else:
            self.sidebar.device_title.setText("CPU 模式")
            self.sidebar.device_title.setStyleSheet(f"color: {theme.palette.secondary}; font-weight: 600;")
            self.sidebar.device_sub.setText("未检测到可用 CUDA")
            self.sidebar.device_icon.setPixmap(make_icon_badge("cpu", "gray", 28))
        self.sidebar.torch_label.setText(f"PyTorch {torch.__version__}")
        self.sidebar.torch_label.setToolTip(f"Python {sys.version.split()[0]}")

    def _on_active_model_changed(self) -> None:
        active = self.state.active
        if active is None:
            self.sidebar.model_pill.setText("尚无模型")
            self.sidebar.model_pill.set_tone("gray")
            self.sidebar.model_pill.setToolTip("")
            return
        summary = active.summary
        self.sidebar.model_pill.setText("模型已就绪")
        self.sidebar.model_pill.set_tone("green")
        task = "分类" if active.task == "classification" else "回归"
        self.sidebar.model_pill.setToolTip(
            f"{task} · {summary['architecture']} · 标签 {summary['label']}"
        )
        self.train_page.export_btn.setEnabled(True)

    # -- 窗口行为 -----------------------------------------------------------
    def toggle_maximize(self) -> None:
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def _apply_window_state(self) -> None:
        maxed = self.isMaximized() or self.isFullScreen()
        self._margin = 0 if maxed else 10
        self.outer_layout.setContentsMargins(self._margin, self._margin, self._margin, self._margin)
        for widget in (self.shell, self.sidebar):
            widget.setProperty("maximized", "true" if maxed else "false")
            repolish(widget)
        self.size_grip.setVisible(not maxed)
        self.sidebar.max_dot.setToolTip("还原" if maxed else "最大化")
        self._place_size_grip()

    def changeEvent(self, event) -> None:  # noqa: N802
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange:
            QTimer.singleShot(0, self._apply_window_state)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        QTimer.singleShot(0, self._place_size_grip)

    def _place_size_grip(self) -> None:
        grip = self.size_grip
        grip.move(self.shell.width() - grip.width() - 3, self.shell.height() - grip.height() - 3)
        grip.raise_()

    def _center(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(1380, 880)
            return
        available = screen.availableGeometry()
        width = min(1440, max(1180, int(available.width() * 0.9)))
        height = min(940, max(780, int(available.height() * 0.9)))
        self.resize(width, height)
        self.move(
            available.left() + (available.width() - width) // 2,
            available.top() + (available.height() - height) // 2,
        )

    def _on_theme_changed(self) -> None:
        dark = theme.is_dark
        self.theme_btn._icon_name = "sun" if dark else "moon"
        self.theme_btn._icon_tone = "secondary"
        self.theme_btn.setIcon(icon("sun" if dark else "moon", theme.palette.secondary, 17))
        refresh_theme_recursive(self)
        self._apply_device_info()
        self.shell.update()

    def closeEvent(self, event) -> None:  # noqa: N802
        self.train_page.shutdown()
        super().closeEvent(event)


def _warmup_torch() -> None:
    """预热优化器初始化。

    torch 2.5 的 torch.optim.* 会延迟导入 torch._dynamo；首次导入在 QThread
    中可能因 importlib/文件系统开销长时间阻塞。在主线程提前完成一次。
    """
    try:
        torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    except Exception:  # noqa: BLE001
        pass


def app_icon(size: int = 128) -> QIcon:
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    gradient = QLinearGradient(0, 0, size, size)
    gradient.setColorAt(0.0, QColor("#0A84FF"))
    gradient.setColorAt(1.0, QColor("#AF52DE"))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(gradient)
    painter.drawRoundedRect(QRectF(0, 0, size, size), size * 0.28, size * 0.28)
    scale = size / 36.0
    columns = [(11, [13, 23]), (18, [9, 18, 27]), (25, [13, 23])]
    painter.setPen(QColor(255, 255, 255, 90))
    for (x1, ys1), (x2, ys2) in zip(columns, columns[1:]):
        for y1 in ys1:
            for y2 in ys2:
                painter.drawLine(
                    int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)
                )
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(255, 255, 255, 240))
    for x, ys in columns:
        for y in ys:
            painter.drawEllipse(
                QRectF(x * scale - 2 * scale, y * scale - 2 * scale, 4 * scale, 4 * scale)
            )
    painter.end()
    return QIcon(pixmap)
