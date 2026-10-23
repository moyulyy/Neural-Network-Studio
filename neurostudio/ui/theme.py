"""iOS 风格设计令牌与全局样式表。"""

from __future__ import annotations

from dataclasses import dataclass
from string import Template

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QColor, QFont, QFontDatabase
from PySide6.QtWidgets import QApplication


@dataclass(frozen=True)
class Palette:
    mode: str
    window: str
    sidebar: str
    card: str
    card_alt: str
    field: str
    text: str
    secondary: str
    tertiary: str
    separator: str
    border: str
    accent: str
    accent_hover: str
    accent_soft: str
    green: str
    green_soft: str
    orange: str
    orange_soft: str
    red: str
    red_soft: str
    purple: str
    purple_soft: str
    teal: str
    teal_soft: str
    grid: str
    shadow_alpha: int
    glow_accent: str
    glow_purple: str


LIGHT = Palette(
    mode="light",
    window="#F2F2F7",
    sidebar="#FFFFFF",
    card="#FFFFFF",
    card_alt="#F5F5F9",
    field="#F2F2F7",
    text="#1C1C1E",
    secondary="#6E6E73",
    tertiary="#A1A1A6",
    separator="#E5E5EA",
    border="#E8E8ED",
    accent="#007AFF",
    accent_hover="#0068DB",
    accent_soft="#E7F1FF",
    green="#34C759",
    green_soft="#E9F9EE",
    orange="#FF9500",
    orange_soft="#FFF5E6",
    red="#FF3B30",
    red_soft="#FFEDEC",
    purple="#AF52DE",
    purple_soft="#F8ECFD",
    teal="#30B0C7",
    teal_soft="#E6F7FA",
    grid="#E9E9EF",
    shadow_alpha=38,
    glow_accent="#007AFF",
    glow_purple="#AF52DE",
)

DARK = Palette(
    mode="dark",
    window="#0A0A0C",
    sidebar="#161618",
    card="#1C1C1E",
    card_alt="#2C2C2E",
    field="#2C2C2E",
    text="#FFFFFF",
    secondary="#A1A1A6",
    tertiary="#6E6E73",
    separator="#2C2C30",
    border="#2E2E32",
    accent="#0A84FF",
    accent_hover="#3E9DFF",
    accent_soft="#0C2A4A",
    green="#30D158",
    green_soft="#11301B",
    orange="#FF9F0A",
    orange_soft="#3A2A10",
    red="#FF453A",
    red_soft="#3D1B19",
    purple="#BF5AF2",
    purple_soft="#2E1B3A",
    teal="#40C8E0",
    teal_soft="#10333A",
    grid="#2A2A2E",
    shadow_alpha=110,
    glow_accent="#0A84FF",
    glow_purple="#BF5AF2",
)


def _font_family() -> str:
    available = set(QFontDatabase.families())
    for name in (
        "SF Pro Display",
        "SF Pro Text",
        "Segoe UI Variable Text",
        "Segoe UI",
        "PingFang SC",
        "Microsoft YaHei UI",
        "Microsoft YaHei",
        "Arial",
    ):
        if name in available:
            return f'"{name}", "Segoe UI", "Microsoft YaHei UI", sans-serif'
    return '"Segoe UI", "Microsoft YaHei UI", sans-serif'


def _rgba(hex_color: str, alpha: int) -> str:
    color = QColor(hex_color)
    return f"rgba({color.red()}, {color.green()}, {color.blue()}, {alpha})"


class ThemeManager(QObject):
    changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.mode = "light"
        self.palette = LIGHT
        # 延迟到 QApplication 创建后再查询字体，避免在导入阶段访问 QFontDatabase
        self._font_family: str | None = None

    # -- 状态 ---------------------------------------------------------------
    def set_mode(self, mode: str, *, notify: bool = True) -> None:
        self.mode = "dark" if mode == "dark" else "light"
        self.palette = DARK if self.mode == "dark" else LIGHT
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(self.qss())
        if notify:
            self.changed.emit()

    def toggle(self) -> None:
        self.set_mode("dark" if self.mode == "light" else "light")

    def color(self, name: str) -> QColor:
        return QColor(getattr(self.palette, name))

    def rgba(self, name: str, alpha: int) -> QColor:
        color = self.color(name)
        color.setAlpha(alpha)
        return color

    @property
    def is_dark(self) -> bool:
        return self.mode == "dark"

    # -- 样式表 -------------------------------------------------------------
    def qss(self) -> str:
        p = self.palette
        hover_bg = _rgba(p.text, 14 if p.mode == "light" else 26)
        soft_sep = _rgba(p.separator, 140)
        shadow = _rgba("#000000", p.shadow_alpha)
        selection = _rgba(p.accent, 46)
        qss = Template(
            """
* {
    font-family: $font_family;
    font-size: 13px;
    color: $text;
    outline: 0;
}
QWidget {
    background: transparent;
}
QMainWindow, QDialog {
    background: $window;
}
QLabel {
    background: transparent;
}
QLabel[role="title"] { font-size: 29px; font-weight: 700; letter-spacing: -0.4px; color: $text; }
QLabel[role="subtitle"] { font-size: 13px; color: $secondary; }
QLabel[role="card-title"] { font-size: 15px; font-weight: 600; color: $text; }
QLabel[role="card-desc"] { font-size: 12px; color: $secondary; }
QLabel[role="muted"] { color: $secondary; font-size: 12px; }
QLabel[role="tiny"] { color: $tertiary; font-size: 11px; }
QLabel[role="value"] { font-size: 25px; font-weight: 700; letter-spacing: -0.6px; color: $text; }
QLabel[role="section"] { font-size: 11px; font-weight: 700; color: $tertiary; letter-spacing: 0.6px; }
QLabel[role="mono"] { font-family: "Cascadia Mono", "Consolas", monospace; font-size: 12px; color: $text; }

QFrame#Card {
    background: $card;
    border: 1px solid $border;
    border-radius: 18px;
}
QFrame#InsetCard {
    background: $card_alt;
    border: 1px solid $border;
    border-radius: 13px;
}
QFrame#DropZone {
    background: $card_alt;
    border: 1.5px dashed $border;
    border-radius: 14px;
}
QFrame#DropZone[dragging="true"] {
    background: $accent_soft;
    border: 1.5px solid $accent;
}
QFrame#Separator { background: $separator; border: none; max-height: 1px; min-height: 1px; }
QFrame#Sidebar {
    background: $sidebar;
    border: none;
    border-top-left-radius: 19px;
    border-bottom-left-radius: 19px;
}
QFrame#Sidebar[maximized="true"] { border-top-left-radius: 0; border-bottom-left-radius: 0; }
QFrame#WindowShell {
    background: $window;
    border: 1px solid $border;
    border-radius: 20px;
}
QFrame#WindowShell[maximized="true"] {
    border-radius: 0;
    border: 1px solid transparent;
}

QSlider::groove:horizontal {
    height: 4px;
    background: $field;
    border-radius: 2px;
}
QSlider::sub-page:horizontal {
    background: $accent;
    border-radius: 2px;
}
QSlider::add-page:horizontal {
    background: $field;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #FFFFFF;
    border: 1px solid $border;
    width: 16px;
    height: 16px;
    margin: -7px 0;
    border-radius: 9px;
}
QSlider::handle:horizontal:hover { border-color: $accent; }

QPushButton {
    background: $card;
    border: 1px solid $border;
    border-radius: 10px;
    padding: 7px 14px;
    font-size: 13px;
    font-weight: 500;
    color: $text;
}
QPushButton:hover { background: $card_alt; }
QPushButton:pressed { background: $field; }
QPushButton:disabled { color: $tertiary; background: $card_alt; border-color: $separator; }
QPushButton[variant="primary"] {
    background: $accent;
    border: 1px solid $accent;
    color: #FFFFFF;
    font-weight: 600;
    padding: 8px 16px;
}
QPushButton[variant="primary"]:hover { background: $accent_hover; border-color: $accent_hover; }
QPushButton[variant="primary"]:pressed { background: $accent_hover; }
QPushButton[variant="primary"]:disabled { background: $accent_soft; border-color: $accent_soft; color: $tertiary; }
QPushButton[variant="quiet"] {
    background: transparent;
    border: none;
    color: $accent;
    font-weight: 500;
    padding: 6px 8px;
}
QPushButton[variant="quiet"]:hover { background: $accent_soft; }
QPushButton[variant="danger"] {
    background: $red_soft;
    border: 1px solid transparent;
    color: $red;
    font-weight: 600;
}
QPushButton[variant="danger"]:hover { border-color: $red; }
QPushButton[variant="danger"]:disabled {
    background: $card_alt;
    color: $tertiary;
    border-color: $separator;
}
QPushButton[variant="segment"] {
    background: transparent;
    border: none;
    border-radius: 7px;
    padding: 5px 14px;
    color: $secondary;
    font-weight: 500;
}
QPushButton[variant="segment"]:hover { color: $text; }
QPushButton[variant="segment"]:checked {
    background: $card;
    color: $text;
    font-weight: 600;
}
QFrame#Segmented {
    background: $field;
    border: 1px solid $border;
    border-radius: 10px;
}
QPushButton[variant="chip"] {
    background: $card_alt;
    border: 1px solid $border;
    border-radius: 13px;
    padding: 4px 11px;
    font-size: 11px;
    font-weight: 600;
    color: $secondary;
}
QPushButton[variant="chip"]:checked {
    background: $accent_soft;
    border-color: $accent_soft;
    color: $accent;
}

QToolButton {
    background: transparent;
    border: none;
    border-radius: 8px;
    padding: 5px;
}
QToolButton:hover { background: $hover_bg; }
QToolButton:pressed { background: $field; }

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background: $field;
    border: 1px solid $border;
    border-radius: 9px;
    padding: 6px 10px;
    min-height: 20px;
    color: $text;
    selection-background-color: $accent;
    selection-color: #FFFFFF;
}
QSpinBox QLineEdit, QDoubleSpinBox QLineEdit {
    background: transparent;
    border: none;
    padding: 0;
    color: $text;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid $accent;
    background: $card;
}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {
    color: $tertiary;
    background: $card_alt;
}
QLineEdit[readOnly="true"] { color: $secondary; }
QComboBox::drop-down { border: none; width: 26px; }
QComboBox::down-arrow { image: none; width: 0px; height: 0px; }
QComboBox QAbstractItemView {
    background: $card;
    border: 1px solid $border;
    border-radius: 10px;
    padding: 4px;
    selection-background-color: $accent_soft;
    selection-color: $text;
    outline: 0;
}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    width: 0px;
    border: none;
}

QCheckBox { spacing: 8px; color: $text; }
QCheckBox::indicator {
    width: 18px; height: 18px;
    border-radius: 5px;
    border: 1.5px solid $tertiary;
    background: $card;
}
QCheckBox::indicator:hover { border-color: $accent; }
QCheckBox::indicator:checked {
    background: $accent;
    border-color: $accent;
    image: none;
}

QTableWidget {
    background: transparent;
    alternate-background-color: $card_alt;
    border: none;
    gridline-color: transparent;
    selection-background-color: $selection;
    selection-color: $text;
    font-size: 12px;
}
QTableWidget::item {
    padding: 5px 8px;
    border: none;
    border-bottom: 1px solid $soft_sep;
}
QTableWidget::item:selected { background: $selection; }
QHeaderView { background: transparent; }
QHeaderView::section {
    background: $card;
    color: $tertiary;
    border: none;
    border-bottom: 1px solid $separator;
    padding: 7px 8px;
    font-size: 11px;
    font-weight: 600;
}
QTableCornerButton::section { background: $card; border: none; }

QListWidget {
    background: transparent;
    border: none;
    outline: 0;
}
QListWidget::item {
    background: $card_alt;
    border-radius: 9px;
    padding: 8px 10px;
    margin: 2px 0;
    color: $text;
}
QListWidget::item:selected { background: $accent_soft; color: $accent; }
QListWidget::item:hover { background: $hover_bg; }

QScrollArea { background: transparent; border: none; }
QScrollArea > QWidget > QWidget { background: transparent; }
QScrollBar:vertical { background: transparent; width: 9px; margin: 2px; }
QScrollBar::handle:vertical {
    background: $scroll_handle;
    min-height: 30px;
    border-radius: 4px;
}
QScrollBar::handle:vertical:hover { background: $tertiary; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
QScrollBar:horizontal { background: transparent; height: 9px; margin: 2px; }
QScrollBar::handle:horizontal {
    background: $scroll_handle;
    min-width: 30px;
    border-radius: 4px;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: transparent; }

QProgressBar {
    background: $field;
    border: none;
    border-radius: 5px;
    height: 10px;
    text-align: center;
    color: transparent;
}
QProgressBar::chunk {
    background: $accent;
    border-radius: 5px;
}

QTextBrowser {
    background: $card;
    border: none;
    border-radius: 10px;
    color: $text;
    selection-background-color: $accent;
    selection-color: #FFFFFF;
}

QToolTip {
    background: $card;
    color: $text;
    border: 1px solid $border;
    border-radius: 8px;
    padding: 6px 9px;
    font-size: 12px;
}

QMenu {
    background: $card;
    border: 1px solid $border;
    border-radius: 10px;
    padding: 5px;
}
QMenu::item { padding: 6px 18px; border-radius: 6px; }
QMenu::item:selected { background: $accent_soft; color: $accent; }

QMessageBox { background: $card; }
QMessageBox QLabel { color: $text; }

QSplitter::handle { background: transparent; width: 1px; }

QSizeGrip { background: transparent; width: 14px; height: 14px; }
"""
        )
        if not self._font_family:
            self._font_family = _font_family()
        return qss.substitute(
            font_family=self._font_family,
            window=p.window,
            sidebar=p.sidebar,
            card=p.card,
            card_alt=p.card_alt,
            field=p.field,
            text=p.text,
            secondary=p.secondary,
            tertiary=p.tertiary,
            separator=p.separator,
            border=p.border,
            accent=p.accent,
            accent_hover=p.accent_hover,
            accent_soft=p.accent_soft,
            green=p.green,
            green_soft=p.green_soft,
            orange=p.orange,
            orange_soft=p.orange_soft,
            red=p.red,
            red_soft=p.red_soft,
            purple=p.purple,
            purple_soft=p.purple_soft,
            teal=p.teal,
            teal_soft=p.teal_soft,
            grid=p.grid,
            hover_bg=hover_bg,
            soft_sep=soft_sep,
            shadow=shadow,
            selection=selection,
            scroll_handle=_rgba(p.text, 44),
        )


theme = ThemeManager()
