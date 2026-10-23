"""自绘图表：训练曲线、散点、混淆矩阵、网络结构与进度环。

全部使用 QPainter 绘制，颜色实时跟随主题，不依赖第三方绘图库。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QRadialGradient,
)
from PySide6.QtWidgets import QSizePolicy, QWidget

from .theme import theme


def _font(widget: QWidget, size: int, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    font = QFont(widget.font())
    font.setPixelSize(size)
    font.setWeight(weight)
    return font


def _fmt_axis(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    if abs(value) == 0:
        return "0"
    if abs(value) >= 0.001:
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return f"{value:.1e}"


def _fmt_value(value: float) -> str:
    if not math.isfinite(value):
        return "—"
    if abs(value) >= 1000:
        return f"{value:,.2f}"
    if abs(value) >= 1:
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return f"{value:.5f}".rstrip("0").rstrip(".")


def _nice_step(span: float, target: int = 4) -> float:
    if span <= 0 or not math.isfinite(span):
        return 1.0
    raw = span / max(target, 1)
    magnitude = 10 ** math.floor(math.log10(raw))
    for multiplier in (1, 2, 2.5, 5, 10):
        if raw <= multiplier * magnitude:
            return multiplier * magnitude
    return 10 * magnitude


@dataclass
class ChartSeries:
    name: str
    values: list[float]
    color: str = "accent"
    fill: bool = False
    dashed: bool = False


class LineChart(QWidget):
    """带平滑曲线、渐变填充、图例与悬停提示的多序列折线图。"""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumHeight(196)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self._series: list[ChartSeries] = []
        self._x_values: list[float] = []
        self._hidden: set[str] = set()
        self._legend_hits: list[tuple[str, QRectF]] = []
        self._hover_x: float | None = None
        self._hover_index: int | None = None
        self._plot_rect = QRectF()
        self._empty_text = "开始训练后，曲线会显示在这里"
        theme.changed.connect(self.update)

    # -- API ---------------------------------------------------------------
    def set_series(self, series: list[ChartSeries], x_values: list[float] | None = None) -> None:
        self._series = series
        if x_values is not None:
            self._x_values = list(x_values)
        else:
            length = max((len(s.values) for s in series), default=0)
            self._x_values = list(range(1, length + 1))
        self._hidden = {name for name in self._hidden if any(s.name == name for s in series)
                        and len(next(s for s in series if s.name == name).values) == len(self._x_values)}
        self.update()

    def clear(self) -> None:
        self._series = []
        self._x_values = []
        self._hidden.clear()
        self._hover_index = None
        self.update()

    def set_empty_text(self, text: str) -> None:
        self._empty_text = text
        self.update()

    # -- 数据范围 -----------------------------------------------------------
    def _visible_series(self) -> list[ChartSeries]:
        return [s for s in self._series if s.name not in self._hidden and s.values]

    def _value_range(self) -> tuple[float, float]:
        values: list[float] = []
        for series in self._visible_series():
            values.extend(v for v in series.values if v is not None and math.isfinite(v))
        if not values:
            return 0.0, 1.0
        lo, hi = min(values), max(values)
        if hi - lo < 1e-12:
            pad = max(abs(hi) * 0.1, 0.5)
            return lo - pad, hi + pad
        span = hi - lo
        if lo >= 0:
            lo = max(0.0, lo - span * 0.14)
            hi = hi + span * 0.12
        else:
            lo = lo - span * 0.12
            hi = hi + span * 0.12
        return lo, hi

    # -- 绘制 ---------------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        visible = self._visible_series()
        if not visible:
            self._paint_empty(painter)
            return
        self._paint_legend(painter)
        plot = QRectF(48, 40, max(self.width() - 64, 10), max(self.height() - 74, 10))
        self._plot_rect = plot
        lo, hi = self._value_range()
        step = _nice_step(hi - lo, 4)
        lo = math.floor(lo / step) * step
        hi = math.ceil(hi / step) * step
        if hi <= lo:
            hi = lo + step

        x_values = self._x_values or list(range(1, len(visible[0].values) + 1))
        x_lo = min(x_values) if x_values else 0
        x_hi = max(x_values) if x_values else 1
        if x_hi <= x_lo:
            x_hi = x_lo + 1

        def px(x: float) -> float:
            return plot.left() + (x - x_lo) / (x_hi - x_lo) * plot.width()

        def py(y: float) -> float:
            return plot.bottom() - (y - lo) / (hi - lo) * plot.height()

        # 网格 & 坐标轴
        painter.setFont(_font(self, 10))
        tick = lo
        grid_pen = QPen(QColor(theme.palette.grid), 1)
        grid_pen.setStyle(Qt.PenStyle.DotLine)
        while tick <= hi + step * 0.001:
            y = py(tick)
            painter.setPen(grid_pen)
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            painter.setPen(QColor(theme.palette.tertiary))
            painter.drawText(
                QRectF(0, y - 8, plot.left() - 9, 16),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                _fmt_axis(tick),
            )
            tick += step

        x_count = min(5, max(2, int(plot.width() // 110)))
        for i in range(x_count + 1):
            x_val = x_lo + (x_hi - x_lo) * i / x_count
            x_pos = px(x_val)
            painter.setPen(grid_pen)
            painter.drawLine(QPointF(x_pos, plot.top()), QPointF(x_pos, plot.bottom()))
            painter.setPen(QColor(theme.palette.tertiary))
            label = f"{x_val:.0f}"
            painter.drawText(
                QRectF(x_pos - 24, plot.bottom() + 5, 48, 16),
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                label,
            )
        painter.setFont(_font(self, 10))
        painter.setPen(QColor(theme.palette.tertiary))
        painter.drawText(
            QRectF(plot.left(), self.height() - 20, plot.width(), 16),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
            "Epoch",
        )

        # 曲线
        for series in visible:
            points: list[QPointF] = []
            for index, value in enumerate(series.values):
                if value is None or not math.isfinite(value):
                    continue
                if index >= len(x_values):
                    break
                points.append(QPointF(px(x_values[index]), py(value)))
            if not points:
                continue
            color = QColor(getattr(theme.palette, series.color, theme.palette.accent))
            if len(points) >= 2:
                line_path = self._smooth_path(points)
                if series.fill:
                    fill_path = QPainterPath(line_path)
                    fill_path.lineTo(points[-1].x(), plot.bottom())
                    fill_path.lineTo(points[0].x(), plot.bottom())
                    fill_path.closeSubpath()
                    gradient = QLinearGradient(0, plot.top(), 0, plot.bottom())
                    top_color = QColor(color)
                    top_color.setAlpha(52)
                    bottom_color = QColor(color)
                    bottom_color.setAlpha(0)
                    gradient.setColorAt(0.0, top_color)
                    gradient.setColorAt(1.0, bottom_color)
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(gradient)
                    painter.drawPath(fill_path)
                pen = QPen(color, 2.1)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                if series.dashed:
                    pen.setStyle(Qt.PenStyle.DashLine)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawPath(line_path)
            else:
                painter.setPen(QPen(color, 2))
            # 末端圆点
            last = points[-1]
            painter.setPen(QPen(QColor(theme.palette.card), 2))
            painter.setBrush(color)
            painter.drawEllipse(last, 3.4, 3.4)

        # 悬停提示
        if self._hover_index is not None and self._hover_x is not None:
            self._paint_hover(painter, plot, visible, x_values, lo, hi, x_lo, x_hi)

    def _smooth_path(self, points: list[QPointF]) -> QPainterPath:
        path = QPainterPath(points[0])
        if len(points) == 2:
            path.lineTo(points[1])
            return path
        for i in range(1, len(points) - 1):
            p0 = points[i - 1]
            p1 = points[i]
            p2 = points[i + 1]
            c1 = QPointF(p1.x() + (p2.x() - p0.x()) / 6.0, p1.y() + (p2.y() - p0.y()) / 6.0)
            c2 = QPointF(p2.x() - (p2.x() - p0.x()) / 6.0, p2.y() - (p2.y() - p0.y()) / 6.0)
            path.cubicTo(c1, c2, p2)
        return path

    def _paint_legend(self, painter: QPainter) -> None:
        self._legend_hits.clear()
        painter.setFont(_font(self, 11, QFont.Weight.DemiBold))
        x = 12.0
        for series in self._series:
            if not series.values:
                continue
            color = QColor(getattr(theme.palette, series.color, theme.palette.accent))
            hidden = series.name in self._hidden
            if hidden:
                color.setAlpha(80)
            metrics = painter.fontMetrics()
            width = metrics.horizontalAdvance(series.name) + 22
            rect = QRectF(x, 9, width, 20)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawEllipse(QPointF(x + 5, 19), 4, 4)
            painter.setPen(QColor(theme.palette.secondary if hidden else theme.palette.text))
            painter.drawText(
                QRectF(x + 13, 9, width - 13, 20),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                series.name,
            )
            self._legend_hits.append((series.name, rect))
            x += width + 8

    def _paint_empty(self, painter: QPainter) -> None:
        painter.setFont(_font(self, 11))
        painter.setPen(QColor(theme.palette.tertiary))
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._empty_text)
        # 装饰性虚线
        pen = QPen(QColor(theme.palette.grid), 1)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        for i in range(1, 4):
            y = self.height() * i / 4
            painter.drawLine(QPointF(self.width() * 0.15, y), QPointF(self.width() * 0.85, y))

    def _paint_hover(
        self,
        painter: QPainter,
        plot: QRectF,
        visible: list[ChartSeries],
        x_values: list[float],
        lo: float,
        hi: float,
        x_lo: float,
        x_hi: float,
    ) -> None:
        index = self._hover_index
        if index is None or index < 0:
            return
        if index >= len(x_values):
            return
        x = plot.left() + (x_values[index] - x_lo) / (x_hi - x_lo) * plot.width()
        pen = QPen(QColor(theme.palette.tertiary), 1)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))

        rows: list[tuple[QColor, str, str]] = []
        for series in visible:
            if index >= len(series.values):
                continue
            value = series.values[index]
            if value is None or not math.isfinite(value):
                continue
            color = QColor(getattr(theme.palette, series.color, theme.palette.accent))
            y = plot.bottom() - (value - lo) / (hi - lo) * plot.height()
            painter.setPen(QPen(QColor(theme.palette.card), 2))
            painter.setBrush(color)
            painter.drawEllipse(QPointF(x, y), 4, 4)
            rows.append((color, series.name, _fmt_value(value)))
        if not rows:
            return

        painter.setFont(_font(self, 10))
        line_height = 17
        width = 132
        height = 12 + line_height * len(rows) + 8
        left = x + 14
        if left + width > plot.right():
            left = x - width - 14
        top = plot.top() + 8
        card = QRectF(left, top, width, height)
        painter.setPen(QPen(QColor(theme.palette.border), 1))
        painter.setBrush(theme.rgba("card", 242))
        painter.drawRoundedRect(card, 9, 9)
        painter.setPen(QColor(theme.palette.tertiary))
        painter.drawText(
            QRectF(card.left() + 10, card.top() + 5, width - 20, 14),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f"Epoch {x_values[index]:.0f}",
        )
        for i, (color, name, value) in enumerate(rows):
            y = card.top() + 20 + i * line_height
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawEllipse(QPointF(card.left() + 13, y + 7), 3, 3)
            painter.setPen(QColor(theme.palette.secondary))
            painter.drawText(
                QRectF(card.left() + 22, y, width - 88, 14),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                name,
            )
            painter.setPen(QColor(theme.palette.text))
            painter.setFont(_font(self, 10, QFont.Weight.DemiBold))
            painter.drawText(
                QRectF(card.left() + 60, y, width - 68, 14),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                value,
            )
            painter.setFont(_font(self, 10))

    # -- 交互 ---------------------------------------------------------------
    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        pos = event.position()
        if not self._plot_rect.contains(pos) or not self._x_values:
            if self._hover_index is not None:
                self._hover_index = None
                self._hover_x = None
                self.update()
            return
        x_values = self._x_values
        x_lo, x_hi = min(x_values), max(x_values)
        if x_hi <= x_lo:
            x_hi = x_lo + 1
        ratio = (pos.x() - self._plot_rect.left()) / max(self._plot_rect.width(), 1)
        target = x_lo + ratio * (x_hi - x_lo)
        index = min(range(len(x_values)), key=lambda i: abs(x_values[i] - target))
        if index != self._hover_index:
            self._hover_index = index
            self._hover_x = pos.x()
            self.update()

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._hover_index = None
        self._hover_x = None
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            for name, rect in self._legend_hits:
                if rect.contains(event.position()):
                    if name in self._hidden:
                        self._hidden.remove(name)
                    else:
                        self._hidden.add(name)
                    self.update()
                    return
        super().mousePressEvent(event)


class ScatterChart(QWidget):
    """真实值 vs 预测值散点图（回归评估）。"""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumHeight(280)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._true: list[float] = []
        self._pred: list[float] = []
        self._hover: tuple[int, float, float] | None = None
        self.setMouseTracking(True)
        theme.changed.connect(self.update)

    def set_data(self, y_true, y_pred) -> None:
        pairs = [
            (float(t), float(p))
            for t, p in zip(y_true, y_pred)
            if math.isfinite(float(t)) and math.isfinite(float(p))
        ]
        if len(pairs) > 4000:
            step = max(1, len(pairs) // 4000)
            pairs = pairs[::step]
        self._true = [p[0] for p in pairs]
        self._pred = [p[1] for p in pairs]
        self._hover = None
        self.update()

    def clear(self) -> None:
        self._true, self._pred = [], []
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        if not self._true:
            painter.setFont(_font(self, 11))
            painter.setPen(QColor(theme.palette.tertiary))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "评估回归模型后显示预测散点")
            return
        plot = QRectF(54, 22, max(self.width() - 74, 10), max(self.height() - 74, 10))
        values = self._true + self._pred
        lo, hi = min(values), max(values)
        if hi - lo < 1e-9:
            pad = max(abs(hi) * 0.1, 1.0)
            lo -= pad
            hi += pad
        span = hi - lo
        lo -= span * 0.06
        hi += span * 0.06
        step = _nice_step(hi - lo, 4)
        lo = math.floor(lo / step) * step
        hi = math.ceil(hi / step) * step

        def px(value: float) -> float:
            return plot.left() + (value - lo) / (hi - lo) * plot.width()

        def py(value: float) -> float:
            return plot.bottom() - (value - lo) / (hi - lo) * plot.height()

        painter.setFont(_font(self, 10))
        grid_pen = QPen(QColor(theme.palette.grid), 1)
        grid_pen.setStyle(Qt.PenStyle.DotLine)
        tick = lo
        while tick <= hi + step * 0.001:
            y = py(tick)
            painter.setPen(grid_pen)
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            painter.drawLine(QPointF(px(tick), plot.top()), QPointF(px(tick), plot.bottom()))
            painter.setPen(QColor(theme.palette.tertiary))
            painter.drawText(
                QRectF(0, y - 8, plot.left() - 8, 16),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                _fmt_axis(tick),
            )
            painter.drawText(
                QRectF(px(tick) - 30, plot.bottom() + 5, 60, 16),
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                _fmt_axis(tick),
            )
            tick += step

        reference = QPen(QColor(theme.palette.tertiary), 1)
        reference.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(reference)
        painter.drawLine(QPointF(px(lo), py(lo)), QPointF(px(hi), py(hi)))

        color = theme.rgba("accent", 150)
        painter.setPen(QPen(theme.rgba("accent", 210), 1))
        painter.setBrush(color)
        for index, (true, pred) in enumerate(zip(self._true, self._pred)):
            radius = 3.2 if len(self._true) < 500 else 2.0
            painter.drawEllipse(QPointF(px(pred), py(true)), radius, radius)

        painter.setFont(_font(self, 10))
        painter.setPen(QColor(theme.palette.tertiary))
        painter.drawText(
            QRectF(plot.left(), self.height() - 20, plot.width(), 16),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
            "预测值",
        )
        painter.save()
        painter.translate(14, plot.center().y())
        painter.rotate(-90)
        painter.drawText(
            QRectF(-70, -12, 140, 16),
            Qt.AlignmentFlag.AlignCenter,
            "真实值",
        )
        painter.restore()

        if self._hover is not None:
            index, true, pred = self._hover
            x, y = px(pred), py(true)
            painter.setPen(QPen(QColor(theme.palette.card), 2))
            painter.setBrush(QColor(theme.palette.accent))
            painter.drawEllipse(QPointF(x, y), 4, 4)
            text = f"真实 {_fmt_value(true)}  预测 {_fmt_value(pred)}"
            painter.setFont(_font(self, 10, QFont.Weight.DemiBold))
            metrics = painter.fontMetrics()
            width = metrics.horizontalAdvance(text) + 20
            rect = QRectF(
                min(max(x + 12, plot.left()), plot.right() - width),
                max(plot.top() + 4, y - 34),
                width,
                24,
            )
            painter.setPen(QPen(QColor(theme.palette.border), 1))
            painter.setBrush(theme.rgba("card", 244))
            painter.drawRoundedRect(rect, 8, 8)
            painter.setPen(QColor(theme.palette.text))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if not self._true:
            return
        plot = QRectF(54, 22, max(self.width() - 74, 10), max(self.height() - 74, 10))
        if not plot.contains(event.position()):
            if self._hover is not None:
                self._hover = None
                self.update()
            return
        values = self._true + self._pred
        lo, hi = min(values), max(values)
        if hi - lo < 1e-9:
            pad = max(abs(hi) * 0.1, 1.0)
            lo -= pad
            hi += pad
        span = hi - lo
        lo -= span * 0.06
        hi += span * 0.06
        step = _nice_step(hi - lo, 4)
        lo = math.floor(lo / step) * step
        hi = math.ceil(hi / step) * step

        def to_px(value: float) -> float:
            return plot.left() + (value - lo) / (hi - lo) * plot.width()

        def to_py(value: float) -> float:
            return plot.bottom() - (value - lo) / (hi - lo) * plot.height()

        best = None
        best_distance = 14.0
        for index, (true, pred) in enumerate(zip(self._true, self._pred)):
            dx = to_px(pred) - event.position().x()
            dy = to_py(true) - event.position().y()
            distance = math.hypot(dx, dy)
            if distance < best_distance:
                best_distance = distance
                best = (index, true, pred)
        if best != self._hover:
            self._hover = best
            self.update()

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._hover = None
        self.update()


class ConfusionMatrixWidget(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumHeight(280)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._matrix: list[list[int]] = []
        self._labels: list[str] = []
        self._hover: tuple[int, int] | None = None
        self.setMouseTracking(True)
        theme.changed.connect(self.update)

    def set_data(self, matrix: list[list[int]], labels: list[str]) -> None:
        self._matrix = matrix
        self._labels = labels
        self._hover = None
        self.update()

    def clear(self) -> None:
        self._matrix = []
        self._labels = []
        self.update()

    def _geometry(self) -> tuple[QRectF, float]:
        n = max(len(self._matrix), 1)
        left = 80 if n <= 6 else 54
        top = 30
        bottom = 64 if n <= 8 else 32
        size = min(max(self.width() - left - 18, 40), max(self.height() - top - bottom, 40))
        plot = QRectF(left, top, max(self.width() - left - 18, 40), max(self.height() - top - bottom, 40))
        cell = min(plot.width() / n, plot.height() / n)
        plot = QRectF(plot.left(), plot.top(), cell * n, cell * n)
        return plot, cell

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        if not self._matrix:
            painter.setFont(_font(self, 11))
            painter.setPen(QColor(theme.palette.tertiary))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "评估分类模型后显示混淆矩阵")
            return
        plot, cell = self._geometry()
        n = len(self._matrix)
        total = max(sum(sum(row) for row in self._matrix), 1)
        max_value = max((max(row) for row in self._matrix), default=1) or 1
        painter.setFont(_font(self, 10))
        painter.setPen(QColor(theme.palette.tertiary))
        painter.drawText(
            QRectF(plot.left(), 4, plot.width(), 18),
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
            "预测类别",
        )
        painter.save()
        painter.translate(10, plot.center().y())
        painter.rotate(-90)
        painter.drawText(
            QRectF(-80, -10, 160, 18),
            Qt.AlignmentFlag.AlignCenter,
            "真实类别",
        )
        painter.restore()

        label_font = _font(self, 10 if n <= 8 else 8)
        painter.setFont(label_font)
        for row in range(n):
            for col in range(n):
                value = self._matrix[row][col]
                rect = QRectF(plot.left() + col * cell + 2, plot.top() + row * cell + 2,
                              cell - 4, cell - 4)
                ratio = value / max_value if max_value else 0
                color = QColor(theme.palette.accent if row == col else theme.palette.orange)
                color.setAlpha(28 + int(190 * ratio))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(color)
                painter.drawRoundedRect(rect, 6, 6)
                if self._hover == (row, col):
                    painter.setPen(QPen(QColor(theme.palette.text), 1.4))
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawRoundedRect(rect, 6, 6)
                painter.setPen(QColor("#FFFFFF" if ratio > 0.55 else theme.palette.text))
                font = _font(self, 12 if n <= 8 else 9, QFont.Weight.DemiBold)
                painter.setFont(font)
                share = value / max(sum(self._matrix[row]), 1) * 100
                text = str(value) if cell < 44 else f"{value}\n{share:.0f}%"
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
                painter.setFont(label_font)

        for row in range(n):
            name = _short_label(self._labels, row)
            painter.setPen(QColor(theme.palette.secondary))
            painter.drawText(
                QRectF(0, plot.top() + row * cell, plot.left() - 10, cell),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                name,
            )
        painter.setFont(_font(self, 9))
        for col in range(n):
            name = _short_label(self._labels, col, max_chars=5)
            painter.setPen(QColor(theme.palette.secondary))
            painter.drawText(
                QRectF(plot.left() + col * cell, plot.bottom() + 6, cell, 15),
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                name,
            )

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if not self._matrix:
            return
        plot, cell = self._geometry()
        col = int((event.position().x() - plot.left()) // cell)
        row = int((event.position().y() - plot.top()) // cell)
        n = len(self._matrix)
        hover = (row, col) if 0 <= row < n and 0 <= col < n and plot.contains(event.position()) else None
        if hover != self._hover:
            self._hover = hover
            self.update()

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._hover = None
        self.update()


def _short_label(labels: list[str], index: int, max_chars: int = 7) -> str:
    if index < len(labels):
        name = str(labels[index])
    else:
        name = str(index)
    return name if len(name) <= max_chars else name[: max_chars - 1] + "…"


class NetworkDiagram(QWidget):
    """分层神经元结构示意图。"""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumHeight(240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._layers: list[dict] = []
        theme.changed.connect(self.update)

    def set_layers(self, layers: list[dict]) -> None:
        self._layers = layers
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        layers = self._layers
        if not layers:
            painter.setFont(_font(self, 11))
            painter.setPen(QColor(theme.palette.tertiary))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "选择数据后生成网络结构")
            return

        # 背景光晕
        glow = QRadialGradient(self.width() * 0.5, self.height() * 0.45, max(self.width(), self.height()) * 0.6)
        accent = theme.rgba("accent", 16 if not theme.is_dark else 26)
        transparent = theme.rgba("accent", 0)
        glow.setColorAt(0.0, accent)
        glow.setColorAt(1.0, transparent)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(glow)
        painter.drawRect(self.rect())

        top = 30
        bottom = self.height() - 54
        left = 34
        right = self.width() - 34
        count = len(layers)
        if count == 1:
            x_positions = [self.width() / 2]
        else:
            x_positions = [left + i * (right - left) / (count - 1) for i in range(count)]
        available = max(bottom - top, 40)
        max_nodes = max(3, min(10, int(available / 20)))
        layer_font = _font(self, 10 if count <= 8 else 8)
        painter.setFont(layer_font)
        metrics = painter.fontMetrics()

        node_positions: list[list[tuple[float, float, bool]]] = []
        radius = 7.0
        for index, layer in enumerate(layers):
            width = max(int(layer.get("width", 1)), 1)
            x = x_positions[index]
            if width <= max_nodes:
                count_nodes = width
                ellipsis = False
                slots = width
            else:
                count_nodes = max_nodes - 1
                ellipsis = True
                slots = count_nodes + 1
            spacing = available / (slots + 1)
            radius = max(2.8, min(8.0, spacing * 0.34))
            positions: list[tuple[float, float, bool]] = []
            for i in range(slots):
                y = top + spacing * (i + 1)
                if ellipsis and i == slots // 2:
                    positions.append((x, y, True))
                else:
                    positions.append((x, y, False))
            node_positions.append(positions)

        # 连线
        line_color = theme.rgba("secondary", 44)
        painter.setPen(QPen(line_color, 0.8))
        for i in range(count - 1):
            for x1, y1, skip1 in node_positions[i]:
                if skip1:
                    continue
                for x2, y2, skip2 in node_positions[i + 1]:
                    if skip2:
                        continue
                    painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

        # 节点
        palette_colors = ["accent", "purple", "teal", "green"]
        for index, layer in enumerate(layers):
            kind = layer.get("kind", "hidden")
            color_name = "accent" if kind == "input" else ("green" if kind == "output" else "purple")
            color = QColor(getattr(theme.palette, color_name))
            for x, y, is_ellipsis in node_positions[index]:
                if is_ellipsis:
                    painter.setPen(QColor(theme.palette.tertiary))
                    painter.setFont(_font(self, int(max(9, radius * 2.4)), QFont.Weight.Bold))
                    painter.drawText(
                        QRectF(x - 12, y - 10, 24, 20),
                        Qt.AlignmentFlag.AlignCenter,
                        "⋮",
                    )
                    painter.setFont(layer_font)
                    continue
                halo = QColor(color)
                halo.setAlpha(40)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(halo)
                painter.drawEllipse(QPointF(x, y), radius + 2.6, radius + 2.6)
                painter.setBrush(color)
                painter.drawEllipse(QPointF(x, y), radius, radius)
                painter.setPen(QPen(theme.rgba("card", 220), 1.2))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(QPointF(x, y), radius, radius)

            label = str(layer.get("label", ""))
            width = int(layer.get("width", 1))
            text = f"{label} · {width}"
            text_width = metrics.horizontalAdvance(text)
            # 层多时只保留宽度，避免文字重叠
            if count > 9:
                text = str(width)
            painter.setPen(QColor(theme.palette.secondary))
            painter.drawText(
                QRectF(x_positions[index] - max(text_width, 40) / 2 - 16, bottom + 18,
                       max(text_width, 40) + 32, 18),
                Qt.AlignmentFlag.AlignCenter,
                text,
            )

        # 顶部图例
        legend = [("输入层", "accent"), ("隐藏层", "purple"), ("输出层", "green")]
        x = 16
        painter.setFont(_font(self, 10))
        for text, color_name in legend:
            painter.setBrush(QColor(getattr(theme.palette, color_name)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(x + 4, 15), 3.5, 3.5)
            painter.setPen(QColor(theme.palette.tertiary))
            painter.drawText(QRectF(x + 12, 6, 56, 18), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, text)
            x += 68


class ProgressRing(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFixedSize(128, 128)
        self._progress = 0.0
        self._title = "0%"
        self._subtitle = "等待开始"
        theme.changed.connect(self.update)

    def set_progress(self, progress: float, title: str | None = None, subtitle: str | None = None) -> None:
        self._progress = max(0.0, min(1.0, float(progress)))
        if title is not None:
            self._title = title
        if subtitle is not None:
            self._subtitle = subtitle
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(9, 9, self.width() - 18, self.height() - 18)
        painter.setPen(QPen(QColor(theme.palette.field), 9))
        painter.drawArc(rect, 0, 360 * 16)
        if self._progress > 0:
            pen = QPen(QColor(theme.palette.accent), 9)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawArc(rect, 90 * 16, -int(360 * 16 * self._progress))
        painter.setPen(QColor(theme.palette.text))
        painter.setFont(_font(self, 21, QFont.Weight.Bold))
        painter.drawText(
            QRectF(0, self.height() / 2 - 22, self.width(), 28),
            Qt.AlignmentFlag.AlignCenter,
            self._title,
        )
        painter.setPen(QColor(theme.palette.tertiary))
        painter.setFont(_font(self, 10))
        painter.drawText(
            QRectF(0, self.height() / 2 + 8, self.width(), 18),
            Qt.AlignmentFlag.AlignCenter,
            self._subtitle,
        )


class GradientLogo(QWidget):
    def __init__(self, size: int = 36, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFixedSize(size, size)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(0, 0, self.width(), self.height())
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        gradient.setColorAt(0.0, QColor(theme.palette.accent))
        gradient.setColorAt(1.0, QColor(theme.palette.purple))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(gradient)
        painter.drawRoundedRect(rect, self.width() * 0.3, self.width() * 0.3)

        # 简化网络图形
        scale = self.width() / 36.0
        columns = [(11, [13, 23], "input"), (18, [9, 18, 27], "hidden"), (25, [13, 23], "output")]
        painter.setPen(QPen(QColor(255, 255, 255, 90), 1.0 * scale))
        for (x1, ys1, _), (x2, ys2, _) in zip(columns, columns[1:]):
            for y1 in ys1:
                for y2 in ys2:
                    painter.drawLine(
                        QPointF(x1 * scale, y1 * scale),
                        QPointF(x2 * scale, y2 * scale),
                    )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 255, 255, 235))
        for x, ys, _ in columns:
            for y in ys:
                painter.drawEllipse(QPointF(x * scale, y * scale), 1.9 * scale, 1.9 * scale)
