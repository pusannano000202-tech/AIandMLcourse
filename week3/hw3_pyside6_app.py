"""
Week 3 Homework - Neural Networks Fundamentals
PySide6 GUI Application

신경망 기초 5개 랩을 통합한 인터랙티브 GUI 앱
- Lab 1: Perceptron (퍼셉트론)
- Lab 2: Activation Functions (활성화 함수)
- Lab 3: Forward Propagation (순전파)
- Lab 4: MLP with Backpropagation (다층 퍼셉트론)
- Lab 5: Universal Approximation Theorem
"""

import sys
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QSpinBox, QTextEdit, QGroupBox,
    QComboBox, QDoubleSpinBox, QSplitter, QGridLayout
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ── 한글 폰트 설정 ────────────────────────────────────────────
def setup_korean_font():
    font_list = [f.name for f in fm.fontManager.ttflist]
    for font in ["Malgun Gothic", "Gulim", "Batang", "AppleGothic"]:
        if font in font_list:
            plt.rcParams["font.family"] = font
            break
    plt.rcParams["axes.unicode_minus"] = False

setup_korean_font()


# ══════════════════════════════════════════════════════════════
# 공통 수학 함수
# ══════════════════════════════════════════════════════════════
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return np.where(x > 0, 1.0, 0.0)

def tanh_fn(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1 - np.tanh(x) ** 2

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_deriv(x, alpha=0.01):
    return np.where(x > 0, 1.0, alpha)


# ══════════════════════════════════════════════════════════════
# Lab 1: Perceptron Tab
# ══════════════════════════════════════════════════════════════
class PerceptronLogic:
    def __init__(self, lr=0.1):
        self.weights = np.random.randn(2)
        self.bias = np.random.randn()
        self.lr = lr

    def predict(self, x):
        return 1 if np.dot(x, self.weights) + self.bias >= 0 else 0

    def train(self, X, y, epochs=200):
        for _ in range(epochs):
            for xi, yi in zip(X, y):
                err = yi - self.predict(xi)
                self.weights += self.lr * err * xi
                self.bias += self.lr * err


class Lab1Tab(QWidget):
    def __init__(self):
        super().__init__()
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # 제목
        title = QLabel("Lab 1: 퍼셉트론 (Perceptron) — 논리 게이트 학습")
        title.setFont(QFont("", 13, QFont.Weight.Bold))
        layout.addWidget(title)

        # 컨트롤
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("학습률(LR):"))
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.01, 1.0)
        self.lr_spin.setSingleStep(0.05)
        self.lr_spin.setValue(0.1)
        ctrl.addWidget(self.lr_spin)

        ctrl.addWidget(QLabel("에폭:"))
        self.epoch_spin = QSpinBox()
        self.epoch_spin.setRange(10, 2000)
        self.epoch_spin.setValue(200)
        ctrl.addWidget(self.epoch_spin)

        self.run_btn = QPushButton("학습 시작")
        self.run_btn.clicked.connect(self.run)
        ctrl.addWidget(self.run_btn)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        # 그래프 캔버스
        self.fig = Figure(figsize=(12, 4))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        # 결과 텍스트
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(120)
        layout.addWidget(self.result_text)

        self.run()  # 초기 실행

    def run(self):
        lr = self.lr_spin.value()
        epochs = self.epoch_spin.value()
        X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
        gates = {
            "AND Gate\n(선형 분리 가능 [O])": np.array([0,0,0,1]),
            "OR Gate\n(선형 분리 가능 [O])":  np.array([0,1,1,1]),
            "XOR Gate\n(선형 분리 불가능 [X])": np.array([0,1,1,0]),
        }

        self.fig.clear()
        axes = self.fig.subplots(1, 3)
        log = []

        for ax, (title, y) in zip(axes, gates.items()):
            p = PerceptronLogic(lr=lr)
            p.train(X, y, epochs=epochs)

            # 결정 경계
            xs = np.linspace(-0.5, 1.5, 120)
            xx, yy = np.meshgrid(xs, xs)
            Z = np.array([p.predict(np.array([a, b]))
                          for a, b in zip(xx.ravel(), yy.ravel())]).reshape(xx.shape)
            ax.contourf(xx, yy, Z, alpha=0.25, levels=[-0.5, 0.5, 1.5],
                        colors=["#4488ff", "#ff4444"])

            for xi, yi in zip(X, y):
                c = "red" if yi == 1 else "blue"
                m = "o" if yi == 1 else "X"
                ax.scatter(xi[0], xi[1], c=c, marker=m, s=200,
                           edgecolors="black", linewidth=1.5, zorder=5)

            errs = sum(p.predict(xi) != yi for xi, yi in zip(X, y))
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.5, 1.5)
            ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
            ax.grid(True, alpha=0.3)
            log.append(f"{title.split(chr(10))[0]}: 오류 {errs}/4")

        self.fig.tight_layout()
        self.canvas.draw()
        self.result_text.setPlainText("\n".join(log) +
            "\n\n핵심: XOR은 직선(선형) 하나로 분리 불가능 → 다층 네트워크 필요!")


# ══════════════════════════════════════════════════════════════
# Lab 2: Activation Functions Tab
# ══════════════════════════════════════════════════════════════
class Lab2Tab(QWidget):
    def __init__(self):
        super().__init__()
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel("Lab 2: 활성화 함수 비교 (Activation Functions)")
        title.setFont(QFont("", 13, QFont.Weight.Bold))
        layout.addWidget(title)

        # x 범위 컨트롤
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("x 범위:"))
        self.xrange_spin = QSpinBox()
        self.xrange_spin.setRange(3, 15)
        self.xrange_spin.setValue(5)
        self.xrange_spin.valueChanged.connect(self.draw)
        ctrl.addWidget(self.xrange_spin)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        self.fig = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        info = QTextEdit()
        info.setReadOnly(True)
        info.setMaximumHeight(100)
        info.setPlainText(
            "Sigmoid: 범위(0,1) | 장점: 확률 해석 | 단점: Vanishing Gradient, 비0중심\n"
            "Tanh:    범위(-1,1)| 장점: 0중심     | 단점: Vanishing Gradient 여전히 존재\n"
            "ReLU:    범위[0,∞) | 장점: 빠름, No Vanishing | 단점: Dying ReLU\n"
            "LeakyReLU: 음수 영역 살아있음 → Dying ReLU 해결. α=0.01"
        )
        layout.addWidget(info)

        self.draw()

    def draw(self):
        rng = self.xrange_spin.value()
        x = np.linspace(-rng, rng, 300)
        self.fig.clear()
        axes = self.fig.subplots(2, 2)

        # 함수 비교
        ax = axes[0, 0]
        ax.plot(x, sigmoid(x), label="Sigmoid", lw=2)
        ax.plot(x, tanh_fn(x), label="Tanh", lw=2)
        ax.plot(x, relu(x), label="ReLU", lw=2)
        ax.plot(x, leaky_relu(x), label="Leaky ReLU", lw=2, ls="--")
        ax.axhline(0, color="k", alpha=0.3); ax.axvline(0, color="k", alpha=0.3)
        ax.set_title("Activation Functions 비교", fontweight="bold")
        ax.legend(); ax.grid(True, alpha=0.3)

        # 미분 비교
        ax = axes[0, 1]
        ax.plot(x, sigmoid_deriv(x), label="Sigmoid'", lw=2)
        ax.plot(x, tanh_deriv(x), label="Tanh'", lw=2)
        ax.plot(x, relu_deriv(x), label="ReLU'", lw=2)
        ax.plot(x, leaky_relu_deriv(x), label="Leaky ReLU'", lw=2, ls="--")
        ax.axhline(0, color="k", alpha=0.3); ax.axvline(0, color="k", alpha=0.3)
        ax.set_title("Gradient (미분) 비교", fontweight="bold")
        ax.legend(); ax.grid(True, alpha=0.3)

        # Sigmoid vs Tanh
        ax = axes[1, 0]
        ax.plot(x, sigmoid(x), label="Sigmoid (0,1)", lw=3)
        ax.plot(x, tanh_fn(x), label="Tanh (-1,1)", lw=3)
        ax.axhline(0.5, color="blue", ls="--", alpha=0.4, label="Sigmoid 중심=0.5")
        ax.axhline(0, color="k", alpha=0.3); ax.axvline(0, color="k", alpha=0.3)
        ax.set_title("Sigmoid vs Tanh (중심 위치 차이)", fontweight="bold")
        ax.legend(); ax.grid(True, alpha=0.3)

        # ReLU vs Leaky ReLU
        ax = axes[1, 1]
        ax.plot(x, relu(x), label="ReLU (x<0 → 죽음)", lw=3)
        ax.plot(x, leaky_relu(x), label="Leaky ReLU (x<0 → 살아있음)", lw=3)
        ax.axvline(0, color="red", ls="--", alpha=0.5, label="경계 x=0")
        ax.set_title("ReLU vs Leaky ReLU — Dying ReLU 문제", fontweight="bold")
        ax.legend(); ax.grid(True, alpha=0.3)

        self.fig.tight_layout()
        self.canvas.draw()


# ══════════════════════════════════════════════════════════════
# Lab 3: Forward Propagation Tab
# ══════════════════════════════════════════════════════════════
class SimpleNetwork:
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.W1 = np.random.randn(2, 3) * 0.5
        self.b1 = np.random.randn(3) * 0.1
        self.W2 = np.random.randn(3, 1) * 0.5
        self.b2 = np.random.randn(1) * 0.1

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2


class Lab3Tab(QWidget):
    def __init__(self):
        super().__init__()
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel("Lab 3: 순전파 시각화 (Forward Propagation)")
        title.setFont(QFont("", 13, QFont.Weight.Bold))
        layout.addWidget(title)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("x₁:"))
        self.x1 = QDoubleSpinBox(); self.x1.setRange(-2, 2); self.x1.setValue(0.5)
        self.x1.setSingleStep(0.1); self.x1.valueChanged.connect(self.draw)
        ctrl.addWidget(self.x1)

        ctrl.addWidget(QLabel("x₂:"))
        self.x2 = QDoubleSpinBox(); self.x2.setRange(-2, 2); self.x2.setValue(0.8)
        self.x2.setSingleStep(0.1); self.x2.valueChanged.connect(self.draw)
        ctrl.addWidget(self.x2)

        self.reset_btn = QPushButton("가중치 랜덤 재초기화")
        self.reset_btn.clicked.connect(self._reset_network)
        ctrl.addWidget(self.reset_btn)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        self.fig = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(110)
        layout.addWidget(self.result_text)

        self.network = SimpleNetwork(seed=42)
        self.draw()

    def _reset_network(self):
        seed = np.random.randint(0, 10000)
        self.network = SimpleNetwork(seed=seed)
        self.draw()

    def draw(self):
        X = np.array([self.x1.value(), self.x2.value()])
        net = self.network
        out = net.forward(X)

        self.fig.clear()
        axes = self.fig.subplots(2, 2)

        # 네트워크 구조 다이어그램
        ax = axes[0, 0]
        ax.set_xlim(0, 4); ax.set_ylim(-0.5, 4.5); ax.axis("off")
        ax.set_title("Network Architecture (2→3→1)", fontweight="bold")

        input_ys = [1, 3]
        hidden_ys = [0.5, 2, 3.5]
        out_y = 2

        for i, y in enumerate(input_ys):
            c = mpatches.Circle((0.5, y), 0.22, color="#aad4f5", ec="black", lw=2)
            ax.add_patch(c)
            ax.text(0.5, y, f"x{i+1}\n{X[i]:.2f}", ha="center", va="center",
                    fontsize=8, fontweight="bold")

        for i, y in enumerate(hidden_ys):
            c = mpatches.Circle((2, y), 0.22, color="#b8f5aa", ec="black", lw=2)
            ax.add_patch(c)
            ax.text(2, y, f"h{i+1}\n{net.a1[i]:.2f}", ha="center", va="center",
                    fontsize=8, fontweight="bold")

        c = mpatches.Circle((3.5, out_y), 0.22, color="#f5b8b8", ec="black", lw=2)
        ax.add_patch(c)
        ax.text(3.5, out_y, f"y\n{out[0]:.2f}", ha="center", va="center",
                fontsize=8, fontweight="bold")

        for iy in input_ys:
            for hy in hidden_ys:
                ax.plot([0.72, 1.78], [iy, hy], "k-", alpha=0.25, lw=1)
        for hy in hidden_ys:
            ax.plot([2.22, 3.28], [hy, out_y], "k-", alpha=0.25, lw=1)

        ax.text(0.5, -0.3, "Input\nLayer", ha="center", fontsize=9)
        ax.text(2, -0.3, "Hidden\n(ReLU)", ha="center", fontsize=9)
        ax.text(3.5, -0.3, "Output\n(Sigmoid)", ha="center", fontsize=9)

        # Layer 1 값
        ax = axes[0, 1]
        pos = np.arange(3)
        w = 0.25
        ax.bar(pos - w, [X[0], X[1], 0], w, label="Input", color="steelblue", alpha=0.8)
        ax.bar(pos, net.z1, w, label="z₁ (before ReLU)", color="orange", alpha=0.8)
        ax.bar(pos + w, net.a1, w, label="a₁ (after ReLU)", color="green", alpha=0.8)
        ax.set_title("Layer 1: Input → Hidden (ReLU)", fontweight="bold")
        ax.set_xticks(pos); ax.set_xticklabels(["Neuron 1", "Neuron 2", "Neuron 3"])
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # Layer 2 값
        ax = axes[1, 0]
        layers = ["a₁[0]\n(Hidden input)", "z₂\n(before Sigmoid)", "a₂\n(after Sigmoid)"]
        values = [net.a1[0], net.z2[0], net.a2[0]]
        colors = ["green", "orange", "red"]
        bars = ax.barh(layers, values, color=colors, alpha=0.75)
        for bar, val in zip(bars, values):
            ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                    f"{val:.4f}", va="center", fontsize=9)
        ax.set_title("Layer 2: Hidden → Output (Sigmoid)", fontweight="bold")
        ax.grid(True, alpha=0.3)

        # 수식 요약
        ax = axes[1, 1]
        ax.axis("off")
        ax.set_title("수식 요약", fontweight="bold")
        lines = [
            "Forward Propagation:",
            "",
            f"입력: X = [{X[0]:.2f}, {X[1]:.2f}]",
            "",
            "▶ Layer 1",
            f"  z₁ = X @ W₁ + b₁ = {np.round(net.z1, 3)}",
            f"  a₁ = ReLU(z₁)     = {np.round(net.a1, 3)}",
            "",
            "▶ Layer 2",
            f"  z₂ = a₁ @ W₂ + b₂ = {net.z2[0]:.4f}",
            f"  a₂ = σ(z₂)         = {net.a2[0]:.4f}",
            "",
            f"→ 최종 출력: {net.a2[0]:.4f}",
        ]
        y_pos = 0.97
        for line in lines:
            fs = 10 if line.startswith("▶") or line.startswith("Forward") else 9
            fw = "bold" if line.startswith("▶") or line.startswith("Forward") else "normal"
            ax.text(0.02, y_pos, line, transform=ax.transAxes,
                    fontsize=fs, fontweight=fw)
            y_pos -= 0.075

        self.fig.tight_layout()
        self.canvas.draw()

        self.result_text.setPlainText(
            f"입력 X = [{X[0]:.2f}, {X[1]:.2f}]\n"
            f"z₁ = {np.round(net.z1, 4)}\n"
            f"a₁ = ReLU(z₁) = {np.round(net.a1, 4)}\n"
            f"z₂ = {net.z2[0]:.4f}  →  a₂ = σ(z₂) = {net.a2[0]:.4f}\n"
            f"[설명] 음수 z₁ 값은 ReLU에 의해 0으로 클리핑되고, 출력은 Sigmoid로 (0,1) 범위가 됩니다."
        )


# ══════════════════════════════════════════════════════════════
# Lab 4: MLP / Backpropagation Tab
# ══════════════════════════════════════════════════════════════
class MLPModel:
    def __init__(self, hidden=4, lr=0.5):
        self.W1 = np.random.randn(2, hidden) * np.sqrt(2.0 / 2)
        self.b1 = np.zeros((1, hidden))
        self.W2 = np.random.randn(hidden, 1) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros((1, 1))
        self.lr = lr
        self.loss_history = []

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def train_step(self, X, y):
        m = X.shape[0]
        out = self.forward(X)
        loss = float(np.mean((out - y) ** 2))
        self.loss_history.append(loss)

        dz2 = out - y
        dW2 = (1/m) * self.a1.T @ dz2
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * sigmoid_deriv(self.z1)
        dW1 = (1/m) * X.T @ dz1
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        return loss


class Lab4Tab(QWidget):
    def __init__(self):
        super().__init__()
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel("Lab 4: MLP + Backpropagation — XOR 문제 해결")
        title.setFont(QFont("", 13, QFont.Weight.Bold))
        layout.addWidget(title)

        # 컨트롤
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("은닉 뉴런:"))
        self.hidden_spin = QSpinBox(); self.hidden_spin.setRange(2, 16); self.hidden_spin.setValue(4)
        ctrl.addWidget(self.hidden_spin)

        ctrl.addWidget(QLabel("학습률:"))
        self.lr_spin = QDoubleSpinBox(); self.lr_spin.setRange(0.01, 2.0)
        self.lr_spin.setSingleStep(0.05); self.lr_spin.setValue(0.5)
        ctrl.addWidget(self.lr_spin)

        ctrl.addWidget(QLabel("에폭:"))
        self.epoch_spin = QSpinBox(); self.epoch_spin.setRange(100, 50000)
        self.epoch_spin.setSingleStep(500); self.epoch_spin.setValue(10000)
        ctrl.addWidget(self.epoch_spin)

        self.run_btn = QPushButton("학습 시작")
        self.run_btn.clicked.connect(self.run)
        ctrl.addWidget(self.run_btn)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        self.fig = Figure(figsize=(12, 4))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(140)
        layout.addWidget(self.result_text)

        self.run()

    def run(self):
        self.run_btn.setEnabled(False)
        self.run_btn.setText("학습 중...")
        QApplication.processEvents()

        X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
        y = np.array([[0],[1],[1],[0]], dtype=float)

        model = MLPModel(hidden=self.hidden_spin.value(), lr=self.lr_spin.value())
        epochs = self.epoch_spin.value()
        for _ in range(epochs):
            model.train_step(X, y)

        # 정확도
        preds = (model.forward(X) > 0.5).astype(int)
        acc = np.mean(preds == y.astype(int)) * 100

        self.fig.clear()
        axes = self.fig.subplots(1, 3)

        # Loss 그래프
        ax = axes[0]
        ax.plot(model.loss_history, lw=1.5, color="steelblue")
        ax.set_yscale("log")
        ax.set_title(f"Training Loss (최종: {model.loss_history[-1]:.6f})", fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss (log)")
        ax.grid(True, alpha=0.3)

        # 결정 경계
        ax = axes[1]
        xs = np.linspace(-0.5, 1.5, 200)
        xx, yy = np.meshgrid(xs, xs)
        Z = model.forward(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        cf = ax.contourf(xx, yy, Z, levels=20, cmap="RdYlBu", alpha=0.85)
        self.fig.colorbar(cf, ax=ax, label="출력 확률")
        labels_arr = np.array([0,1,1,0])
        for pt, lbl in zip(X, labels_arr):
            c = "red" if lbl == 1 else "blue"
            m = "o" if lbl == 1 else "X"
            ax.scatter(pt[0], pt[1], c=c, marker=m, s=250,
                       edgecolors="black", lw=2, zorder=5)
        ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.5, 1.5)
        ax.set_title(f"결정 경계 (정확도 {acc:.0f}%)", fontweight="bold")
        ax.grid(True, alpha=0.3)

        # 은닉층 활성화
        ax = axes[2]
        ha = model.a1  # (4, hidden)
        im = ax.imshow(ha.T, cmap="viridis", aspect="auto")
        ax.set_yticks(range(self.hidden_spin.value()))
        ax.set_yticklabels([f"H{i+1}" for i in range(self.hidden_spin.value())])
        ax.set_xticks(range(4))
        ax.set_xticklabels(["(0,0)","(0,1)","(1,0)","(1,1)"])
        ax.set_title("은닉층 활성화", fontweight="bold")
        self.fig.colorbar(im, ax=ax, label="활성화")
        for i in range(self.hidden_spin.value()):
            for j in range(4):
                ax.text(j, i, f"{ha[j,i]:.2f}", ha="center", va="center",
                        color="white", fontsize=7, fontweight="bold")

        self.fig.tight_layout()
        self.canvas.draw()

        raw = model.forward(X)
        lines = ["입력  → 예측값   → 이진예측 → 정답"]
        for xi, pi, yi in zip(X, raw, y):
            lines.append(f"{xi} → {pi[0]:.4f} → {int(pi[0]>0.5)} → {int(yi[0])}")
        lines.append(f"\n정확도: {acc:.0f}%  |  최종 Loss: {model.loss_history[-1]:.6f}")
        lines.append("→ MLP + Backpropagation으로 XOR 문제 해결 성공!")
        self.result_text.setPlainText("\n".join(lines))

        self.run_btn.setEnabled(True)
        self.run_btn.setText("학습 시작")


# ══════════════════════════════════════════════════════════════
# Lab 5: Universal Approximation Tab
# ══════════════════════════════════════════════════════════════
class UniversalApproximator:
    def __init__(self, n_hidden, activation="tanh"):
        self.n_hidden = n_hidden
        self.activation = activation
        lim = np.sqrt(6 / (1 + n_hidden))
        self.W1 = np.random.uniform(-lim, lim, (1, n_hidden))
        self.b1 = np.zeros(n_hidden)
        lim2 = np.sqrt(6 / (n_hidden + 1))
        self.W2 = np.random.uniform(-lim2, lim2, (n_hidden, 1))
        self.b2 = np.zeros(1)

    def _act(self, z):
        return {"tanh": tanh_fn, "relu": relu, "sigmoid": sigmoid}[self.activation](z)

    def _act_deriv(self, z, a):
        if self.activation == "tanh":   return 1 - a**2
        if self.activation == "relu":   return (z > 0).astype(float)
        return a * (1 - a)

    def forward(self, x):
        z1 = x @ self.W1 + self.b1
        a1 = self._act(z1)
        return a1 @ self.W2 + self.b2

    def train(self, X, y, epochs=5000, lr=0.05):
        for _ in range(epochs):
            z1 = X @ self.W1 + self.b1
            a1 = self._act(z1)
            out = a1 @ self.W2 + self.b2
            dout = 2 * (out - y) / len(X)
            dW2 = a1.T @ dout
            db2 = np.sum(dout, axis=0)
            da1 = dout @ self.W2.T
            dz1 = da1 * self._act_deriv(z1, a1)
            dW1 = X.T @ dz1
            db1 = np.sum(dz1, axis=0)
            self.W2 -= lr * dW2; self.b2 -= lr * db2
            self.W1 -= lr * dW1; self.b1 -= lr * db1


class Lab5Tab(QWidget):
    def __init__(self):
        super().__init__()
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        title = QLabel("Lab 5: Universal Approximation Theorem")
        title.setFont(QFont("", 13, QFont.Weight.Bold))
        layout.addWidget(title)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("활성화 함수:"))
        self.act_combo = QComboBox()
        self.act_combo.addItems(["tanh", "relu", "sigmoid"])
        ctrl.addWidget(self.act_combo)

        ctrl.addWidget(QLabel("에폭:"))
        self.epoch_spin = QSpinBox()
        self.epoch_spin.setRange(500, 20000); self.epoch_spin.setSingleStep(500)
        self.epoch_spin.setValue(5000)
        ctrl.addWidget(self.epoch_spin)

        self.run_btn = QPushButton("학습 시작")
        self.run_btn.clicked.connect(self.run)
        ctrl.addWidget(self.run_btn)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        self.fig = Figure(figsize=(14, 9))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(90)
        self.result_text.setPlainText(
            "Universal Approximation Theorem (Cybenko, 1989):\n"
            "충분한 뉴런 수를 가진 1개의 은닉층 신경망은 어떤 연속 함수도 임의 정확도로 근사 가능!\n"
            "[학습 시작] 버튼을 눌러서 3개 / 10개 / 50개 뉴런 성능을 비교해보세요."
        )
        layout.addWidget(self.result_text)

        self.run()

    def run(self):
        self.run_btn.setEnabled(False)
        self.run_btn.setText("학습 중... (잠시 기다려주세요)")
        QApplication.processEvents()

        act = self.act_combo.currentText()
        epochs = self.epoch_spin.value()
        neuron_counts = [3, 10, 50]
        targets = [
            ("Sine Wave",      lambda x: np.sin(2 * np.pi * x)),
            ("Step Function",  lambda x: np.where(x < 0.5, 0.0, 1.0)),
            ("Complex Func",   lambda x: np.sin(2*np.pi*x) + 0.5*np.sin(4*np.pi*x)),
        ]

        x_train = np.linspace(0, 1, 100).reshape(-1, 1)
        x_test  = np.linspace(0, 1, 200).reshape(-1, 1)

        self.fig.clear()
        axes = self.fig.subplots(3, 3)
        mse_summary = []

        for col, (func_name, func) in enumerate(targets):
            y_train = func(x_train)
            y_true  = func(x_test)
            for row, n in enumerate(neuron_counts):
                lr = 0.05 if n < 20 else 0.01
                model = UniversalApproximator(n_hidden=n, activation=act)
                model.train(x_train, y_train, epochs=epochs, lr=lr)
                y_pred = model.forward(x_test)
                mse = float(np.mean((y_pred - y_true)**2))
                mse_summary.append(f"{func_name} / {n}뉴런: MSE={mse:.5f}")

                ax = axes[row, col]
                ax.plot(x_test, y_true, "b-", lw=2, label="True", alpha=0.7)
                ax.plot(x_test, y_pred, "r--", lw=2, label=f"NN({n})")
                ax.scatter(x_train[::10], y_train[::10], c="green", s=20, alpha=0.5)
                heading = func_name if row == 0 else ""
                ax.set_title(f"{heading}\n{n} neurons  MSE={mse:.4f}",
                             fontsize=9, fontweight="bold")
                ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        self.fig.tight_layout()
        self.canvas.draw()
        self.result_text.setPlainText(
            "학습 완료! MSE 요약:\n" + "\n".join(mse_summary) +
            "\n→ 뉴런이 많을수록(50개) MSE가 작아져 함수를 더 정확히 근사합니다."
        )
        self.run_btn.setEnabled(True)
        self.run_btn.setText("학습 시작")


# ══════════════════════════════════════════════════════════════
# 메인 윈도우
# ══════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Week 3: 신경망 기초 — PySide6 Interactive Lab")
        self.resize(1200, 820)

        tabs = QTabWidget()
        tabs.addTab(Lab1Tab(), "Lab 1: Perceptron")
        tabs.addTab(Lab2Tab(), "Lab 2: Activation Fn")
        tabs.addTab(Lab3Tab(), "Lab 3: Forward Prop")
        tabs.addTab(Lab4Tab(), "Lab 4: MLP / Backprop")
        tabs.addTab(Lab5Tab(), "Lab 5: Universal Approx")

        self.setCentralWidget(tabs)

        # 상태바
        self.statusBar().showMessage(
            "AI와 머신러닝 (PH2002141-033) | Week 3 Homework | PySide6 GUI | 김충현"
        )


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
