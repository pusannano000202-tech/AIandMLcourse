"""
Week 4 Homework — 물리 데이터로 학습하기
PySide6 + TensorFlow Interactive GUI Application

Labs:
  Lab 1: 1D Function Approximation (Universal Approximation Theorem)
  Lab 2: Projectile Motion Regression (포물선 운동)
  Lab 3: Overfitting vs Underfitting (과적합/과소적합)
  Lab 4: Pendulum Period Prediction (진자 주기 예측)

Superpowers:
  - Real-time loss streaming (QThread + Signal)
  - Stop training mid-way
  - Interactive post-training prediction sliders
  - Physics equation panel per lab
  - Model architecture summary panel
"""

import sys
import os
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress TF logs

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QDoubleSpinBox, QTextEdit, QGroupBox,
    QComboBox, QProgressBar, QSplitter, QFrame,
    QSlider, QGridLayout, QSizePolicy,
)
from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtGui import QFont, QColor, QPalette

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import tensorflow as tf
from tensorflow import keras


# ── 한글 폰트 ──────────────────────────────────────────────────
def _setup_font():
    names = [f.name for f in fm.fontManager.ttflist]
    for fn in ["Malgun Gothic", "Gulim", "Batang", "AppleGothic"]:
        if fn in names:
            plt.rcParams["font.family"] = fn
            break
    plt.rcParams["axes.unicode_minus"] = False

_setup_font()

# ── 공통 색상 팔레트 ────────────────────────────────────────────
CLR_PRIMARY   = "#1565C0"
CLR_SUCCESS   = "#2E7D32"
CLR_WARN      = "#E65100"
CLR_BG        = "#FAFAFA"
CLR_STOP      = "#C62828"


# ══════════════════════════════════════════════════════════════
# 공통 유틸
# ══════════════════════════════════════════════════════════════
def make_title(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setFont(QFont("", 13, QFont.Weight.Bold))
    lbl.setStyleSheet(f"color: {CLR_PRIMARY}; padding: 4px 0;")
    return lbl

def make_group(title: str) -> QGroupBox:
    g = QGroupBox(title)
    g.setFont(QFont("", 10, QFont.Weight.Bold))
    return g

def make_eq_box(html: str) -> QTextEdit:
    t = QTextEdit()
    t.setReadOnly(True)
    t.setHtml(html)
    t.setMaximumHeight(130)
    t.setStyleSheet("background:#f0f4ff; border:1px solid #90A4AE; border-radius:4px;")
    return t


def make_canvas(w=11, h=4.5) -> tuple[Figure, FigureCanvas]:
    fig = Figure(figsize=(w, h), tight_layout=True)
    canvas = FigureCanvas(fig)
    canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    return fig, canvas


# ══════════════════════════════════════════════════════════════
# 커스텀 Keras Callback — epoch마다 signal emit
# ══════════════════════════════════════════════════════════════
class StreamCallback(keras.callbacks.Callback):
    """학습 중 epoch 종료마다 loss를 signal로 전달"""
    def __init__(self, signal, total_epochs, freq=20):
        super().__init__()
        self._sig = signal
        self._total = total_epochs
        self._freq = freq

    def on_epoch_end(self, epoch, logs=None):
        if logs and (epoch % self._freq == 0 or epoch == self._total - 1):
            self._sig.emit(epoch, float(logs.get("loss", 0)),
                           float(logs.get("val_loss", -1)))


# ══════════════════════════════════════════════════════════════
# Lab 1 — 1D Function Approximation
# ══════════════════════════════════════════════════════════════
class Lab1Worker(QThread):
    epoch_update = Signal(int, float, float)  # epoch, loss, val_loss
    finished     = Signal(object, object, object, object)  # x_test, y_true, y_pred, history

    FUNC_MAP = {
        "sin(x)":              lambda x: np.sin(x),
        "cos(x)+0.5sin(2x)":   lambda x: np.cos(x) + 0.5 * np.sin(2 * x),
        "x·sin(x)":            lambda x: x * np.sin(x),
        "Extreme (multi-freq)": lambda x: (np.sin(x) + 0.5*np.sin(2*x) +
                                            0.3*np.cos(3*x) + 0.2*np.sin(5*x)),
    }
    ARCH_MAP = {
        "Small [32]":           [32],
        "Medium [64,64]":       [64, 64],
        "Large [128,128]":      [128, 128],
        "XL [128,128,64]":      [128, 128, 64],
        "XXL [256,256,128,64]": [256, 256, 128, 64],
    }

    def __init__(self, func_name, arch_name, epochs, lr):
        super().__init__()
        self.func_name = func_name
        self.arch_name = arch_name
        self.epochs    = epochs
        self.lr        = lr
        self._stop     = False

    def stop(self): self._stop = True

    def run(self):
        fn     = self.FUNC_MAP[self.func_name]
        layers = self.ARCH_MAP[self.arch_name]

        x_train = np.linspace(-2*np.pi, 2*np.pi, 300).reshape(-1, 1)
        x_test  = np.linspace(-2*np.pi, 2*np.pi, 500).reshape(-1, 1)
        y_train = fn(x_train)
        y_true  = fn(x_test)

        model = keras.Sequential()
        model.add(keras.layers.Input(shape=(1,)))
        for u in layers:
            model.add(keras.layers.Dense(u, activation="tanh"))
        model.add(keras.layers.Dense(1, activation="linear"))
        model.compile(optimizer=keras.optimizers.Adam(self.lr), loss="mse")

        cb = StreamCallback(self.epoch_update, self.epochs, freq=max(1, self.epochs // 100))
        stop_cb = keras.callbacks.LambdaCallback(
            on_epoch_end=lambda e, l: self._stop and self.model.stop_training.__setattr__
        )

        class StopCheck(keras.callbacks.Callback):
            def __init__(s): super().__init__()
            def on_epoch_end(s, epoch, logs=None):
                if self._stop:
                    s.model.stop_training = True

        history = model.fit(
            x_train, y_train,
            epochs=self.epochs, batch_size=32, verbose=0,
            callbacks=[cb, StopCheck(),
                       keras.callbacks.ReduceLROnPlateau(
                           monitor="loss", factor=0.85, patience=50,
                           min_lr=1e-6, verbose=0)]
        )
        y_pred = model.predict(x_test, verbose=0)
        self.finished.emit(x_test, y_true, y_pred, history.history)


class Lab1Tab(QWidget):
    def __init__(self):
        super().__init__()
        self._worker = None
        self._loss_hist: list[float] = []
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(6)
        root.addWidget(make_title("Lab 1: 1D 함수 근사 (Universal Approximation Theorem)"))
        root.addWidget(make_eq_box(
            "<b>Universal Approximation Theorem (Cybenko, 1989)</b><br>"
            "충분히 넓은 은닉층을 가진 NN은 어떤 연속 함수도 임의 정확도로 근사 가능<br>"
            "<b>구조:</b> 입력 x → [Dense(tanh)] × N → Dense(linear) → 출력 y<br>"
            "<b>Loss:</b> MSE = (1/n)Σ(y_true − y_pred)²"
        ))

        # 컨트롤
        ctrl_grp = make_group("학습 설정")
        ctrl_lay = QHBoxLayout(ctrl_grp)

        ctrl_lay.addWidget(QLabel("함수:"))
        self.func_cb = QComboBox()
        self.func_cb.addItems(list(Lab1Worker.FUNC_MAP.keys()))
        ctrl_lay.addWidget(self.func_cb)

        ctrl_lay.addWidget(QLabel("아키텍처:"))
        self.arch_cb = QComboBox()
        self.arch_cb.addItems(list(Lab1Worker.ARCH_MAP.keys()))
        self.arch_cb.setCurrentIndex(2)
        ctrl_lay.addWidget(self.arch_cb)

        ctrl_lay.addWidget(QLabel("에폭:"))
        self.epoch_sp = QSpinBox()
        self.epoch_sp.setRange(100, 10000); self.epoch_sp.setSingleStep(200)
        self.epoch_sp.setValue(3000)
        ctrl_lay.addWidget(self.epoch_sp)

        ctrl_lay.addWidget(QLabel("LR:"))
        self.lr_sp = QDoubleSpinBox()
        self.lr_sp.setRange(1e-4, 0.1); self.lr_sp.setSingleStep(0.001)
        self.lr_sp.setDecimals(4); self.lr_sp.setValue(0.01)
        ctrl_lay.addWidget(self.lr_sp)

        self.run_btn  = QPushButton("학습 시작")
        self.stop_btn = QPushButton("정지")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(f"background:{CLR_STOP}; color:white; font-weight:bold;")
        self.run_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self._stop)
        ctrl_lay.addWidget(self.run_btn)
        ctrl_lay.addWidget(self.stop_btn)
        ctrl_lay.addStretch()
        root.addWidget(ctrl_grp)

        # 진행률
        self.prog = QProgressBar(); self.prog.setMaximum(100); self.prog.setValue(0)
        root.addWidget(self.prog)

        # 캔버스
        self.fig, self.canvas = make_canvas(12, 4.5)
        root.addWidget(self.canvas)

        # 결과
        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(90)
        root.addWidget(self.log)

        self._draw_placeholder()

    def _draw_placeholder(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, "학습 시작 버튼을 누르세요",
                ha="center", va="center", fontsize=14, color="gray",
                transform=ax.transAxes)
        ax.axis("off")
        self.canvas.draw()

    def _start(self):
        self._loss_hist.clear()
        self.prog.setValue(0)
        self._worker = Lab1Worker(
            self.func_cb.currentText(), self.arch_cb.currentText(),
            self.epoch_sp.value(), self.lr_sp.value()
        )
        self._worker.epoch_update.connect(self._on_epoch)
        self._worker.finished.connect(self._on_done)
        self._worker.start()
        self.run_btn.setEnabled(False); self.stop_btn.setEnabled(True)
        self.log.setPlainText("학습 중...")

    def _stop(self):
        if self._worker: self._worker.stop()

    def _on_epoch(self, epoch, loss, _val):
        self._loss_hist.append(loss)
        pct = int(epoch / self.epoch_sp.value() * 100)
        self.prog.setValue(pct)
        self.log.setPlainText(f"Epoch {epoch+1} | Loss: {loss:.6f}")

        # 실시간 loss 플롯
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.plot(self._loss_hist, color=CLR_PRIMARY, lw=1.5)
        ax.set_yscale("log"); ax.set_xlabel("Epoch (x sample_freq)"); ax.set_ylabel("MSE Loss")
        ax.set_title("실시간 학습 Loss", fontweight="bold"); ax.grid(True, alpha=0.3)
        self.canvas.draw()

    def _on_done(self, x_test, y_true, y_pred, hist):
        self.prog.setValue(100)
        self.run_btn.setEnabled(True); self.stop_btn.setEnabled(False)

        mse = float(np.mean((y_pred - y_true) ** 2))
        max_e = float(np.max(np.abs(y_pred - y_true)))

        self.fig.clear()
        axes = self.fig.subplots(1, 3)

        # 함수 근사
        ax = axes[0]
        ax.plot(x_test, y_true, "b-", lw=2.5, label="True", alpha=0.8)
        ax.plot(x_test, y_pred, "r--", lw=2, label="NN Prediction")
        ax.set_title(f"함수 근사\nMSE={mse:.6f}", fontweight="bold")
        ax.legend(); ax.grid(True, alpha=0.3)

        # Loss 곡선
        ax = axes[1]
        ax.plot(hist["loss"], color=CLR_SUCCESS, lw=1.5)
        ax.set_yscale("log"); ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
        ax.set_title("Training Loss", fontweight="bold"); ax.grid(True, alpha=0.3)

        # 절대 오차
        ax = axes[2]
        err = np.abs(y_pred - y_true)
        ax.plot(x_test, err, color=CLR_WARN, lw=1.5)
        ax.fill_between(x_test.flatten(), 0, err.flatten(), color=CLR_WARN, alpha=0.25)
        ax.set_title(f"절대 오차\nMax={max_e:.6f}", fontweight="bold")
        ax.set_xlabel("x"); ax.set_ylabel("|error|"); ax.grid(True, alpha=0.3)

        self.fig.tight_layout()
        self.canvas.draw()
        self.log.setPlainText(
            f"학습 완료! | 함수: {self.func_cb.currentText()} | "
            f"아키텍처: {self.arch_cb.currentText()}\n"
            f"MSE: {mse:.8f} | Max Error: {max_e:.8f} | "
            f"Epochs 실행: {len(hist['loss'])}"
        )


# ══════════════════════════════════════════════════════════════
# Lab 2 — Projectile Motion Regression
# ══════════════════════════════════════════════════════════════
class Lab2Worker(QThread):
    epoch_update = Signal(int, float, float)
    finished     = Signal(object, object, object, object, object)  # model, X_test, Y_test, history, scaler

    def __init__(self, n_samples, epochs, lr):
        super().__init__()
        self.n_samples = n_samples
        self.epochs    = epochs
        self.lr        = lr
        self._stop     = False

    def stop(self): self._stop = True

    def run(self):
        g = 9.81
        np.random.seed(42)
        n = self.n_samples

        v0    = np.random.uniform(10, 50, n)
        theta = np.random.uniform(20, 70, n)
        tr    = np.deg2rad(theta)
        t_max = 2 * v0 * np.sin(tr) / g
        t     = np.random.uniform(0, t_max * 0.9, n)
        x_pos = v0 * np.cos(tr) * t
        y_pos = v0 * np.sin(tr) * t - 0.5 * g * t ** 2

        valid = y_pos >= 0
        X = np.column_stack([v0[valid], theta[valid], t[valid]])
        Y = np.column_stack([x_pos[valid], y_pos[valid]])

        # 정규화
        X_mean, X_std = X.mean(0), X.std(0)
        Y_mean, Y_std = Y.mean(0), Y.std(0)
        Xn = (X - X_mean) / X_std
        Yn = (Y - Y_mean) / Y_std

        split = int(len(Xn) * 0.8)
        X_tr, X_te = Xn[:split], Xn[split:]
        Y_tr, Y_te = Yn[:split], Yn[split:]

        model = keras.Sequential([
            keras.layers.Input(shape=(3,)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(64,  activation="relu"),
            keras.layers.Dense(2,   activation="linear"),
        ])
        model.compile(optimizer=keras.optimizers.Adam(self.lr), loss="mse", metrics=["mae"])

        class StopCheck(keras.callbacks.Callback):
            def on_epoch_end(s, ep, logs=None):
                if self._stop: s.model.stop_training = True

        cb = StreamCallback(self.epoch_update, self.epochs, freq=max(1, self.epochs // 100))
        model.fit(X_tr, Y_tr, epochs=self.epochs, batch_size=64, verbose=0,
                  validation_split=0.1,
                  callbacks=[cb, StopCheck(),
                             keras.callbacks.ReduceLROnPlateau(
                                 monitor="val_loss", factor=0.85,
                                 patience=30, min_lr=1e-6, verbose=0)])

        scaler = (X_mean, X_std, Y_mean, Y_std)
        self.finished.emit(model, X_te, Y_te, model.history.history, scaler)


class Lab2Tab(QWidget):
    def __init__(self):
        super().__init__()
        self._worker = None
        self._model  = None
        self._scaler = None
        self._loss_hist: list = []
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self); root.setSpacing(6)
        root.addWidget(make_title("Lab 2: 포물선 운동 회귀 (Projectile Motion Regression)"))
        root.addWidget(make_eq_box(
            "<b>포물선 운동 물리 법칙</b><br>"
            "x(t) = v₀·cos(θ)·t &nbsp;&nbsp; y(t) = v₀·sin(θ)·t − ½g·t²<br>"
            "<b>입력:</b> (v₀, θ, t) &nbsp; <b>출력:</b> (x, y)<br>"
            "<b>목표:</b> NN이 물리 방정식 없이 데이터로부터 포물선 운동을 학습"
        ))

        ctrl_grp = make_group("학습 설정")
        cl = QHBoxLayout(ctrl_grp)
        cl.addWidget(QLabel("샘플 수:"))
        self.n_sp = QSpinBox(); self.n_sp.setRange(500, 10000); self.n_sp.setSingleStep(500)
        self.n_sp.setValue(2000); cl.addWidget(self.n_sp)
        cl.addWidget(QLabel("에폭:"))
        self.ep_sp = QSpinBox(); self.ep_sp.setRange(100, 5000); self.ep_sp.setSingleStep(100)
        self.ep_sp.setValue(500); cl.addWidget(self.ep_sp)
        cl.addWidget(QLabel("LR:"))
        self.lr_sp = QDoubleSpinBox(); self.lr_sp.setRange(1e-4, 0.01)
        self.lr_sp.setDecimals(4); self.lr_sp.setValue(0.001); cl.addWidget(self.lr_sp)
        self.run_btn  = QPushButton("학습 시작")
        self.stop_btn = QPushButton("정지")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(f"background:{CLR_STOP}; color:white; font-weight:bold;")
        self.run_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(lambda: self._worker and self._worker.stop())
        cl.addWidget(self.run_btn); cl.addWidget(self.stop_btn); cl.addStretch()
        root.addWidget(ctrl_grp)

        self.prog = QProgressBar(); self.prog.setValue(0); root.addWidget(self.prog)

        self.fig, self.canvas = make_canvas(12, 4.5)
        root.addWidget(self.canvas)

        # 인터랙티브 예측 패널
        pred_grp = make_group("인터랙티브 예측 (학습 후 활성화)")
        pl = QHBoxLayout(pred_grp)
        for label, attr, rng, val in [
            ("v₀ (m/s):", "v0_sp", (10, 50), 30),
            ("θ (deg):",  "th_sp", (20, 70), 45),
            ("t (s):",    "t_sp",  (0.1, 6), 2.0),
        ]:
            pl.addWidget(QLabel(label))
            sp = QDoubleSpinBox(); sp.setRange(*rng); sp.setValue(val)
            sp.setSingleStep(0.5); sp.valueChanged.connect(self._predict_live)
            setattr(self, attr, sp); pl.addWidget(sp)
        self.pred_lbl = QLabel("x=?, y=?")
        self.pred_lbl.setFont(QFont("", 11, QFont.Weight.Bold))
        self.pred_lbl.setStyleSheet(f"color:{CLR_SUCCESS};")
        pl.addWidget(self.pred_lbl); pl.addStretch()
        root.addWidget(pred_grp)

        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(80)
        root.addWidget(self.log)
        self._draw_placeholder()

    def _draw_placeholder(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, "학습 시작 버튼을 누르세요",
                ha="center", va="center", fontsize=14, color="gray", transform=ax.transAxes)
        ax.axis("off"); self.canvas.draw()

    def _start(self):
        self._loss_hist.clear(); self.prog.setValue(0)
        self._worker = Lab2Worker(self.n_sp.value(), self.ep_sp.value(), self.lr_sp.value())
        self._worker.epoch_update.connect(self._on_epoch)
        self._worker.finished.connect(self._on_done)
        self._worker.start()
        self.run_btn.setEnabled(False); self.stop_btn.setEnabled(True)

    def _on_epoch(self, epoch, loss, val_loss):
        self._loss_hist.append((loss, val_loss if val_loss >= 0 else loss))
        self.prog.setValue(int(epoch / self.ep_sp.value() * 100))
        self.log.setPlainText(f"Epoch {epoch+1} | Loss: {loss:.5f} | Val: {val_loss:.5f}")
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        losses = [l[0] for l in self._loss_hist]
        vals   = [l[1] for l in self._loss_hist]
        ax.plot(losses, label="Train Loss", color=CLR_PRIMARY, lw=1.5)
        ax.plot(vals,   label="Val Loss",   color=CLR_WARN,    lw=1.5, ls="--")
        ax.set_yscale("log"); ax.legend(); ax.grid(True, alpha=0.3)
        ax.set_title("실시간 Loss (Train vs Val)", fontweight="bold")
        self.canvas.draw()

    def _on_done(self, model, X_te, Y_te, hist, scaler):
        self.prog.setValue(100)
        self.run_btn.setEnabled(True); self.stop_btn.setEnabled(False)
        self._model  = model
        self._scaler = scaler

        Xm, Xs, Ym, Ys = scaler
        Y_pred_n = model.predict(X_te, verbose=0)
        Y_pred   = Y_pred_n * Ys + Ym
        Y_true   = X_te * Xs + Xm  # wrong, need original Y
        # recompute true Y
        Y_true_raw = Y_te * Ys + Ym
        mse = float(np.mean((Y_pred - Y_true_raw) ** 2))

        self.fig.clear()
        axes = self.fig.subplots(1, 3)

        # x 예측 vs 실제
        ax = axes[0]
        ax.scatter(Y_true_raw[:200, 0], Y_pred[:200, 0], s=8, alpha=0.5, color=CLR_PRIMARY)
        lim = [min(Y_true_raw[:, 0].min(), Y_pred[:, 0].min()),
               max(Y_true_raw[:, 0].max(), Y_pred[:, 0].max())]
        ax.plot(lim, lim, "r--", lw=1.5, label="Perfect")
        ax.set_xlabel("True x (m)"); ax.set_ylabel("Predicted x (m)")
        ax.set_title("x 위치 예측", fontweight="bold"); ax.legend(); ax.grid(True, alpha=0.3)

        # y 예측 vs 실제
        ax = axes[1]
        ax.scatter(Y_true_raw[:200, 1], Y_pred[:200, 1], s=8, alpha=0.5, color=CLR_SUCCESS)
        lim = [min(Y_true_raw[:, 1].min(), Y_pred[:, 1].min()),
               max(Y_true_raw[:, 1].max(), Y_pred[:, 1].max())]
        ax.plot(lim, lim, "r--", lw=1.5, label="Perfect")
        ax.set_xlabel("True y (m)"); ax.set_ylabel("Predicted y (m)")
        ax.set_title("y 위치 예측", fontweight="bold"); ax.legend(); ax.grid(True, alpha=0.3)

        # Loss 곡선
        ax = axes[2]
        ax.plot(hist["loss"],     label="Train", color=CLR_PRIMARY, lw=1.5)
        ax.plot(hist["val_loss"], label="Val",   color=CLR_WARN,    lw=1.5, ls="--")
        ax.set_yscale("log"); ax.legend(); ax.grid(True, alpha=0.3)
        ax.set_title("Training / Validation Loss", fontweight="bold")

        self.fig.tight_layout(); self.canvas.draw()
        self.log.setPlainText(
            f"학습 완료! | 샘플 {self.n_sp.value()}개 | MSE: {mse:.4f}\n"
            f"학습 Epoch: {len(hist['loss'])} | 이제 아래 슬라이더로 실시간 예측 가능!"
        )

    def _predict_live(self):
        if self._model is None: return
        Xm, Xs, Ym, Ys = self._scaler
        v0 = self.v0_sp.value(); th = self.th_sp.value(); t = self.t_sp.value()
        inp = (np.array([[v0, th, t]]) - Xm) / Xs
        out = self._model.predict(inp, verbose=0) * Ys + Ym
        self.pred_lbl.setText(f"x = {out[0,0]:.2f} m | y = {out[0,1]:.2f} m")


# ══════════════════════════════════════════════════════════════
# Lab 3 — Overfitting vs Underfitting
# ══════════════════════════════════════════════════════════════
class Lab3Worker(QThread):
    epoch_update = Signal(int, float, float, str)   # epoch, loss, val, model_name
    finished     = Signal(object)                   # dict of results

    def __init__(self, n_train, epochs, noise):
        super().__init__()
        self.n_train = n_train
        self.epochs  = epochs
        self.noise   = noise
        self._stop   = False

    def stop(self): self._stop = True

    def run(self):
        np.random.seed(42)
        def f_true(x): return np.sin(2 * x) + 0.5 * x

        x_tr = np.random.uniform(-2, 2, self.n_train).reshape(-1, 1)
        y_tr = f_true(x_tr) + np.random.normal(0, self.noise, x_tr.shape)
        x_va = np.random.uniform(-2, 2, 50).reshape(-1, 1)
        y_va = f_true(x_va) + np.random.normal(0, self.noise, x_va.shape)
        x_te = np.linspace(-2, 2, 200).reshape(-1, 1)
        y_te = f_true(x_te)

        configs = {
            "Underfit": keras.Sequential([
                keras.layers.Input(shape=(1,)),
                keras.layers.Dense(4,   activation="relu"),
                keras.layers.Dense(1)]),
            "Good Fit": keras.Sequential([
                keras.layers.Input(shape=(1,)),
                keras.layers.Dense(32,  activation="relu"),
                keras.layers.Dropout(0.1),
                keras.layers.Dense(16,  activation="relu"),
                keras.layers.Dense(1)]),
            "Overfit": keras.Sequential([
                keras.layers.Input(shape=(1,)),
                keras.layers.Dense(256, activation="relu"),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dense(64,  activation="relu"),
                keras.layers.Dense(32,  activation="relu"),
                keras.layers.Dense(1)]),
        }
        results = {"x_tr": x_tr, "y_tr": y_tr, "x_te": x_te, "y_te": y_te,
                   "histories": {}, "predictions": {}, "metrics": {}}

        for name, mdl in configs.items():
            if self._stop: break
            mdl.compile(optimizer=keras.optimizers.Adam(0.002), loss="mse", metrics=["mae"])

            class TaggedCB(keras.callbacks.Callback):
                def __init__(s, sig, nm, total):
                    super().__init__(); s._sig=sig; s._nm=nm; s._tot=total; s._freq=max(1,total//80)
                def on_epoch_end(s, ep, logs=None):
                    if logs and ep % s._freq == 0:
                        s._sig.emit(ep, float(logs.get("loss",0)),
                                    float(logs.get("val_loss",-1)), s._nm)
                    if self._stop: s.model.stop_training = True

            h = mdl.fit(x_tr, y_tr, epochs=self.epochs, batch_size=32, verbose=0,
                        validation_data=(x_va, y_va),
                        callbacks=[TaggedCB(self.epoch_update, name, self.epochs)])
            y_pred = mdl.predict(x_te, verbose=0)
            test_mse, test_mae = mdl.evaluate(x_te, y_te, verbose=0)
            results["histories"][name]    = h.history
            results["predictions"][name]  = y_pred
            results["metrics"][name]      = {"mse": test_mse, "mae": test_mae,
                                             "final_train": h.history["loss"][-1],
                                             "final_val":   h.history["val_loss"][-1]}
        self.finished.emit(results)


class Lab3Tab(QWidget):
    CLR_MAP = {"Underfit": "#1565C0", "Good Fit": "#2E7D32", "Overfit": "#C62828"}

    def __init__(self):
        super().__init__()
        self._worker = None
        self._hist_buf: dict = {"Underfit": [], "Good Fit": [], "Overfit": []}
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self); root.setSpacing(6)
        root.addWidget(make_title("Lab 3: 과적합 vs 과소적합 (Overfitting vs Underfitting)"))
        root.addWidget(make_eq_box(
            "<b>핵심 개념</b><br>"
            "Underfitting: 모델이 너무 단순 → Train/Val Loss 모두 높음<br>"
            "Good Fit: 적절한 복잡도 + Dropout → 일반화 최적<br>"
            "Overfitting: 모델이 너무 복잡 → Train Loss 낮지만 Val Loss 높음<br>"
            "<b>해결책:</b> Dropout, L2 Regularization, Early Stopping, 데이터 증가"
        ))

        ctrl_grp = make_group("실험 설정")
        cl = QHBoxLayout(ctrl_grp)
        cl.addWidget(QLabel("학습 데이터:"))
        self.n_sp = QSpinBox(); self.n_sp.setRange(30,500); self.n_sp.setValue(100)
        cl.addWidget(self.n_sp)
        cl.addWidget(QLabel("에폭:"))
        self.ep_sp = QSpinBox(); self.ep_sp.setRange(100,3000); self.ep_sp.setSingleStep(100)
        self.ep_sp.setValue(500); cl.addWidget(self.ep_sp)
        cl.addWidget(QLabel("노이즈:"))
        self.noise_sp = QDoubleSpinBox(); self.noise_sp.setRange(0.05, 1.0)
        self.noise_sp.setSingleStep(0.05); self.noise_sp.setValue(0.3); cl.addWidget(self.noise_sp)
        self.run_btn  = QPushButton("학습 시작 (3 모델)")
        self.stop_btn = QPushButton("정지")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(f"background:{CLR_STOP}; color:white; font-weight:bold;")
        self.run_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(lambda: self._worker and self._worker.stop())
        cl.addWidget(self.run_btn); cl.addWidget(self.stop_btn); cl.addStretch()
        root.addWidget(ctrl_grp)

        self.prog_lbl = QLabel("대기 중...")
        root.addWidget(self.prog_lbl)
        self.prog = QProgressBar(); self.prog.setValue(0); root.addWidget(self.prog)

        self.fig, self.canvas = make_canvas(13, 4.5)
        root.addWidget(self.canvas)
        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(90)
        root.addWidget(self.log)
        self._draw_placeholder()

    def _draw_placeholder(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, "학습 시작 버튼을 누르세요",
                ha="center", va="center", fontsize=14, color="gray", transform=ax.transAxes)
        ax.axis("off"); self.canvas.draw()

    def _start(self):
        for k in self._hist_buf: self._hist_buf[k].clear()
        self.prog.setValue(0)
        self._worker = Lab3Worker(self.n_sp.value(), self.ep_sp.value(), self.noise_sp.value())
        self._worker.epoch_update.connect(self._on_epoch)
        self._worker.finished.connect(self._on_done)
        self._worker.start()
        self.run_btn.setEnabled(False); self.stop_btn.setEnabled(True)

    def _on_epoch(self, epoch, loss, val, name):
        self._hist_buf[name].append((loss, val))
        self.prog_lbl.setText(f"학습 중: {name} | Epoch {epoch+1}")
        pct = {"Underfit": 0, "Good Fit": 33, "Overfit": 66}.get(name, 0)
        pct += int(epoch / self.ep_sp.value() * 33)
        self.prog.setValue(min(pct, 99))

        self.fig.clear()
        ax = self.fig.add_subplot(111)
        for nm, buf in self._hist_buf.items():
            if buf:
                ax.plot([b[1] for b in buf], label=f"{nm} Val",
                        color=self.CLR_MAP[nm], lw=1.5)
        ax.set_yscale("log"); ax.legend(); ax.grid(True, alpha=0.3)
        ax.set_title("실시간 Validation Loss 비교", fontweight="bold")
        self.canvas.draw()

    def _on_done(self, results):
        self.prog.setValue(100); self.prog_lbl.setText("완료!")
        self.run_btn.setEnabled(True); self.stop_btn.setEnabled(False)

        x_tr  = results["x_tr"]; y_tr = results["y_tr"]
        x_te  = results["x_te"]; y_te = results["y_te"]
        preds = results["predictions"]
        hists = results["histories"]
        mets  = results["metrics"]

        self.fig.clear()
        axes = self.fig.subplots(1, 3)

        # 함수 근사 비교
        ax = axes[0]
        ax.scatter(x_tr, y_tr, s=15, alpha=0.4, color="gray", label="Train data")
        ax.plot(x_te, y_te, "k-", lw=2.5, label="True", alpha=0.8)
        for nm, yp in preds.items():
            ax.plot(x_te, yp, "--", lw=2, color=self.CLR_MAP[nm], label=nm)
        ax.set_title("모델 비교", fontweight="bold"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # Train vs Val Loss
        ax = axes[1]
        for nm, h in hists.items():
            ax.plot(h["loss"],     color=self.CLR_MAP[nm], lw=1.5, label=f"{nm} Train")
            ax.plot(h["val_loss"], color=self.CLR_MAP[nm], lw=1.5, ls=":", label=f"{nm} Val")
        ax.set_yscale("log"); ax.set_title("Train vs Val Loss", fontweight="bold")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        # 성능 표
        ax = axes[2]; ax.axis("off")
        headers = ["모델", "Train MSE", "Val MSE", "Test MSE"]
        rows = [[nm,
                 f"{m['final_train']:.5f}",
                 f"{m['final_val']:.5f}",
                 f"{m['mse']:.5f}"] for nm, m in mets.items()]
        tbl = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 2.2)
        for j in range(4):
            tbl[(0, j)].set_facecolor("#37474F")
            tbl[(0, j)].set_text_props(color="white", weight="bold")
        row_colors = ["#BBDEFB", "#C8E6C9", "#FFCDD2"]
        for i, rc in enumerate(row_colors, 1):
            for j in range(4): tbl[(i, j)].set_facecolor(rc)
        ax.set_title("성능 요약", fontweight="bold", pad=20)

        self.fig.tight_layout(); self.canvas.draw()
        lines = ["학습 완료!"]
        for nm, m in mets.items():
            lines.append(f"  {nm:12s}: Train={m['final_train']:.5f} | "
                         f"Val={m['final_val']:.5f} | Test MSE={m['mse']:.5f}")
        self.log.setPlainText("\n".join(lines))


# ══════════════════════════════════════════════════════════════
# Lab 4 — Pendulum Period Prediction
# ══════════════════════════════════════════════════════════════
class Lab4Worker(QThread):
    epoch_update = Signal(int, float, float)
    finished     = Signal(object, object, object, object)  # model, X_te, Y_te, hist

    def __init__(self, n_samples, epochs, lr):
        super().__init__()
        self.n_samples = n_samples
        self.epochs    = epochs
        self.lr        = lr
        self._stop     = False

    def stop(self): self._stop = True

    def run(self):
        g = 9.81
        np.random.seed(42)
        n = self.n_samples

        L      = np.random.uniform(0.5, 3.0, n)
        theta0 = np.random.uniform(5, 80, n)
        theta0_rad = np.deg2rad(theta0)
        T_small = 2 * np.pi * np.sqrt(L / g)
        correction = 1 + (1/16)*theta0_rad**2 + (11/3072)*theta0_rad**4
        T_true  = T_small * correction
        T_noisy = T_true * (1 + np.random.normal(0, 0.01, n))

        X = np.column_stack([L, theta0])
        Y = T_noisy.reshape(-1, 1)

        # 정규화
        Xm, Xs = X.mean(0), X.std(0)
        Ym, Ys = float(Y.mean()), float(Y.std())
        Xn = (X - Xm) / Xs
        Yn = (Y - Ym) / Ys

        split = int(n * 0.8)
        X_tr, X_te = Xn[:split], Xn[split:]
        Y_tr, Y_te = Yn[:split], Yn[split:]
        Y_te_raw   = Y[split:]

        model = keras.Sequential([
            keras.layers.Input(shape=(2,)),
            keras.layers.Dense(64,  activation="relu"),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(32,  activation="relu"),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(16,  activation="relu"),
            keras.layers.Dense(1,   activation="linear"),
        ])
        model.compile(optimizer=keras.optimizers.Adam(self.lr), loss="mse", metrics=["mae"])

        class StopCB(keras.callbacks.Callback):
            def on_epoch_end(s, ep, logs=None):
                if self._stop: s.model.stop_training = True

        cb = StreamCallback(self.epoch_update, self.epochs, freq=max(1, self.epochs // 100))
        model.fit(X_tr, Y_tr, epochs=self.epochs, batch_size=64, verbose=0,
                  validation_split=0.1,
                  callbacks=[cb, StopCB(),
                             keras.callbacks.ReduceLROnPlateau(
                                 monitor="val_loss", factor=0.85,
                                 patience=30, min_lr=1e-6, verbose=0)])

        # 역정규화 포함 scaler 저장
        model._scaler = (Xm, Xs, Ym, Ys)
        model._X_te_raw = X[split:]          # 원본 X (L, theta0)
        model._Y_te_raw = Y_te_raw.flatten() # 원본 T
        self.finished.emit(model, X_te, Y_te, model.history.history)


class Lab4Tab(QWidget):
    def __init__(self):
        super().__init__()
        self._worker = None
        self._model  = None
        self._loss_hist: list = []
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self); root.setSpacing(6)
        root.addWidget(make_title("Lab 4: 진자 주기 예측 (Pendulum Period Prediction)"))
        root.addWidget(make_eq_box(
            "<b>진자 주기 물리 법칙</b><br>"
            "소각도: T = 2π√(L/g) &nbsp;&nbsp; "
            "대각도 보정: T ≈ T₀[1 + θ²/16 + 11θ⁴/3072 + ...]<br>"
            "<b>입력:</b> (L, θ₀) &nbsp; <b>출력:</b> T (주기, 초)<br>"
            "<b>목표:</b> NN이 비선형 보정항을 데이터로부터 학습"
        ))

        ctrl_grp = make_group("학습 설정")
        cl = QHBoxLayout(ctrl_grp)
        cl.addWidget(QLabel("샘플 수:"))
        self.n_sp = QSpinBox(); self.n_sp.setRange(500, 5000); self.n_sp.setSingleStep(200)
        self.n_sp.setValue(2000); cl.addWidget(self.n_sp)
        cl.addWidget(QLabel("에폭:"))
        self.ep_sp = QSpinBox(); self.ep_sp.setRange(100, 5000); self.ep_sp.setSingleStep(100)
        self.ep_sp.setValue(1000); cl.addWidget(self.ep_sp)
        cl.addWidget(QLabel("LR:"))
        self.lr_sp = QDoubleSpinBox(); self.lr_sp.setRange(1e-4, 0.01)
        self.lr_sp.setDecimals(4); self.lr_sp.setValue(0.001); cl.addWidget(self.lr_sp)
        self.run_btn  = QPushButton("학습 시작")
        self.stop_btn = QPushButton("정지")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(f"background:{CLR_STOP}; color:white; font-weight:bold;")
        self.run_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(lambda: self._worker and self._worker.stop())
        cl.addWidget(self.run_btn); cl.addWidget(self.stop_btn); cl.addStretch()
        root.addWidget(ctrl_grp)

        self.prog = QProgressBar(); self.prog.setValue(0); root.addWidget(self.prog)

        self.fig, self.canvas = make_canvas(13, 4.5)
        root.addWidget(self.canvas)

        # 인터랙티브 예측
        pred_grp = make_group("인터랙티브 예측 (학습 후 활성화)")
        pl = QHBoxLayout(pred_grp)
        pl.addWidget(QLabel("길이 L (m):"))
        self.L_sp = QDoubleSpinBox(); self.L_sp.setRange(0.5, 3.0); self.L_sp.setValue(1.0)
        self.L_sp.setSingleStep(0.1); self.L_sp.valueChanged.connect(self._predict_live)
        pl.addWidget(self.L_sp)
        pl.addWidget(QLabel("각도 θ₀ (deg):"))
        self.th_sp = QDoubleSpinBox(); self.th_sp.setRange(5, 80); self.th_sp.setValue(30)
        self.th_sp.setSingleStep(5); self.th_sp.valueChanged.connect(self._predict_live)
        pl.addWidget(self.th_sp)
        self.pred_lbl = QLabel("NN=?, 물리식=?")
        self.pred_lbl.setFont(QFont("", 11, QFont.Weight.Bold))
        self.pred_lbl.setStyleSheet(f"color:{CLR_SUCCESS};")
        pl.addWidget(self.pred_lbl); pl.addStretch()
        root.addWidget(pred_grp)

        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(80)
        root.addWidget(self.log)
        self._draw_placeholder()

    def _draw_placeholder(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, "학습 시작 버튼을 누르세요",
                ha="center", va="center", fontsize=14, color="gray", transform=ax.transAxes)
        ax.axis("off"); self.canvas.draw()

    def _start(self):
        self._loss_hist.clear(); self.prog.setValue(0)
        self._worker = Lab4Worker(self.n_sp.value(), self.ep_sp.value(), self.lr_sp.value())
        self._worker.epoch_update.connect(self._on_epoch)
        self._worker.finished.connect(self._on_done)
        self._worker.start()
        self.run_btn.setEnabled(False); self.stop_btn.setEnabled(True)

    def _on_epoch(self, epoch, loss, val):
        self._loss_hist.append((loss, val))
        self.prog.setValue(int(epoch / self.ep_sp.value() * 100))
        self.log.setPlainText(f"Epoch {epoch+1} | Loss: {loss:.6f} | Val: {val:.6f}")
        self.fig.clear(); ax = self.fig.add_subplot(111)
        ax.plot([l[0] for l in self._loss_hist], label="Train", color=CLR_PRIMARY, lw=1.5)
        ax.plot([l[1] for l in self._loss_hist], label="Val",   color=CLR_WARN,    lw=1.5, ls="--")
        ax.set_yscale("log"); ax.legend(); ax.grid(True, alpha=0.3)
        ax.set_title("실시간 학습 Loss", fontweight="bold"); self.canvas.draw()

    def _on_done(self, model, X_te, Y_te, hist):
        self.prog.setValue(100)
        self.run_btn.setEnabled(True); self.stop_btn.setEnabled(False)
        self._model = model

        Xm, Xs, Ym, Ys = model._scaler
        X_raw = model._X_te_raw   # (L, theta0) unnormalized
        T_true = model._Y_te_raw  # T true unnormalized

        Y_pred_n = model.predict(X_te, verbose=0)
        T_pred   = (Y_pred_n.flatten() * Ys + Ym)
        mape = float(np.mean(np.abs((T_pred - T_true) / T_true)) * 100)

        self.fig.clear()
        axes = self.fig.subplots(1, 3)

        # 예측 vs 실제
        ax = axes[0]
        ax.scatter(T_true[:300], T_pred[:300], s=8, alpha=0.5, color=CLR_PRIMARY)
        lim = [T_true.min()*0.95, T_true.max()*1.05]
        ax.plot(lim, lim, "r--", lw=1.5, label="Perfect")
        ax.set_xlabel("True T (s)"); ax.set_ylabel("Predicted T (s)")
        ax.set_title(f"주기 예측\nMAPE={mape:.2f}%", fontweight="bold")
        ax.legend(); ax.grid(True, alpha=0.3)

        # 길이별 오차 분포
        ax = axes[1]
        L_arr  = X_raw[:, 0]
        errors = np.abs(T_pred - T_true) / T_true * 100
        sc = ax.scatter(L_arr[:300], errors[:300], c=X_raw[:300, 1],
                        cmap="viridis", s=8, alpha=0.6)
        self.fig.colorbar(sc, ax=ax, label="theta0 (deg)")
        ax.set_xlabel("L (m)"); ax.set_ylabel("MAPE (%)")
        ax.set_title("길이·각도별 오차", fontweight="bold"); ax.grid(True, alpha=0.3)

        # Loss 곡선
        ax = axes[2]
        ax.plot(hist["loss"],     label="Train", color=CLR_PRIMARY, lw=1.5)
        ax.plot(hist["val_loss"], label="Val",   color=CLR_WARN,    lw=1.5, ls="--")
        ax.set_yscale("log"); ax.legend(); ax.grid(True, alpha=0.3)
        ax.set_title("Training Loss", fontweight="bold")

        self.fig.tight_layout(); self.canvas.draw()
        self.log.setPlainText(
            f"학습 완료! | MAPE: {mape:.3f}% | Epochs: {len(hist['loss'])}\n"
            f"아래 슬라이더로 L, theta0 조정 → NN 예측 vs 물리식 비교!"
        )
        self._predict_live()

    def _predict_live(self):
        if self._model is None: return
        g = 9.81
        L  = self.L_sp.value(); th = self.th_sp.value()
        # 물리식
        thr = np.deg2rad(th)
        T_phys = 2 * np.pi * np.sqrt(L / g) * (1 + thr**2/16 + 11*thr**4/3072)
        # NN
        Xm, Xs, Ym, Ys = self._model._scaler
        inp = (np.array([[L, th]]) - Xm) / Xs
        T_nn = float(self._model.predict(inp, verbose=0).flatten()[0]) * Ys + Ym
        err = abs(T_nn - T_phys) / T_phys * 100
        self.pred_lbl.setText(
            f"NN = {T_nn:.4f} s | 물리식 = {T_phys:.4f} s | 오차 = {err:.3f}%"
        )


# ══════════════════════════════════════════════════════════════
# 메인 윈도우
# ══════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "Week 4: 물리 데이터 Neural Network — PySide6 + TensorFlow Interactive Lab"
        )
        self.resize(1300, 880)

        # 탭 구성
        tabs = QTabWidget()
        tabs.setFont(QFont("", 10))
        tabs.addTab(Lab1Tab(), "Lab 1: 1D 함수 근사")
        tabs.addTab(Lab2Tab(), "Lab 2: 포물선 운동")
        tabs.addTab(Lab3Tab(), "Lab 3: 과적합/과소적합")
        tabs.addTab(Lab4Tab(), "Lab 4: 진자 주기 예측")
        self.setCentralWidget(tabs)

        self.statusBar().showMessage(
            "AI와 머신러닝 (PH2002141-033) | Week 4 Homework | PySide6 + TensorFlow | 김충현  "
            f"| TF {tf.__version__}"
        )

    def closeEvent(self, event):
        # 모든 탭의 worker 정리
        for i in range(self.centralWidget().count()):
            tab = self.centralWidget().widget(i)
            if hasattr(tab, "_worker") and tab._worker and tab._worker.isRunning():
                tab._worker.stop()
                tab._worker.wait(2000)
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # 다크 accents 팔레트 약간 커스터마이징
    pal = app.palette()
    pal.setColor(QPalette.ColorRole.Window, QColor(CLR_BG))
    app.setPalette(pal)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
