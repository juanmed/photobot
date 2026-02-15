import sys
import time
import threading
from collections import deque
from dataclasses import dataclass, field

import pyqtgraph as pg
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from gpiozero import Button, RotaryEncoder
from rpi_hardware_pwm import HardwarePWM


@dataclass
class SharedState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    steps: int = 0
    duty_cycle: float = 0.0


class EncoderWorker(threading.Thread):
    def __init__(
        self,
        state: SharedState,
        stop_event: threading.Event,
        *,
        a_pin: int = 17,
        b_pin: int = 27,
        button_pin: int = 22,
        step_period_s: float = 0.05,
        button_period_s: float = 0.1,
    ) -> None:
        super().__init__(daemon=True)
        self.state = state
        self.stop_event = stop_event
        self.a_pin = a_pin
        self.b_pin = b_pin
        self.button_pin = button_pin
        self.step_period_s = step_period_s
        self.button_period_s = button_period_s

    def run(self) -> None:
        encoder = RotaryEncoder(a=self.a_pin, b=self.b_pin, wrap=False, max_steps=0)
        button = Button(self.button_pin)
        next_step_read = time.monotonic()
        next_button_read = time.monotonic()

        try:
            while not self.stop_event.is_set():
                now = time.monotonic()

                if now >= next_step_read:
                    with self.state.lock:
                        self.state.steps = encoder.steps
                    next_step_read += self.step_period_s

                if now >= next_button_read:
                    if button.is_pressed:
                        print("Button pressed!")
                    next_button_read += self.button_period_s

                next_tick = min(next_step_read, next_button_read)
                sleep_for = max(0.0, min(next_tick - time.monotonic(), 0.01))
                self.stop_event.wait(sleep_for)
        finally:
            encoder.close()
            button.close()


class PwmWorker(threading.Thread):
    def __init__(
        self,
        state: SharedState,
        stop_event: threading.Event,
        *,
        pwm_channel: int = 2,
        hz: int = 20_000,
        chip: int = 0,
    ) -> None:
        super().__init__(daemon=True)
        self.state = state
        self.stop_event = stop_event
        self.pwm_channel = pwm_channel
        self.hz = hz
        self.chip = chip

    def run(self) -> None:
        pwm = HardwarePWM(pwm_channel=self.pwm_channel, hz=self.hz, chip=self.chip)
        ramp_values = list(range(0, 20, 5)) + list(range(20, 0, -5))

        try:
            pwm.start(0.0)
            with self.state.lock:
                self.state.duty_cycle = 0.0

            while not self.stop_event.is_set():
                for duty in ramp_values:
                    if self.stop_event.is_set():
                        break
                    pwm.change_duty_cycle(float(duty))
                    with self.state.lock:
                        self.state.duty_cycle = float(duty)
                    print(f"DT: {duty}%")
                    if self.stop_event.wait(1.0):
                        break
        finally:
            pwm.stop()
            print("Stopping pwm...")


class PlotDataWorker(QThread):
    sample_ready = pyqtSignal(float, float, float)

    def __init__(
        self,
        state: SharedState,
        stop_event: threading.Event,
        *,
        sample_period_s: float = 0.05,
    ) -> None:
        super().__init__()
        self.state = state
        self.stop_event = stop_event
        self.sample_period_s = sample_period_s

    def run(self) -> None:
        t0 = time.monotonic()
        while not self.stop_event.is_set():
            with self.state.lock:
                step = float(self.state.steps)
                duty = float(self.state.duty_cycle)
            t_now = time.monotonic() - t0
            self.sample_ready.emit(t_now, step, duty)
            if self.stop_event.wait(self.sample_period_s):
                break


class EncoderPwmWindow(QMainWindow):
    def __init__(self, state: SharedState, stop_event: threading.Event) -> None:
        super().__init__()
        self.state = state
        self.stop_event = stop_event
        self.setWindowTitle("Encoder Steps and PWM Duty Cycle")
        self.resize(1000, 600)

        self.times: deque[float] = deque()
        self.steps: deque[float] = deque()
        self.duties: deque[float] = deque()

        container = QWidget()
        layout = QVBoxLayout(container)
        self.setCentralWidget(container)

        pg.setConfigOptions(antialias=True)
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)

        self.plot_item = self.plot_widget.getPlotItem()
        self.plot_item.setLabel("bottom", "Time (s)")
        self.plot_item.setLabel("left", "Encoder Steps")
        self.plot_item.setYRange(-50_000, 50_000, padding=0.0)
        self.plot_item.showGrid(x=True, y=True, alpha=0.25)

        self.step_curve = self.plot_item.plot(
            pen=pg.mkPen(color=(40, 180, 99), width=2),
            name="Steps",
        )

        self.duty_viewbox = pg.ViewBox()
        self.plot_item.showAxis("right")
        right_axis = self.plot_item.getAxis("right")
        right_axis.setLabel("Duty Cycle (%)")
        right_axis.setPen(pg.mkPen(color=(235, 87, 87)))
        self.plot_item.scene().addItem(self.duty_viewbox)
        right_axis.linkToView(self.duty_viewbox)
        self.duty_viewbox.setXLink(self.plot_item.vb)
        self.duty_viewbox.setYRange(0, 100, padding=0.0)
        self.duty_curve = pg.PlotCurveItem(pen=pg.mkPen(color=(235, 87, 87), width=2))
        self.duty_viewbox.addItem(self.duty_curve)

        self.plot_item.vb.sigResized.connect(self._sync_right_axis)
        self._sync_right_axis()

        self.plot_worker = PlotDataWorker(self.state, self.stop_event, sample_period_s=0.05)
        self.plot_worker.sample_ready.connect(self.on_sample_ready)
        self.plot_worker.start()

    def _sync_right_axis(self) -> None:
        self.duty_viewbox.setGeometry(self.plot_item.vb.sceneBoundingRect())
        self.duty_viewbox.linkedViewChanged(self.plot_item.vb, self.duty_viewbox.XAxis)

    @pyqtSlot(float, float, float)
    def on_sample_ready(self, t_now: float, step: float, duty: float) -> None:
        self.times.append(t_now)
        self.steps.append(step)
        self.duties.append(duty)

        cutoff = t_now - 10.0
        while self.times and self.times[0] < cutoff:
            self.times.popleft()
            self.steps.popleft()
            self.duties.popleft()

        x_data = list(self.times)
        self.step_curve.setData(x_data, list(self.steps))
        self.duty_curve.setData(x_data, list(self.duties))
        self.plot_item.setXRange(max(0.0, t_now - 10.0), max(10.0, t_now), padding=0.0)

    def closeEvent(self, event) -> None:  # noqa: N802
        self.stop_event.set()
        self.plot_worker.wait(1000)
        super().closeEvent(event)


def run_gui() -> int:
    state = SharedState()
    stop_event = threading.Event()
    pwm_worker = PwmWorker(state, stop_event)
    encoder_worker = EncoderWorker(state, stop_event)

    pwm_worker.start()
    encoder_worker.start()

    app = QApplication(sys.argv)
    window = EncoderPwmWindow(state, stop_event)
    window.show()

    try:
        return app.exec_()
    finally:
        stop_event.set()
        pwm_worker.join(timeout=2.0)
        encoder_worker.join(timeout=2.0)


if __name__ == "__main__":
    raise SystemExit(run_gui())
