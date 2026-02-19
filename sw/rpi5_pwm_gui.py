# Configure 4 Hardware PWMs following:
# https://gist.github.com/Gadgetoid/b92ad3db06ff8c264eef2abf0e09d569?permalink_comment_id=5045536


import threading
import time
import tkinter as tk
from collections import deque
from dataclasses import dataclass, field

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


class EncoderPwmTkGui:
    def __init__(
        self,
        state: SharedState,
        stop_event: threading.Event,
        *,
        sample_period_ms: int = 50,
    ) -> None:
        self.state = state
        self.stop_event = stop_event
        self.sample_period_ms = sample_period_ms
        self.window_s = 10.0

        self.times: deque[float] = deque()
        self.steps: deque[float] = deque()
        self.duties: deque[float] = deque()
        self.t0 = time.monotonic()

        self.root = tk.Tk()
        self.root.title("Encoder Steps and PWM Duty Cycle")
        self.root.geometry("1000x600")
        self.root.configure(bg="#111111")

        self.canvas = tk.Canvas(self.root, bg="#1b1b1b", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<Configure>", self._on_resize)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_resize(self, _event: tk.Event) -> None:
        self._draw()

    def _on_close(self) -> None:
        self.stop_event.set()
        self.root.destroy()

    @staticmethod
    def _map_x(t_now: float, x0: float, width: float, window_s: float) -> float:
        return x0 + (t_now / window_s) * width

    @staticmethod
    def _map_y(value: float, y_min: float, y_max: float, y0: float, height: float) -> float:
        if y_max == y_min:
            return y0 + height / 2.0
        ratio = (value - y_min) / (y_max - y_min)
        return y0 + (1.0 - ratio) * height

    def _trim_to_window(self, t_now: float) -> None:
        cutoff = t_now - self.window_s
        while self.times and self.times[0] < cutoff:
            self.times.popleft()
            self.steps.popleft()
            self.duties.popleft()

    def _append_sample(self) -> None:
        with self.state.lock:
            step = float(self.state.steps)
            duty = float(self.state.duty_cycle)

        t_now = time.monotonic() - self.t0
        self.times.append(t_now)
        self.steps.append(step)
        self.duties.append(duty)
        self._trim_to_window(t_now)

    def _draw_axes(self, left: int, top: int, right: int, bottom: int) -> None:
        self.canvas.create_rectangle(left, top, right, bottom, outline="#3a3a3a", width=1)

        self.canvas.create_text(left, top - 12, text="Encoder Steps", fill="#45d483", anchor="w")
        self.canvas.create_text(right, top - 12, text="Duty Cycle (%)", fill="#ef6b6b", anchor="e")
        self.canvas.create_text((left + right) / 2, bottom + 18, text="Time (s)", fill="#c8c8c8")

        self.canvas.create_text(left - 8, top, text="50k", fill="#c8c8c8", anchor="e")
        self.canvas.create_text(left - 8, (top + bottom) / 2, text="0", fill="#c8c8c8", anchor="e")
        self.canvas.create_text(left - 8, bottom, text="-50k", fill="#c8c8c8", anchor="e")

        self.canvas.create_text(right + 8, top, text="100", fill="#c8c8c8", anchor="w")
        self.canvas.create_text(right + 8, (top + bottom) / 2, text="50", fill="#c8c8c8", anchor="w")
        self.canvas.create_text(right + 8, bottom, text="0", fill="#c8c8c8", anchor="w")

        for i in range(11):
            x = left + (right - left) * (i / 10.0)
            self.canvas.create_line(x, top, x, bottom, fill="#2a2a2a")
            label = f"{-10 + i:d}"
            self.canvas.create_text(x, bottom + 4, text=label, fill="#999999", anchor="n")

        for i in range(1, 5):
            y = top + (bottom - top) * (i / 5.0)
            self.canvas.create_line(left, y, right, y, fill="#262626")

    def _draw_series(self, left: int, top: int, right: int, bottom: int) -> None:
        if len(self.times) < 2:
            return

        t_end = self.times[-1]
        t_start = max(0.0, t_end - self.window_s)
        plot_width = right - left
        plot_height = bottom - top

        step_points: list[float] = []
        duty_points: list[float] = []

        for t, step, duty in zip(self.times, self.steps, self.duties):
            t_local = t - t_start
            if t_local < 0.0:
                continue

            x = self._map_x(t_local, left, plot_width, self.window_s)
            y_step = self._map_y(step, -50_000.0, 50_000.0, top, plot_height)
            y_duty = self._map_y(duty, 0.0, 100.0, top, plot_height)

            step_points.extend((x, y_step))
            duty_points.extend((x, y_duty))

        if len(step_points) >= 4:
            self.canvas.create_line(*step_points, fill="#45d483", width=2, smooth=False)
        if len(duty_points) >= 4:
            self.canvas.create_line(*duty_points, fill="#ef6b6b", width=2, smooth=False)

    def _draw_latest_values(self, left: int, top: int) -> None:
        if not self.times:
            return
        step = self.steps[-1]
        duty = self.duties[-1]
        text = f"step={step:.0f}   duty={duty:.1f}%"
        self.canvas.create_text(left, top - 28, text=text, fill="#e6e6e6", anchor="w")

    def _draw(self) -> None:
        self.canvas.delete("all")
        width = max(400, self.canvas.winfo_width())
        height = max(300, self.canvas.winfo_height())

        left = 70
        right = width - 70
        top = 50
        bottom = height - 55

        self._draw_axes(left, top, right, bottom)
        self._draw_series(left, top, right, bottom)
        self._draw_latest_values(left, top)

    def _tick(self) -> None:
        if self.stop_event.is_set():
            return
        self._append_sample()
        self._draw()
        self.root.after(self.sample_period_ms, self._tick)

    def run(self) -> int:
        self.root.after(self.sample_period_ms, self._tick)
        self.root.mainloop()
        return 0


def run_gui() -> int:
    state = SharedState()
    stop_event = threading.Event()

    pwm_worker = PwmWorker(state, stop_event)
    encoder_worker = EncoderWorker(state, stop_event)

    pwm_worker.start()
    encoder_worker.start()

    gui = EncoderPwmTkGui(state, stop_event, sample_period_ms=50)

    try:
        return gui.run()
    finally:
        stop_event.set()
        pwm_worker.join(timeout=2.0)
        encoder_worker.join(timeout=2.0)


if __name__ == "__main__":
    raise SystemExit(run_gui())
