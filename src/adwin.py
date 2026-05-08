import math
from collections import deque

class ADWIN:
    def __init__(self, delta: float = 0.002):
        self.delta = delta
        self.drift_detected = False
        self._window: deque = deque()
        self._window_sum: float = 0.0
        self._window_size: int = 0

    def update(self, value: float) -> "ADWIN":
        self.drift_detected = False
        self._window.append(value)
        self._window_sum += value
        self._window_size += 1
        self._detect_change()
        return self

    def _detect_change(self):
        if self._window_size < 2:
            return
        found = True
        while found:
            found = False
            n = self._window_size
            total_sum = self._window_sum
            right_sum = 0.0
            right_n = 0
            window_list = list(self._window)
            for i in range(n - 1, 0, -1):
                right_sum += window_list[i]
                right_n += 1
                left_n = n - right_n
                left_sum = total_sum - right_sum
                if left_n < 1:
                    continue
                left_mean = left_sum / left_n
                right_mean = right_sum / right_n
                epsilon_cut = self._compute_epsilon(n, left_n, right_n)
                if abs(left_mean - right_mean) >= epsilon_cut:
                    for _ in range(left_n):
                        removed = self._window.popleft()
                        self._window_sum -= removed
                    self._window_size -= left_n
                    self.drift_detected = True
                    found = True
                    break

    def _compute_epsilon(self, n, n0, n1):
        if n0 == 0 or n1 == 0:
            return float("inf")
        m = 1.0 / (1.0 / n0 + 1.0 / n1)
        argument = 4.0 * n / self.delta
        if argument <= 0:
            return float("inf")
        return math.sqrt((1.0 / (2.0 * m)) * math.log(argument))

    @property
    def width(self):
        return self._window_size

    @property
    def mean(self):
        return self._window_sum / self._window_size if self._window_size else 0.0
