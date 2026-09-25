"""Small logging / progress helpers."""
from __future__ import annotations

import sys
import time


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


class Ticker:
    def __init__(self, interval: float):
        self.interval, self.last = interval, time.time()

    def due(self) -> bool:
        now = time.time()
        if now - self.last >= self.interval:
            self.last = now
            return True
        return False
