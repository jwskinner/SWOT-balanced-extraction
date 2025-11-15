import time

# --- Simple timing class ---
class Timer:
    def __init__(self):
        self.start = time.time()
        self.last = self.start
        print("Starting balanced extraction")

    def lap(self, msg):
        now = time.time()
        print(f"{msg} in {now - self.last:.2f} s")
        self.last = now

    def total(self):
        print(f"All done in {time.time() - self.start:.2f} s total.")