import time


class Benchmark:

    def __init__(self):
        self.results = {}

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.results["Inference Time (sec)"] = round(
            time.time() - self.start_time, 2
        )

    def summary(self):
        return self.results