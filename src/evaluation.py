import time
import torch


class Benchmark:

    def __init__(self):

        self.results = {}

    def start(self):

        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        self.start_time = time.time()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def stop(self):

        self.results["Inference Time (sec)"] = round(
            time.time() - self.start_time,
            3,
        )

        self.results["Device"] = (
            "CUDA" if torch.cuda.is_available() else "CPU"
        )

        if torch.cuda.is_available():

            self.results["GPU Memory (MB)"] = round(
                torch.cuda.max_memory_allocated() / 1024 / 1024,
                2,
            )

    def add_model(self, name):

        self.results["Model"] = name

    def add_task(self, task):

        self.results["Task"] = task

    def summary(self):

        return self.results