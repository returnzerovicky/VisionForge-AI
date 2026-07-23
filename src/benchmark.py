import pandas as pd

from inference import VisionForge
from evaluation import Benchmark
from dataset import ImageDataset


engine = VisionForge()

dataset = ImageDataset("data")

results = []

for i in range(len(dataset)):

    benchmark = Benchmark()

    benchmark.add_model(engine.model.get_model_name())

    benchmark.add_task("Caption")

    benchmark.start()

    caption = engine.caption(dataset.path(i))

    benchmark.stop()

    row = benchmark.summary()

    row["Image"] = dataset.path(i)

    row["Prediction"] = caption

    results.append(row)

df = pd.DataFrame(results)

df.to_csv("reports/results.csv", index=False)

print(df)