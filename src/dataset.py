import os
from PIL import Image


class ImageDataset:

    def __init__(self, folder="data"):
        self.folder = folder

        self.images = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

    def __len__(self):
        return len(self.images)

    def get(self, index):
        return Image.open(self.images[index]).convert("RGB")

    def path(self, index):
        return self.images[index]