import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image


class VisionForgeModel:

    def __init__(self):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Using device: {self.device}")

        self.model_name = "HuggingFaceTB/SmolVLM-256M-Instruct"

        self.processor = AutoProcessor.from_pretrained(self.model_name)

        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)

    def ask(self, image_path, question):

        image = Image.open(image_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]

        prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=256
        )

        output = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        return output