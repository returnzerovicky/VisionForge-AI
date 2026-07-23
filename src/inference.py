from model import VisionForgeModel
import prompts


class VisionForge:

    def __init__(self):
        self.model = VisionForgeModel()

    def caption(self, image):
        return self.model.ask(image, prompts.CAPTION)

    def ocr(self, image):
        return self.model.ask(image, prompts.OCR)

    def document_analysis(self, image):
        return self.model.ask(image, prompts.DOCUMENT)

    def detect_objects(self, image):
        return self.model.ask(image, prompts.OBJECTS)

    def summarize(self, image):
        return self.model.ask(image, prompts.SUMMARY)

    def ask_question(self, image, question):
        query = f"{prompts.QA}\n{question}"
        return self.model.ask(image, query)