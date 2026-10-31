from pathlib import Path
import json

import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0


class PlantDiseaseModel:

    def __init__(self, model_path: str, classes_path: str):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model_path = Path(model_path)
        self.classes_path = Path(classes_path)

        self.model = None
        self.classes = []
        self.transform = None
        self.loaded = False

    def load(self):
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Model checkpoint not found: {self.model_path}"
            )

        if not self.classes_path.is_file():
            raise FileNotFoundError(
                f"Class mapping not found: {self.classes_path}"
            )

        with open(self.classes_path, "r", encoding="utf-8") as f:
            class_to_idx = json.load(f)

        self.classes = [
            name for name, index in sorted(
                class_to_idx.items(),
                key=lambda item: item[1]
            )
        ]

        if len(self.classes) != 38:
            raise ValueError(
                f"Expected 38 classes, found {len(self.classes)}"
            )

        # Recreate the EfficientNet-B0 architecture.
        self.model = efficientnet_b0(weights=None)

        self.model.classifier[1] = nn.Linear(
            self.model.classifier[1].in_features,
            len(self.classes)
        )

        # Load checkpoint.
        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=True
        )

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict, strict=True)

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.loaded = True

        print(f"Plant disease model loaded on {self.device}")
        print(f"Number of disease classes: {len(self.classes)}")

    def predict(self, image: Image.Image, top_k: int = 5):

        if not self.loaded:
            raise RuntimeError("Model has not been loaded")

        image = image.convert("RGB")

        tensor = self.transform(image).unsqueeze(0)
        tensor = tensor.to(self.device)

        with torch.inference_mode():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]

        k = min(top_k, len(self.classes))

        scores, indices = torch.topk(probabilities, k)

        predictions = []

        for score, index in zip(scores.tolist(), indices.tolist()):
            predictions.append({
                "disease": self.classes[index],
                "confidence": round(score * 100, 2)
            })

        return predictions