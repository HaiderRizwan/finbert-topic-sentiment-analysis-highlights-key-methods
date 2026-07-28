import argparse  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

from PIL import Image  # noqa: E402

# Provide resolving for local importing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from torchvision import transforms  # noqa: E402

# Import the model factory from our training script
from scripts.train_models import get_model  # noqa: E402
from src.config import PROJECT_ROOT  # noqa: E402

# The classes inferred by PyTorch ImageFolder alphabetically
CLASSES = ["erotism", "normal", "violent"]


def predict_image(image_path: str, model_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Ensure file exists
    if not os.path.exists(image_path):
        print(f"Error: Could not find image at {image_path}")
        return

    # 2. Setup the model
    model = get_model(model_name, num_classes=len(CLASSES), pretrained=False)

    weights_path = PROJECT_ROOT / "models" / f"{model_name}_best.pth"
    if not weights_path.exists():
        print(
            f"Error: Could not find trained weights at {weights_path}. "
            f"You might need to train the {model_name} model first."
        )
        return

    print(f"Loading trained weights from: {weights_path}")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    # 3. Apply exactly the same preprocessing as the Validation Set
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Failed to open image: {e}")
        return

    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device)  # Create a mini-batch of 1

    # 4. Infer
    with torch.no_grad():
        output = model(input_batch)

    # Calculate probabilities
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Determine top predicted class
    top_prob, top_class_idx = torch.max(probabilities, 0)
    top_class_name = CLASSES[top_class_idx]

    # Map 'normal' back to user terminology 'safe', and 'violent' to 'violence'
    mapping = {"normal": "Safe", "violent": "Violence", "erotism": "Erotism"}

    display_class = mapping.get(top_class_name, top_class_name)

    print("\n" + "=" * 40)
    print(f"🧠 INFERENCE RESULTS FOR: {Path(image_path).name}")
    print("=" * 40)
    print(f"Category:  -> {display_class.upper()} <- ")
    print(f"Confidence:   {top_prob.item() * 100:.2f}%\n")

    print("Detailed Probabilities:")
    for i, cls_name in enumerate(CLASSES):
        print(
            f" - {mapping.get(cls_name, cls_name)}: {probabilities[i].item() * 100:.2f}%"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test a single image against trained models."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Absolute or relative path to the image to test.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="nasnet",
        choices=["nasnet", "mobilenet"],
        help="Which trained model to use.",
    )

    args = parser.parse_args()
    predict_image(args.image, args.model)
