import glob  # noqa: E402
import os  # noqa: E402

from PIL import Image  # noqa: E402
from torchvision import transforms  # noqa: E402
from tqdm import tqdm  # noqa: E402


def augment_dataset(input_dir, output_dir, augmentations_per_image=3):
    """
    Augments images in the input directory and saves them to the output directory.
    Assumes a structure like:
    input_dir/
      class_1/
        img1.jpg
        ...
      class_2/
      class_3/
    """
    # Define the transforms to apply
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            # You can add more transforms here if needed
        ]
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    classes = [
        d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))
    ]

    print(f"Found classes: {classes}")

    for class_name in classes:
        class_input_dir = os.path.join(input_dir, class_name)
        class_output_dir = os.path.join(output_dir, class_name)

        if not os.path.exists(class_output_dir):
            os.makedirs(class_output_dir)

        # Supported image formats
        valid_extensions = (
            "*.jpg",
            "*.jpeg",
            "*.png",
            "*.webp",
            "*.bmp",
            "*.tif",
            "*.tiff",
        )
        image_paths = []
        for ext in valid_extensions:
            image_paths.extend(glob.glob(os.path.join(class_input_dir, ext)))
            image_paths.extend(
                glob.glob(os.path.join(class_input_dir, ext.upper()))
            )  # Handle uppercase extensions

        print(f"Processing class '{class_name}': found {len(image_paths)} images.")

        for img_path in tqdm(image_paths, desc=f"Augmenting {class_name}"):
            try:
                # Open the original image
                img = Image.open(img_path).convert("RGB")
                base_name = os.path.basename(img_path)
                name, ext = os.path.splitext(base_name)

                # Save the original image to the output directory as well
                original_out_path = os.path.join(class_output_dir, f"{name}_orig{ext}")
                img.save(original_out_path)

                # Generate and save augmented images
                for i in range(augmentations_per_image):
                    aug_img = transform(img)
                    aug_out_path = os.path.join(
                        class_output_dir, f"{name}_aug_{i}{ext}"
                    )
                    aug_img.save(aug_out_path)

            except Exception as e:
                print(f"Error processing {img_path}: {e}")


if __name__ == "__main__":
    # Define your paths
    # Assuming this script is run from the project root or scripts directory
    # The dataset is located at c:\Users\dotsm\Desktop\FYP\SafeToon\dataset\manual dataset

    # Use absolute or relative paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_dataset_dir = os.path.join(base_dir, "dataset", "manual dataset")
    output_dataset_dir = os.path.join(base_dir, "dataset", "manual dataset augmented")

    print(f"Input directory: {input_dataset_dir}")
    print(f"Output directory: {output_dataset_dir}")

    # You can customize the number of augmented versions generated for each image
    augment_dataset(input_dataset_dir, output_dataset_dir, augmentations_per_image=3)

    print("Data augmentation complete!")
