import os  # noqa: E402
import shutil  # noqa: E402

import splitfolders  # noqa: E402

input_dir = r"c:\Users\dotsm\Desktop\FYP\SafeToon\dataset\manual dataset augmented"
output_dir = r"c:\Users\dotsm\Desktop\FYP\SafeToon\dataset_split"

# Remove the empty dataset_split directory if it exists to avoid splitfolders complaining
if os.path.exists(output_dir):
    print(f"Removing existing directory {output_dir}")
    shutil.rmtree(output_dir)

print(f"Splitting dataset from {input_dir}")
splitfolders.ratio(
    input_dir,
    output=output_dir,
    seed=42,
    ratio=(0.7, 0.15, 0.15),
    group_prefix=None,
    move=False,
)
print("Splitting completed successfully.")
