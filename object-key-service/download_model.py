from huggingface_hub import hf_hub_download
import os
import shutil


REPO_ID = "nocontextdoruk/asl-landmark-mlp"

MODEL_FILES = [
    "mlp_asl.onnx",
    "mlp_classes.json",
]


os.makedirs(
    "models",
    exist_ok=True
)


for filename in MODEL_FILES:

    print(
        f"Download {filename}..."
    )

    downloaded_file = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename
    )

    destination = os.path.join(
        "models",
        filename
    )

    shutil.copyfile(
        downloaded_file,
        destination
    )

    print(
        f"Berhasil: {destination}"
    )


print()
print(
    "Model gesture A-Z dan 0-9 selesai didownload."
)