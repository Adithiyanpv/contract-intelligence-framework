import os
import gdown

GDRIVE_FOLDER_ID = "1hanu85tSennwbLadJsgEnIxF0ZrvniHL"

def ensure_models():
    """
    Downloads model artifacts from Google Drive only if not already present.
    Paths are resolved relative to this file's location so it works on
    Streamlit Cloud and locally regardless of working directory.
    """
    base = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base, "resources")

    required_paths = [
        os.path.join(model_dir, "deberta-clause-final"),
        os.path.join(model_dir, "clause_centroids.npy"),
        os.path.join(model_dir, "clause_thresholds.npy"),
        os.path.join(model_dir, "clause_applicability.npy"),
        os.path.join(model_dir, "clause_polarity.npy"),
    ]

    if all(os.path.exists(p) for p in required_paths):
        print("✅ Models already present. Skipping download.")
        return

    print("⬇️ Downloading models from Google Drive...")
    os.makedirs(model_dir, exist_ok=True)

    gdown.download_folder(
        id=GDRIVE_FOLDER_ID,
        output=model_dir,
        quiet=False,
        use_cookies=False
    )

    print("✅ Model download complete.")
