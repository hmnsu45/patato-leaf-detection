"""Potato Disease Classification - Streamlit website.

Place this file in the project root (next to `saved_models/` and `potatoes.h5`)
or inside the `saved_models` folder. It finds the models on its own.

Run:  streamlit run app.py
"""
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

IMAGE_SIZE = 256
# Same order as the training notebook (alphabetical PlantVillage folders)
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

INFO = {
    "Early Blight": (
        "warning",
        "Caused by the fungus *Alternaria solani*. Remove infected leaves, rotate crops "
        "each season, avoid overhead watering and use a suitable fungicide if it spreads.",
    ),
    "Late Blight": (
        "error",
        "Caused by *Phytophthora infestans* and spreads very fast in cool, wet weather. "
        "Remove and destroy infected plants, do not compost them, and treat the rest of the "
        "crop with a recommended fungicide quickly.",
    ),
    "Healthy": (
        "success",
        "The leaf looks healthy. Keep monitoring the plants and maintain good spacing, "
        "watering and soil nutrition.",
    ),
}

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------- models
def find_models():
    """Return {label: Path} for every model found near this file."""
    found = {}
    bases = [HERE, HERE.parent, Path.cwd()]

    def add(path, label):
        found.setdefault(path.resolve(), label)

    for base in bases:
        sm = base / "saved_models"
        if sm.is_dir():
            for d in sm.iterdir():
                if d.is_dir() and (d / "saved_model.pb").exists():
                    add(d, f"saved_models/{d.name}")
        # app.py placed inside the saved_models folder itself
        if base.is_dir():
            for d in base.iterdir():
                if d.is_dir() and (d / "saved_model.pb").exists():
                    add(d, f"saved_models/{d.name}")
            for f in list(base.glob("*.h5")) + list(base.glob("*.keras")):
                add(f, f.name)

    items = sorted(found.items(), key=lambda kv: (kv[1].endswith((".h5", ".keras")), kv[1]))
    return {label: path for path, label in items}


@st.cache_resource(show_spinner="Loading model, please wait...")
def load_model(path_str):
    """Load a model and return (function, needs_rescale).

    function: batch(N,256,256,3) -> class probabilities
    """
    import tensorflow as tf

    path = Path(path_str)

    if path.is_dir():
        # SavedModel folder. Loading it directly avoids Keras version problems
        # (for example the old `random_flip` error on new TensorFlow versions).
        loaded = tf.saved_model.load(str(path))
        fn = loaded.signatures["serving_default"]
        input_name = list(fn.structured_input_signature[1].keys())[0]

        def run(batch):
            out = fn(**{input_name: tf.constant(batch, dtype=tf.float32)})
            return list(out.values())[0].numpy()

        run.keepalive = loaded  # keep the model variables alive
        return run, False  # SavedModels rescale pixels inside the model

    model = tf.keras.models.load_model(str(path), compile=False)

    def run(batch):
        return np.asarray(model(batch, training=False))

    return run, True  # the .h5 model has no rescaling layer, so we divide by 255


# ------------------------------------------------------------ prediction
def preprocess(image, rescale):
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.asarray(image, dtype="float32")
    if rescale:
        arr = arr / 255.0
    return np.expand_dims(arr, 0)


def predict(run, rescale, image):
    probs = np.asarray(run(preprocess(image, rescale)))[0].astype("float64")
    if probs.min() < 0 or abs(probs.sum() - 1) > 1e-3:  # logits -> probabilities
        e = np.exp(probs - probs.max())
        probs = e / e.sum()
    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]), probs


def sample_images():
    for base in (HERE, HERE.parent, Path.cwd()):
        d = base / "test_images_from_internet"
        if d.is_dir():
            files = sorted(p for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
            return {p.stem.replace("_", " ").title(): p for p in files}
    return {}


# -------------------------------------------------------------------- UI
def show_result(run, rescale, image, threshold):
    left, right = st.columns([1, 1.2], gap="large")
    with left:
        st.image(image, caption="Your image", width=360)

    with st.spinner("Analysing leaf..."):
        label, confidence, probs = predict(run, rescale, image)

    with right:
        st.subheader("Result")
        if confidence < threshold:
            st.warning("**Not sure.** This may not be a potato leaf, or the photo is unclear.")
            st.metric("Best guess (low confidence)", f"{label} - {confidence * 100:.2f}%")
            st.caption("Try a clear, close-up photo of a single potato leaf on a plain background.")
        else:
            kind, advice = INFO[label]
            getattr(st, kind)(f"**{label}**")
            st.metric("Confidence", f"{confidence * 100:.2f}%")
            st.markdown(advice)

    st.markdown("**Probability for each class**")
    for name, p in zip(CLASS_NAMES, probs):
        st.progress(float(p), text=f"{name}: {p * 100:.2f}%")


def main():
    st.set_page_config(page_title="Potato Disease Classifier", page_icon="🥔", layout="wide")
    st.title("🥔 Potato Disease Classifier")
    st.write(
        "Upload a photo of a potato leaf and the model will tell you whether it is "
        "**healthy**, has **Early Blight** or has **Late Blight**."
    )

    models = find_models()
    if not models:
        st.error(
            "No model found. Put this file next to the `saved_models` folder "
            "(or `potatoes.h5`) and run it again."
        )
        st.stop()

    with st.sidebar:
        st.header("Settings")
        choice = st.selectbox("Model", list(models.keys()))
        st.caption("saved_models/1, 2, 3 are the versions trained in the notebook. "
                   "potatoes.h5 is the older single-file model.")
        threshold = st.slider(
            "Minimum confidence", 0.50, 0.99, 0.85, 0.01,
            help="Below this, the app says 'not sure' instead of naming a disease. "
                 "Raise it to reject more wrong images.",
        )
        st.divider()
        st.caption("Trained on the PlantVillage dataset, so it works best on a clear photo "
                   "of one potato leaf. It cannot detect other plants or diseases.")

    try:
        run, rescale = load_model(str(models[choice]))
    except Exception as e:  # show a readable message instead of a crash
        st.error("Could not load this model.")
        st.exception(e)
        st.info("Try another model in the sidebar, or install a different TensorFlow version.")
        st.stop()

    tab_up, tab_cam, tab_sample = st.tabs(["Upload image", "Use camera", "Sample images"])

    with tab_up:
        f = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])
        if f is not None:
            show_result(run, rescale, Image.open(f), threshold)

    with tab_cam:
        shot = st.camera_input("Take a photo of a leaf")
        if shot is not None:
            show_result(run, rescale, Image.open(shot), threshold)

    with tab_sample:
        samples = sample_images()
        if not samples:
            st.info("No sample images found (folder `test_images_from_internet`).")
        else:
            name = st.selectbox("Pick a sample", list(samples.keys()))
            show_result(run, rescale, Image.open(samples[name]), threshold)


if __name__ == "__main__":
    main()
