import pandas as pd
import streamlit as st

from src.predict import predict_text



st.set_page_config(page_title="Document Classifier", page_icon="📄")

SAMPLES = {
    "🚀 Space": "NASA launched a new satellite to study the orbit of Mars and its atmosphere.",
    "🏒 Hockey": "The team scored in overtime and won the playoff game against their rivals.",
    "💻 Graphics": "Which GPU supports ray tracing and has enough VRAM for 3D rendering?",
    "🚗 Cars": "My car engine makes a strange noise when I accelerate on the highway.",
    "🏥 Medical": "The doctor recommended a new treatment for the infection and chronic pain.",
}


def set_text(value):
    st.session_state["text"] = value


st.title("📄 AI Document Classification")
st.caption("TF-IDF + Machine Learning | trained on 20 Newsgroups categories")

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top predictions to show", 1, 10, 5)
    st.header("Try a sample")
    for name, txt in SAMPLES.items():
        st.button(name, on_click=set_text, args=(txt,), use_container_width=True)

text = st.text_area("Paste your document / text here", key="text", height=200)
uploaded = st.file_uploader("Or upload a .txt file", type=["txt"])

if uploaded is not None:
    text = uploaded.read().decode("utf-8", errors="ignore")
    st.info(f"Using uploaded file: {uploaded.name}")

if st.button("Classify", type="primary"):
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        try:
            with st.spinner("Predicting..."):
                result = predict_text(text, top_k=top_k)
        except FileNotFoundError:
            st.error("Model not found. Run python -m src.train first.")
            st.stop()

        st.success(f"Predicted category: *{result['label']}*")
        st.metric("Confidence", f"{result['confidence']:.1%}")

        if result["confidence"] < 0.40:
            st.warning("Low confidence. Text chhota ho sakta hai ya in 20 categories se bahar ka.")

        df = pd.DataFrame(result["top_k"]).rename(columns={"prob": "probability"})
        st.bar_chart(df.set_index("label"))
        st.dataframe(df, hide_index=True, use_container_width=True)

