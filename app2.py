"""
Calorie estimator with LangChain + RAG (tuned for speed).

Pipeline
  1. Vision step  - Gemini (via LangChain) looks at the (downscaled) photo and returns the food
                    items + portions as structured output.
  2. Retrieval    - each detected item is looked up in a local TF-IDF retriever built from
                    knowledge/foods.csv. Runs in milliseconds with no API calls or quota.
  3. Answer step  - an LCEL chain (prompt | llm | parser) computes calories grounded in the
                    retrieved nutrition facts and streams the answer as it is generated.

Speed choices
  - thinking_budget=0 on Gemini: ~3x faster responses (measured 9.0s -> 2.7s).
  - Images are resized to max 1024 px JPEG before upload.
  - Local TF-IDF retrieval instead of API embeddings (free tier allows only 100 embed
    requests/minute, and every lookup would be a network round trip).

Run:  streamlit run app2.py
"""

import base64
import io
import logging
import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from pydantic import BaseModel, Field

from langchain_community.retrievers import TFIDFRetriever
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("calories-rag")

BASE_DIR = Path(__file__).parent
KB_CSV = BASE_DIR / "knowledge" / "foods.csv"

CHAT_MODEL = "gemini-3.5-flash"  # 3.7/3.8-flash were timing out for this API key
REQUEST_TIMEOUT = 60  # seconds
TOP_K = 3  # knowledge-base entries retrieved per detected food item
MAX_IMAGE_SIDE = 1024  # px; smaller upload = faster request


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.2,
        thinking_budget=0,  # skip the model's internal reasoning pass: much lower latency
        timeout=REQUEST_TIMEOUT,
        max_retries=1,
    )


# --------------------------------------------------------------------------- #
# Knowledge base / retriever
# --------------------------------------------------------------------------- #

def load_food_documents(csv_path: Path) -> list[Document]:
    """One Document per CSV row.

    page_content is only the food name: that's what TF-IDF matches against, so serving
    sizes / notes don't add noise. The full nutrition text lives in metadata["text"].
    """
    df = pd.read_csv(csv_path).fillna("")
    docs = []
    for _, row in df.iterrows():
        text = (
            f"{row['food']} ({row['category']}). Serving: {row['serving']}. "
            f"Calories: {row['calories']} kcal. Protein: {row['protein_g']} g, "
            f"Carbs: {row['carbs_g']} g, Fat: {row['fat_g']} g."
        )
        if row["notes"]:
            text += f" Note: {row['notes']}"
        docs.append(Document(page_content=row["food"], metadata={"food": row["food"], "text": text}))
    return docs


@st.cache_resource(show_spinner="Loading nutrition knowledge base...")
def get_retriever(csv_mtime: float) -> TFIDFRetriever:
    """Rebuilt automatically whenever foods.csv changes (cache key = file modified time)."""
    docs = load_food_documents(KB_CSV)
    start = time.time()
    retriever = TFIDFRetriever.from_documents(
        docs,
        k=TOP_K,
        # character n-grams: matches partial names and spelling variants ("naan" -> "Butter Naan")
        tfidf_params={"analyzer": "char_wb", "ngram_range": (2, 4), "lowercase": True},
    )
    log.info("Indexed %d food entries in %.2fs", len(docs), time.time() - start)
    return retriever


# --------------------------------------------------------------------------- #
# Step 1: vision -> structured list of foods
# --------------------------------------------------------------------------- #

class FoodItem(BaseModel):
    name: str = Field(description="Common name of the dish or food, e.g. 'Butter Naan', 'Dal Tadka'")
    portion: str = Field(description="Estimated visible portion, e.g. '2 pieces', '1 cup', '150 g'")


class DetectedFoods(BaseModel):
    items: list[FoodItem] = Field(description="Every distinct food or drink visible in the image")


DETECT_PROMPT = (
    "You are a food recognition expert. Identify every distinct food and drink item visible "
    "in this image and estimate the portion of each as precisely as you can (pieces, cups, "
    "grams). Use specific common dish names, including the variant when visible "
    "(e.g. 'plain white rice' not 'rice', 'butter naan' not 'bread'). "
    "If the image contains no food, return an empty list."
)


def prepare_image(raw: bytes) -> bytes:
    """Downscale to MAX_IMAGE_SIDE and re-encode as JPEG to cut upload time."""
    img = Image.open(io.BytesIO(raw))
    img = img.convert("RGB")
    img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=85)
    return out.getvalue()


def detect_foods(jpeg_bytes: bytes, user_hint: str) -> DetectedFoods:
    text = DETECT_PROMPT
    if user_hint.strip():
        text += f"\nExtra context from the user: {user_hint.strip()}"

    b64 = base64.b64encode(jpeg_bytes).decode()
    message = HumanMessage(content=[
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{b64}"},
    ])
    detector = get_llm().with_structured_output(DetectedFoods)
    log.info("Detecting foods with %s (%d KB image)...", CHAT_MODEL, len(jpeg_bytes) // 1024)
    start = time.time()
    result = detector.invoke([message])
    log.info("Detected %d items in %.1fs", len(result.items), time.time() - start)
    return result


# --------------------------------------------------------------------------- #
# Step 2: retrieval
# --------------------------------------------------------------------------- #

def retrieve_context(retriever: TFIDFRetriever, items: list[FoodItem]) -> tuple[str, dict[str, list[str]]]:
    """Look up each detected food. Returns the context string for the LLM and per-item hits for the UI."""
    blocks, hits = [], {}
    for item in items:
        docs = retriever.invoke(item.name)
        hits[item.name] = [d.metadata["text"] for d in docs]
        lines = "\n".join(f"  - {d.metadata['text']}" for d in docs)
        blocks.append(f"Detected: {item.name} (portion: {item.portion})\nKnowledge base matches:\n{lines}")
    return "\n\n".join(blocks), hits


# --------------------------------------------------------------------------- #
# Step 3: grounded answer (LCEL chain)
# --------------------------------------------------------------------------- #

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert nutritionist. Calculate calories ONLY from the nutrition facts in the "
     "knowledge-base context below, scaling each entry's serving to the detected portion. "
     "For each item pick the best-matching knowledge-base entry. If no entry is a reasonable "
     "match, give your own estimate and mark it '(estimate, not in knowledge base)'.\n\n"
     "Respond in this format:\n"
     "1. Item - portion - calories (matched entry: <name>)\n"
     "2. ...\n"
     "**Total: X kcal**\n"
     "Then a short macronutrient summary (protein / carbs / fat) and one or two practical "
     "health notes about the meal. Keep it concise."),
    ("human",
     "Knowledge-base context:\n{context}\n\n"
     "User question (may be empty): {question}"),
])


def build_answer_chain():
    return ANSWER_PROMPT | get_llm() | StrOutputParser()


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #

st.set_page_config(page_title="Calorie Advisor (LangChain + RAG)")
st.header("Calorie Advisor: LangChain + RAG")
st.caption(f"Model: {CHAT_MODEL} | Knowledge base: {KB_CSV.relative_to(BASE_DIR)}")

with st.sidebar:
    st.subheader("Knowledge base")
    try:
        retriever = get_retriever(KB_CSV.stat().st_mtime)
        st.success(f"{len(retriever.docs)} food entries indexed")
    except Exception as e:
        log.exception("Failed to load knowledge base")
        st.error(f"Could not load the knowledge base: {type(e).__name__}: {e}")
        st.stop()
    st.write("Edit `knowledge/foods.csv` to add foods; the index rebuilds automatically.")

    st.subheader("Try a lookup")
    probe = st.text_input("Search the knowledge base", placeholder="e.g. paneer")
    if probe.strip():
        for doc in retriever.invoke(probe):
            st.write(f"- {doc.metadata['text']}")

question = st.text_input("Question (optional)", placeholder="e.g. Is this good for weight loss?")
uploaded_file = st.file_uploader("Choose a food image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    st.image(Image.open(uploaded_file), caption="Uploaded image", use_container_width=True)

if st.button("Calculate calories"):
    if uploaded_file is None:
        st.warning("Please upload an image first.")
        st.stop()

    try:
        t0 = time.time()
        with st.status("Analyzing meal...", expanded=True) as status:
            st.write("Step 1: identifying foods in the image")
            detected = detect_foods(prepare_image(uploaded_file.getvalue()), question)
            if not detected.items:
                status.update(label="No food found", state="error")
                st.warning("No food items were detected in this image.")
                st.stop()
            st.write(", ".join(f"{i.name} ({i.portion})" for i in detected.items))

            st.write("Step 2: retrieving nutrition facts from the knowledge base")
            context, hits = retrieve_context(retriever, detected.items)
            status.update(label="Calculating calories...", state="running", expanded=False)

        st.subheader("Result")
        # Stream tokens so the answer starts appearing immediately
        answer_stream = build_answer_chain().stream({"context": context, "question": question or "(none)"})
        st.write_stream(answer_stream)
        elapsed = time.time() - t0
        log.info("Full pipeline finished in %.1fs", elapsed)
        status.update(label=f"Done in {elapsed:.1f}s", state="complete", expanded=False)

        with st.expander("Retrieved knowledge-base entries (RAG context)"):
            for name, matches in hits.items():
                st.markdown(f"**{name}**")
                for text in matches:
                    st.write(f"- {text}")
    except Exception as e:
        log.exception("Pipeline failed")
        st.error(f"Request failed: {type(e).__name__}: {e}")
