# Q&A Chatbot
#from langchain.llms import OpenAI

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

import streamlit as st
import os
import pathlib
import textwrap
from PIL import Image


import google.generativeai as genai


import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("calories-app")

# REST transport: the default gRPC transport was hanging silently with no output
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"), transport="rest")

# gemini-3.8-flash / 3.7-flash were timing out for this key; 3.5-flash responds in seconds
MODEL_NAME = "gemini-3.5-flash"
REQUEST_TIMEOUT = 90  # seconds

## Function to load Gemini model and get responses

def get_gemini_response(input,image,prompt):
    model = genai.GenerativeModel(MODEL_NAME)
    log.info("Calling %s (image %d bytes)...", MODEL_NAME, len(image[0]["data"]))
    start = time.time()
    # Drop empty text parts: an empty "" part (blank Input Prompt box) makes the API hang
    parts = [p for p in [input, image[0], prompt] if not (isinstance(p, str) and not p.strip())]
    response = model.generate_content(
        parts,
        request_options={"timeout": REQUEST_TIMEOUT},
    )
    log.info("Got response in %.1fs", time.time() - start)
    return response.text
    

def input_image_setup(uploaded_file):
    # Check if a file has been uploaded
    if uploaded_file is not None:
        # Read the file into bytes
        bytes_data = uploaded_file.getvalue()

        image_parts = [
            {
                "mime_type": uploaded_file.type,  # Get the mime type of the uploaded file
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")


##initialize our streamlit app

st.set_page_config(page_title="Gemini Image Demo")

st.header("Gemini Application")
input=st.text_input("Input Prompt: ",key="input")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
image=""   
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)


submit=st.button("Tell me about the image")

input_prompt = """
               You are an expert in nutritionist where you need to see the food items from the image
               and calculate the total calories, also provide the details of every food items with calories intake
               is below format

               1. Item 1 - no of calories
               2. Item 2 - no of calories
               ----
               ----
               """

## If ask button is clicked

if submit:
    if uploaded_file is None:
        st.warning("Please upload an image first.")
    else:
        image_data = input_image_setup(uploaded_file)
        try:
            with st.spinner(f"Analyzing image with {MODEL_NAME}..."):
                response=get_gemini_response(input_prompt,image_data,input)
            st.subheader("The Response is")
            st.write(response)
        except Exception as e:
            log.exception("Gemini call failed")
            st.error(f"Request failed: {type(e).__name__}: {e}")