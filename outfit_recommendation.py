import streamlit as st
import pandas as pd
import cv2
import numpy as np
import requests
import datetime
import os
import time
from sklearn.cluster import KMeans

OUTFIT_CSV = "outfits.csv"
HISTORY_CSV = "outfit_history.csv"
JACKET_TEMP_THRESHOLD = 16
API_KEY = "48b8cf776845b1b3b76e183c60826568"  # Replace with your actual API key

# Load outfit database

df = pd.read_csv(OUTFIT_CSV)

def load_history():
    if not os.path.exists(HISTORY_CSV):
        return []
    with open(HISTORY_CSV) as f:
        return f.readlines()

def log_outfit(outfit_id):
    today = datetime.date.today()
    with open(HISTORY_CSV, "a") as f:
        f.write(f"{today},{outfit_id}\n")

def get_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url).json()
    if "main" in response:
        return response['main']['temp']
    return None

def get_skin_tone(image_bytes):
    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (640, 480))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return "neutral"

    x, y, w, h = faces[0]
    cx, cy = w // 2, h // 2
    swatch = image[y+cy-30:y+cy+30, x+cx-30:x+cx+30]
    data = swatch.reshape((-1, 3))
    kmeans = KMeans(n_clusters=1, n_init=10)
    kmeans.fit(data)
    r, g, b = kmeans.cluster_centers_[0]

    if r > g and r > b:
        return "warm"
    elif b > r and b > g:
        return "cool"
    else:
        return "neutral"

def suggest_outfit(skin_tone, temp, history):
    filtered = df[df['tone'].str.lower() == skin_tone]
    if temp < JACKET_TEMP_THRESHOLD:
        filtered = filtered[filtered['jacket'] == True]
    else:
        filtered = filtered[filtered['jacket'] == False]

    recent_ids = [int(x.split(',')[1]) for x in history
                  if (datetime.date.today() - datetime.date.fromisoformat(x.split(',')[0])).days < 7]
    filtered = filtered[~filtered['outfit_id'].isin(recent_ids)]

    return filtered.sample(1).iloc[0] if not filtered.empty else None

# Streamlit UI

st.set_page_config(page_title="Outfit Recommender")
st.title("Outfit Recommender")

city = st.text_input("Enter your city")

photo_method = st.radio("Choose photo method:", ["Upload photo", "Use webcam"])

if photo_method == "Upload photo":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
else:
    uploaded_image = st.camera_input("Take a photo")

if st.button("Get Outfit Recommendation"):
    if not city or not uploaded_image:
        st.error("Please enter a city and provide a photo.")
    else:
        with st.spinner("Matching tones..."):
            tone = get_skin_tone(uploaded_image)
            time.sleep(1.5)

        with st.spinner("Analysing weather..."):
            temp = get_weather(city)
            time.sleep(1.5)

        if temp is None:
            st.error("Could not fetch weather data. Please check the city name.")
        else:
            history = load_history()
            outfit = suggest_outfit(tone, temp, history)

            if outfit is None:
                st.warning("No suitable outfit found that hasn't been worn in the last week.")
            else:
                log_outfit(int(outfit['outfit_id']))
                st.success("Here's your outfit recommendation:")
                st.markdown(f"**Outfit ID:** {outfit['outfit_id']}")

                st.markdown(f"**Pants:** {outfit['pants']}")
                st.color_picker("Pants Color", outfit['pants_color'], label_visibility="collapsed")

                st.markdown(f"**Shirt:** {outfit['shirt']}")
                st.color_picker("Shirt Color", outfit['shirt_color'], label_visibility="collapsed")

                st.markdown(f"**Jacket:** {'Yes' if outfit['jacket'] else "No need for a jacket, it's quite warm out"}")
