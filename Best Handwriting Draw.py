import streamlit as st
import sqlite3
import io
import time
import numpy as np
import cv2
import os
import cloudinary
import cloudinary.uploader
import cloudinary.api
from dotenv import load_dotenv
from datetime import datetime, timedelta
from PIL import Image
import easyocr
import os
import shutil
import tempfile
import requests
from concurrent.futures import ThreadPoolExecutor
from typing import List

# Load environment variables from .env file
load_dotenv()

CLOUDINARY_CLOUD_NAME = st.secrets["CLOUDINARY_CLOUD_NAME "]
CLOUDINARY_API_KEY = st.secrets["CLOUDINARY_API_KEY "]
CLOUDINARY_API_SECRET = st.secrets["CLOUDINARY_API_SECRET "]

cloudinary.config(
    cloud_name=CLOUDINARY_CLOUD_NAME,
    api_key=CLOUDINARY_API_KEY,
    api_secret=CLOUDINARY_API_SECRET
)

def upload_to_cloudinary(uploaded_file):
    """Uploads a file to Cloudinary and returns the Cloudinary URL."""
    if uploaded_file is not None:
        # Convert file to bytes for Cloudinary
        file_bytes = uploaded_file.read()

        # Upload to Cloudinary with a unique name
        cloudinary_response = cloudinary.uploader.upload(
            file_bytes,
            folder="handwriting_analyzer",
            public_id=f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{uploaded_file.name}",
            resource_type="image",
            tags=["handwriting_analyzer"]  # Tag for filtering
        )
        return cloudinary_response["secure_url"]  # Return URL
    return None

def init_db():
    conn = sqlite3.connect('DB-participants-imageURLs.db')
    c = conn.cursor()
    
    # Create tables if they don't exist
    c.execute('''CREATE TABLE IF NOT EXISTS participants
                 (id INTEGER PRIMARY KEY, name TEXT)''')
    
    # Store Cloudinary URL instead of local image paths
    c.execute('''CREATE TABLE IF NOT EXISTS portraits
                 (id INTEGER PRIMARY KEY, participant_id INTEGER, image_url TEXT)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS winners
                 (id INTEGER PRIMARY KEY, participant_id INTEGER, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    
    # Check if the column 'last_raffle_time' exists, if not, add it
    try:
        c.execute('PRAGMA table_info(winners)')
        columns = [column[1] for column in c.fetchall()]
        if 'last_raffle_time' not in columns:
            c.execute('''ALTER TABLE winners ADD COLUMN last_raffle_time DATETIME''')
            conn.commit()
    except sqlite3.DatabaseError as e:
        print(f"Error while checking or adding column: {e}")
    
    conn.commit()
    return conn

# Initialize the database
conn = init_db()

# Initialize EasyOCR Reader
reader = easyocr.Reader(
    ['en'],
    model_storage_directory=r"C:\One Drive\OneDrive - Tredence\Work\Temporary works-assigned\Raffle Draw\check-backend\models",
    download_enabled=False
)

def process_image(image_url: str) -> float:
    """Loads an image from a URL and performs OCR to get the confidence score."""
    try:
        # Fetch the image from the Cloudinary URL
        response = requests.get(image_url)
        if response.status_code != 200:
            print(f"Error: Unable to load image from {image_url}")
            return -1  # Return low confidence if the image cannot be fetched

        # Convert image to NumPy array
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if img is None:
            print(f"Error: Unable to decode image from {image_url}")
            return -1  # Return low confidence if the image cannot be decoded

        # Perform OCR
        results = reader.readtext(img, detail=1)
        if results:
            confidences = [res[2] for res in results]  # Extract confidence scores
            return sum(confidences) / len(confidences)  # Return average confidence

        return -1  # No text detected, low confidence

    except Exception as e:
        print(f"Error processing image {image_url}: {e}")
        return -1  # Return low confidence on error

def process_images_in_parallel(image_urls: List[str]) -> List[float]:
    """Processes multiple images concurrently using ThreadPoolExecutor."""
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_image, image_urls))  # Map function to all URLs
    return results  # Return list of confidence scores

# Function to invoke parallel processing
def start_processing(image_urls: List[str]) -> List[float]:
    return process_images_in_parallel(image_urls)
    
def clear_old_images():
    """Deletes images from Cloudinary after a raffle draw."""
    try:
        result = cloudinary.api.resources(
            type="upload",
            tags="handwriting_analyzer"
        )
        images = result.get("resources", [])

        for image in images:
            public_id = image["public_id"]
            cloudinary.uploader.destroy(public_id)  # Delete image

        print("Cloudinary images cleared successfully.")
    except Exception as e:
        print(f"Error while clearing Cloudinary images: {e}")

def raffle_draw():
    """Finds the best handwriting from Cloudinary and selects a winner."""
    # Get current hour in format YYYY-MM-DD-HH
    current_hour = datetime.now().strftime("%Y-%m-%d-%H")

    # Fetch images uploaded in the last hour
    result = cloudinary.api.resources(
    type="upload", 
    prefix="handwriting_analyzer"  # Fetch all images from the handwriting_analyzer folder
    )
    
    images = result.get("resources", [])

    if not images:
        return None  # No images found

    best_participant = None
    best_confidence = -1
    winner_image_url = None

    # Extract image URLs from the list of images
    image_urls = [image["secure_url"] for image in images]

    # Process all images in parallel
    confidence_scores = start_processing(image_urls)  # Calls the parallelized function

    # Find the best participant
    best_confidence = -1
    best_participant = None
    winner_image_url = None

    for image, confidence in zip(images, confidence_scores):  # Iterate over images & their confidence scores
        if confidence > best_confidence and confidence <=95 :
            best_confidence = confidence
            best_participant = image["public_id"]
            winner_image_url = image["secure_url"]

    # If a winner is found, update the database
    if best_participant:
        c = conn.cursor()

        # Check if participant exists in winners table
        c.execute("SELECT id FROM winners WHERE participant_id = ?", (best_participant,))
        existing_winner = c.fetchone()

        if existing_winner:
            winner_id = existing_winner[0]
            c.execute("UPDATE winners SET last_raffle_time = ? WHERE id = ?", (datetime.now(), winner_id))
        else:
            c.execute("INSERT INTO winners (participant_id, last_raffle_time) VALUES (?, ?)", (best_participant, datetime.now()))

        conn.commit()

        return best_participant, winner_image_url

    return None  # No valid images found

# Function to check if an hour has passed since the last raffle draw
def can_perform_raffle(conn):
    c = conn.cursor()
    c.execute("SELECT last_raffle_time FROM winners ORDER BY last_raffle_time DESC LIMIT 1")
    last_raffle_time = c.fetchone()
    
    if last_raffle_time and last_raffle_time[0]:  # Check if last_raffle_time is not None
        last_raffle_time = datetime.strptime(last_raffle_time[0], "%Y-%m-%d %H:%M:%S.%f")
        time_since_last_raffle = datetime.now() - last_raffle_time
        return time_since_last_raffle >= timedelta(hours=1)
    return True # If no previous raffle, don't perform automatically

# Streamlit app
st.title("🤖AI-assisted raffle draw for best handwriting") # AI-assisted raffle draw for best handwriting

# Sidebar for QR Code
with st.sidebar:
    st.markdown(
        "<p style='text-align: center; font-size: 14px; margin-bottom: 5px;'>"
        "Scan the QR to upload your handwriting image</p>",
        unsafe_allow_html=True
    )
    st.image("QR-Handwriting-analyzer-app.png")
    st.markdown(
        "<p style='text-align: center; font-size: 14px; margin-top: 5px;'>"
        "Write participant name, and upload an image.</p>",
        unsafe_allow_html=True
    )

# Main app logic
st.sidebar.header("Settings")
st.sidebar.write("Configure the raffle draw here.")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

participant_name = st.sidebar.text_input("Add a participant:")
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    cloudinary_url = upload_to_cloudinary(uploaded_file)
    
    if cloudinary_url:
        # st.success("Image uploaded successfully!")
        st.image(cloudinary_url, caption="Selected Image", use_container_width=True)
    else:
        st.error("Image upload failed. Please try again.")

if uploaded_file:
    st.sidebar.info("Image selected. Click 'Add Participant and Upload Image' to proceed.")

# Process only when the button is clicked
if st.sidebar.button("Add Participant and Upload Image"):
    if participant_name and uploaded_file:
        try:
            # Read file bytes inside the button click block
            uploaded_file.seek(0)

            # Upload to Cloudinary and get the URL
            cloudinary_url = upload_to_cloudinary(uploaded_file)

            # Insert or update database entry
            c = conn.cursor()
            c.execute("SELECT id FROM participants WHERE LOWER(name) = LOWER(?)", (participant_name,))
            participant = c.fetchone()

            if participant:
                participant_id = participant[0]
                c.execute("UPDATE portraits SET image_url = ? WHERE participant_id = ?", (cloudinary_url, participant_id))
            else:
                c.execute("INSERT INTO participants (name) VALUES (?)", (participant_name,))
                participant_id = c.lastrowid
                c.execute("INSERT INTO portraits (participant_id, image_url) VALUES (?, ?)", (participant_id, cloudinary_url))

            conn.commit()
            st.sidebar.success(f"Added/Updated participant: {participant_name}")

        except Exception as e:
            st.sidebar.error(f"Error uploading image: {e}")
    else:
        st.sidebar.warning("Please enter a participant name and upload an image.")

# Check if the first raffle has occurred
c = conn.cursor()
c.execute("SELECT COUNT(*) FROM winners")
raffle_count = c.fetchone()[0]

if raffle_count == 0:  # If no raffles have occurred yet
    if st.button("Start First Raffle"):
        winner_result = raffle_draw()
        if winner_result:
            winner_id, winner_image_url = winner_result
            st.success(f"🎉 The winner is: {winner_id}!")
            st.image(winner_image_url, caption="🎭 Best Handwriting Image", use_container_width=True)
            clear_old_images()
        else:
            st.warning("No images found for this hour.")
else:
    # Timer for next raffle draw
    def show_timer():
        """Displays the countdown timer for the next raffle draw."""
        c = conn.cursor()
        c.execute("SELECT last_raffle_time FROM winners ORDER BY last_raffle_time DESC LIMIT 1")
        last_raffle_time = c.fetchone()

        if last_raffle_time and last_raffle_time[0]:
            try:
                # Parse the last raffle time correctly
                last_raffle_time = datetime.strptime(last_raffle_time[0], "%Y-%m-%d %H:%M:%S.%f")

                # Calculate time remaining
                time_elapsed = datetime.now() - last_raffle_time
                time_remaining = timedelta(hours=1) - time_elapsed

                if time_remaining > timedelta(0):  # If there's still time left
                    return str(time_remaining).split('.')[0]  # Display as HH:MM:SS
                else:
                    return "Next raffle draw in 1 hour"
            except ValueError:
                return "Next raffle draw in 1 hour (Time format error)"
        return "Next raffle draw in 1 hour"  # Default message if no raffle has occurred

    # Display countdown timer
    timer_text = show_timer()

    # Check if an hour has passed since the last raffle draw
    if can_perform_raffle(conn):
        if st.button("Start Raffle"):
            winner_result = raffle_draw()
            if winner_result:
                winner_id, winner_image_url = winner_result
                st.success(f"🎉 The winner is: {winner_id}!")
                st.image(winner_image_url, caption="🎭 Best Handwriting Image", use_container_width=True)
                clear_old_images()
            else:
                st.warning("No images found for this hour.")
    else:
        st.write(f"⏰ Time remaining for next raffle draw: **{timer_text}**")
        st.write("Next raffle draw will occur in 1 hour.")

# Close the database connection when the app stops
conn.close()
