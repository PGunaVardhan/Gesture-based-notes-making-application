# streamlit_app.py
import streamlit as st
import subprocess
import os
import cv2
from text_extractor import extract_text_from_image  # Import the function

# Initialize Streamlit session state variables
if 'note_maker_process' not in st.session_state:
    st.session_state['note_maker_process'] = None
if 'extract_frame' not in st.session_state:
    st.session_state['extract_frame'] = None

# Title for Streamlit interface
st.title("Hand Gesture-Based Note Maker")

# Function to start the note-making application
def start_note_maker():
    if st.session_state['note_maker_process'] is None:
        st.session_state['note_maker_process'] = subprocess.Popen(["python", "app.py"])

# Function to stop the note-making application
def stop_note_maker():
    if st.session_state['note_maker_process']:
        st.session_state['note_maker_process'].terminate()
        st.session_state['note_maker_process'] = None

# Sidebar buttons
if st.sidebar.button("Start window"):
    start_note_maker()

if st.sidebar.button("Close window"):
    stop_note_maker()

if st.sidebar.button("Extract text"):
    if st.session_state['note_maker_process']:
        frame_path = "current_frame.jpg"
        if os.path.exists(frame_path):
            # Display the captured frame
            image = cv2.imread(frame_path)
            if image is not None:
                st.image(image, caption="Captured Frame", use_container_width=True)
                
                # Extract text using LLAVA
                extracted_text = extract_text_from_image(image)
                
                # Display extracted text
                st.text_area("Extracted Text:", value=extracted_text)
            else:
                st.error("Failed to load the captured frame. Please try again.")
        else:
            st.warning("No frame available for extraction.")
    else:
        st.warning("Please start the application window first.")
