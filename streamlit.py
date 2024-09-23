import streamlit as st
import requests
import numpy as np
from PIL import Image, ImageDraw
import io
import base64
import cv2

st.title("HAND-REKOG APP")

input_img = st.camera_input("Take a picture")

if input_img is not None:
    # Convert the image to a base64-encoded string
    img = Image.open(input_img)
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    # Send the base64-encoded string to the FastAPI endpoint
    res = requests.post("http://localhost:8000/inference", json={"image": img_str})

    # Log the status code and response text
    st.write(f"Status code: {res.status_code}")
    st.write(f"Response text: {res.text}")
    
    if res.status_code == 200:
        response_data = res.json()
        bbox = response_data["output_bbox"]

        # Convert the PIL image to a NumPy array
        img_np = np.array(img)
        
        # Draw the bounding box on the image
        for box in bbox:
            startX, startY, endX, endY = int(box["startX"]), int(box["startY"]), int(box["endX"]), int(box["endY"])
            cv2.rectangle(img_np, (startX, startY), (endX, endY), (0, 0, 255), 3)
        
        # Convert the NumPy array back to a PIL image
        img_with_bbox = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))

        # Display the image with the bounding box
        st.image(img_with_bbox, caption="Image with Bounding Box", use_column_width=True)
        st.write("Inference result:", res.json())

    else:
        st.write("Error:", res.text)