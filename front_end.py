import streamlit as st
import cv2
import numpy as np
from PIL import Image
from backend import *

detector = Face_Detection()

st.set_page_config(page_title="Face Verification System", layout="wide")#

def convert_uploaded_image(uploaded_file):
    if uploaded_file is None:
        return None

    img = Image.open(uploaded_file)               
    img = img.convert("RGB") 
    img_np = np.array(img)                        
    img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    return img_cv2, img


tab1, tab2 = st.tabs(["Photo Verification", "Live Detection"])

with tab1:
    st.title("Face Photo Verification")

    col1, col2 = st.columns(2)

    with col1:
        ref_upload = st.file_uploader("Upload Reference Image", type=["jpg", "jpeg", "png"])
        ref_cv2, ref_pil = None, None
        if ref_upload:
            ref_cv2, ref_pil = convert_uploaded_image(ref_upload)
            st.image(ref_pil, caption="Reference", width=300)

    with col2:
        act_upload = st.file_uploader("Upload Actual Image", type=["jpg", "jpeg", "png"])
        act_cv2, act_pil = None, None
        if act_upload:
            act_cv2, act_pil = convert_uploaded_image(act_upload)
            st.image(act_pil, caption="Actual", width=300)

    if st.button("Verify Face", use_container_width=True):
        if ref_cv2 is None or act_cv2 is None:
            st.error("Please upload both images.")
        else:
            with st.spinner("Verifying..."):
                result = detector.web_verify(ref_cv2, act_cv2)

            st.subheader("Verification Result")

            if result["status"] == "match":
                st.success("FACE MATCHED ✔")
            else:
                st.error("FACE NOT MATCHED ")

            st.json(result)
with tab2:
    st.title("Live Face Detection")

    start_cam = st.checkbox("Start Camera")

    FRAME_WINDOW = st.image([])

    if start_cam:
        cap = cv2.VideoCapture(0)

        while start_cam:
            success, frame = cap.read()
            if not success:
                st.error("Failed to access webcam.")
                break

           
            result = detector.nose_detection(frame)

            if result.get("detected"):
                cv2.putText(frame, "Face Detected", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Face Not Detected", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame_rgb)

        cap.release()
