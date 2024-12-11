import streamlit as st
import numpy as np
import cv2
import base64
import util  # Importing your provided utility file

# Load saved model artifacts
util.load_saved_artifacts()

# Streamlit app
st.title("Image Classifier")

st.header("Upload an Image for Classification")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Read the uploaded file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to base64 for classification
    a, buffer = cv2.imencode('.jpg', image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    print(cv2.imencode('.jpg', image))

    # Classify the image
    results = util.classify_image(image_base64_data=image_base64)

    # Prepare data for table
    if results:
        st.subheader("Classification Results")

        for idx, result in enumerate(results):
            st.write(f"Result :")
            # Prepare a list of dictionaries for the table
            table_data = [
                {"Class Name": class_name, "Probability (%)": round(probability, 2)}
                for class_name, probability in zip(result['class_dictionary'].keys(), result['class_probability'])
            ]

            # Display the table
            st.table(table_data)
    else:
        st.error("No valid classification results were returned.")
