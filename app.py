import streamlit as st
from pred import prediction
from utils import image_paths
import tempfile
import os
from PIL import Image


def main():
    st.markdown("## Furniture Prediction Through Images")
    tab1, tab2 = st.tabs(['Choose From Sample Images', "Upload an Image"])
    # Input fields
    with tab1:
        st.markdown("### Enter your inputs:")
        sample_choice = st.pills('Which Image would you like to use ?',
                                ("Sample 1","Sample 2","Sample 3", "Sample 4", "Sample 5"),
                                selection_mode='single')
        image_choice = {"Sample 1":1,"Sample 2":2,
                        "Sample 3":3, "Sample 4":4,
                        "Sample 5":5}

        st.image([image_paths[1],image_paths[2],
                image_paths[3],image_paths[4],image_paths[5]],
                ["Sample 1","Sample 2","Sample 3",
                "Sample 4", "Sample 5"],
                use_container_width=False,width=140)
        model_choice = st.pills('Which Model will you like to use for prediction ? ',
                            ('Densenet201','Resnet50','VGG16'),
                            selection_mode='single')
        models = {'Densenet201':'densenet','Resnet50':'resnet','VGG16':'vgg'}
        # Button to predict
        if st.button("Predict", key=1):
            predict = prediction(ca=image_choice[sample_choice], model_name_for_prediction=models[model_choice])
            st.success(f"The predicted class is: {predict}")

    with tab2:
        st.markdown("### Upload an Image of Furniture:")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Store the uploaded image temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name
            
            # Display the uploaded image
            st.image(temp_file_path, caption="Uploaded Image", use_container_width=False,width=140)
            
            # Perform prediction
            if st.button("Predict",key=2):
                predict = prediction(model_name_for_prediction='densenet',img_path = temp_file_path)
                st.success(f"The predicted class is: {predict}")
            
            # Clean up the temporary file after use
            if st.button("Delete Temporary File"):
                try:
                    os.remove(temp_file_path)
                    st.info("Temporary file deleted.")
                except Exception as e:
                    st.error(f"Error deleting file: {e}")





if __name__ == "__main__":
    main()