# importing the libraries and dependencies needed for creating the UI and supporting the deep learning models used in the project
import streamlit as st  
import tensorflow as tf
import random
from PIL import Image, ImageOps
import numpy as np
import os
import pickle
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import zipfile
import cv2
import base64
import warnings


        
with open('class_labels.txt', 'r') as labels:
    # Create a dictionary mapping class to risk level
    class_mapping = {}
    
    for line in labels:
        # Split the line into class and risk level
        class_label, risk_level = line.strip().split(':', 1)
        class_mapping[class_label.strip()] = risk_level.strip()


# set some pre-defined configurations for the page, such as the page title, logo-icon, page loading state (whether the page is loaded automatically or you need to perform some action for loading)
st.set_page_config(
    page_title="Waste No More",
    page_icon="♻️",
    initial_sidebar_state = 'auto',
    layout = 'wide'
)

# hide deprication warnings which directly don't affect the working of the application

warnings.filterwarnings("ignore")


# Inject custom CSS to hide Streamlit style elements and center content
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
stApp {
    background-color: transparent; /* Ensure background color is transparent */
}

h1, h2, h3, h4, h5, h6 {
    font-size: 40px; /* Increase header font size */
    line-height: 1.5; /* Increase line height for headers */
    margin-top: 30px; /* Adjust this value as needed to shift downwards */
    margin-bottom: 20px; /* Add margin at the bottom */
}

p {
    font-size: 20px; /* Increase paragraph font size */
    line-height: 1.5; /* Increase line height for paragraphs */
}

}
.success-message {
    text-align: left;
    font-size: 18px; /* Optional: Adjust font size */

/* Change color of st.success box */
.stSuccess {
    background-color: #14092f !important; /* blue background */
    color: white !important; /* White text */
    border-color: #28a745 !important; /* Green border */
}
</style>
"""

# Function to convert an image file to base64
def get_base64(image_file_path):
    with open(image_file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Function to set background image from base64
def set_background_image_from_file(image_file_path):
    bin_str = get_base64(image_file_path)
    background_image = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        height: 100vh; /* Cover full viewport height */
        width: 100vw; /* Cover full viewport width */
        overflow: hidden; /* Prevent scrollbars */
        position: fixed; /* Fix position for parallax effect */
    }}

    </style>
    """
    st.markdown(background_image, unsafe_allow_html=True)

# Set your local image path
set_background_image_from_file(os.path.join(os.getcwd(),'BG1.jpg'))

# hide the CSS code from the screen as they are embedded in markdown text. Also, allow streamlit to unsafely process as HTML
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

#Function to load and return the model

def load_model_function(model_pkl_file):
    with open(model_pkl_file, 'rb') as file:  
        model = pickle.load(file)
    return model


#Getting current directory regardless of where user places it
cwd = os.getcwd()
#pklFile = 'b3_1.pkl'
pklFile = 'tfm1.pkl'
pklFilePath = os.path.join(cwd,pklFile)

# Instantiate Object to preprocess image for EfficientNet
dataGen = ImageDataGenerator(
    preprocessing_function = preprocess_input  
)

#Storing the model in a variable 
model = load_model_function(pklFilePath)

# Function to preprocess images using ImageDataGenerator
def preprocess_images(images):
    processed_images = []
    for image in images:
        # Convert PIL image to numpy array and expand dimensions for model input
        
        # for EfficientNetB3
        image = image.resize((224,224))
        # Convert image to RGB if not already in that format
        if image.mode != 'RGB':
            image = image.convert('RGB')

        image_array = np.array(image)
        image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
        
        # Use the datagen to preprocess the image
        processed_image = dataGen.standardize(image_array)  # Standardize the image
        #processed_images.append(processed_image)
        ##Uncomment to Check shape of processed_image and append it to the list
        #st.write(f"Processed image shape: {processed_image.shape}")  # Display shape in Streamlit
        processed_images.append(processed_image)
    
    ##Uncomment to Check shapes of all processed images before stacking 
    #if processed_images:
    #    shapes = [img.shape for img in processed_images]
    #    st.write(f"Shapes of all processed images before stacking: {shapes}")  # Display all shapes
    
    return np.vstack(processed_images)  # Stack all processed images into a single array


# Use session state to manage uploaded files and uploader key
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'image_uploader_key' not in st.session_state:
    st.session_state.image_uploader_key = 0
if 'zip_uploader_key' not in st.session_state:
    st.session_state.zip_uploader_key = 0

probability_threshold = 0.6  # Adjust this value as needed

# Create columns for layout
col1, col2 = st.columns([7, 3])  # Adjust ratios as necessary

with col1: 

    st.markdown('<div class="container">', unsafe_allow_html=True)  # Start of container

    st.header("""AI Waste Sorting""")
             
    st.header("""Because Even Trash Deserves a Second Chance""")

    st.write("Classify your waste objects with this app.")

    # Option to choose streaming method
    ## uncoment the below if you plan to use device's camera
    #data_input_option = st.radio("Choose Image source:", ("Image files", "Zip folder", "Livestream through your device"))  
    data_input_option = st.radio("Choose Image source:", ("Image files", "Zip folder"))

    if data_input_option == "Image files":

        # File uploader for single or multiple images
        uploaded_files = st.file_uploader("Upload Images", 
                                        type=["jpg", "jpeg", "png"], accept_multiple_files=True,
                                        key=f'image_uploader_{st.session_state.image_uploader_key}')


        if uploaded_files:
            valid_files = [file for file in uploaded_files if file.type in ["image/jpeg", "image/png"]]
            
            if len(valid_files) != len(uploaded_files):
                st.error("Please upload only image files (JPG, JPEG, PNG).")
            else:
                st.session_state.uploaded_files = valid_files

            if st.session_state.uploaded_files:
                original_images = []
                try:
                    for uploaded_file in st.session_state.uploaded_files:
                        image = Image.open(uploaded_file)  # Open each uploaded image
                        original_images.append(image)  # Append to the list
                except Exception as e:
                    st.error(f"Error loading image: {str(e)}")
                
        
                # Preprocess the images using the defined function
                if st.button("Predict"):
                    processed_images = preprocess_images(original_images)
                    with st.spinner("Let's see what you have got..."):
                        predictions = model.predict(processed_images)
                        st.empty()

                        
                        # Display predictions beneath each image
                        for i, (image, prediction) in enumerate(zip(original_images, predictions)):
                            predicted_class_index = np.argmax(prediction)  # Get index of highest probability
                            predicted_probability = prediction[predicted_class_index]  # Get the highest probability
                            
                            # Convert to original label and risk level using the mapping
                            if predicted_class_index < len(class_mapping):
                                original_label = list(class_mapping.keys())[predicted_class_index]
                                risk_level = class_mapping[original_label]  # Get the corresponding risk level
                            else:
                                original_label = "Unknown Class"
                                risk_level = "Unknown Risk Level"

                            # Resize image for display (smaller size)
                            small_image = image.resize((150, 150))  # Resize to 150x150 for display purposes
                            
                            # Display resized image
                            st.image(small_image, use_column_width=False)

                            # Check if the predicted probability meets the threshold
                            if predicted_probability >= probability_threshold:
                                st.success(f'I believe this is a {original_label}. ({risk_level})')
                            else:
                                st.warning(f"I think I am wrong, but my best guess is: {original_label} ({risk_level}).")
                            
                        # Increment key to reset uploader
                        st.session_state.image_uploader_key += 1
            
                    # Clear button to reset uploaded files
                    if st.button("Clear Uploaded Files"):
                        # Clear the list of uploaded files
                        st.session_state.uploaded_files.clear()  

                        # Increment key to reset uploader
                        st.session_state.image_uploader_key += 1
                        
                        # Show success message after clearing
                        st.success("Uploaded files cleared!")  


    # Handle ZIP folder upload
    elif data_input_option == "Zip folder":

        # Optionally, allow users to upload a ZIP folder (as before)
        zip_file = st.file_uploader("Upload a ZIP folder of images", 
                                    type="zip",
                                    key=f'zip_uploader_{st.session_state.zip_uploader_key}')
           
        if zip_file:
            try:
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall("temp_folder")  # Extract to a temporary folder
                    
                    extracted_images = []

                    invalid_zip_files = []

                    for filename in os.listdir("temp_folder"):
                        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                            try:
                                zip_image_path = os.path.join("temp_folder", filename)
                                zip_image = Image.open(zip_image_path)
                                extracted_images.append(zip_image)

                            except Exception as e:
                                st.error(f"Error loading image {filename}: {str(e)}")
                        else:
                            invalid_zip_files.append(filename)  # Store invalid filenames

                    # Check if there were any invalid files in the ZIP
                    if invalid_zip_files:
                        st.error(f"The following files are not valid images and will be ignored: {', '.join(invalid_zip_files)}")
                        st.warning("Prediction will not proceed due to the presence of non-image files.")
                        st.session_state.zip_uploader_key += 1 
                    else:    
                            # Preprocess the extracted images using the defined function
                            if st.button("Predict from ZIP"):
                                zip_processed_images = preprocess_images(extracted_images)
                
                                with st.spinner("Gosh, wonder how many images have you in that zip folder..."):
                                    zip_predictions = model.predict(zip_processed_images)
                                    st.empty()
                                    for i, (zip_image, zip_prediction) in enumerate(zip(zip_processed_images, zip_predictions)):
                                        zip_predicted_class_index = np.argmax(zip_prediction)  # Get index of highest probability
                                        zip_predicted_probability = zip_prediction[zip_predicted_class_index]  # Get the highest probability
                                        
                                        # Convert to original label and risk level using the mapping
                                        if zip_predicted_class_index < len(class_mapping):
                                            zip_original_label = list(class_mapping.keys())[zip_predicted_class_index]
                                            risk_level = class_mapping[zip_original_label]  # Get the corresponding risk level
                                        else:
                                            zip_original_label = "Unknown Class"
                                            risk_level = "Unknown Risk Level"
        
                                        zip_small_image = Image.fromarray((zip_image).astype(np.uint8))  # Convert back to PIL Image
        
                                        zip_small_image = zip_small_image.resize((150, 150))  # Resize to 150x150 for display purposes                    
                                        # Display resized image and prediction below it
                                        st.image(zip_small_image, use_column_width=False)
        
                                        # Check if the predicted probability meets the threshold
                                        if zip_predicted_probability >= probability_threshold:
                                            st.success(f'I believe this is a {zip_original_label}. ({risk_level}).')
                                        else:
                                            st.warning(f"I think I am wrong, but my best guess is: {zip_original_label} ({risk_level}).")
        
                                        # Reset uploader by changing its key
                                        st.session_state.zip_uploader_key += 1  # Increment key to reset uploader


                # Clear button to reset uploaded files
                if st.button("Clear Uploaded Files"):
                    # Clear the list of uploaded files
                    st.session_state.uploaded_files.clear()  

                    # Increment key to reset uploader
                    st.session_state.zip_uploader_key += 1 
                    
                    # Show success message after clearing
                    st.success("Uploaded files cleared!")  
    
                    # Clean up extracted files after processing if needed
                    for filename in os.listdir("temp_folder"):
                        os.remove(os.path.join("temp_folder", filename))

                    st.session_state.zip_uploader_key += 1 
            
            except Exception as e:
                st.error(f"Error processing ZIP file: {str(e)}")

with col2:
    # Empty column for spacing (optional)
    st.empty()  # For adding any other content here if needed
