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
import io
import cv2
from streamlit_webrtc import webrtc_streamer
import av

class_labels = [
 'aerosol_cans',
 'aluminum_soda_cans',
 'cardboard_packaging',
 'clothing',
 'disposable_plastic_cutlery',
 'eggshells',
 'food_waste',
 'glass_beverage_bottles',
 'glass_cosmetic_containers',
 'glass_food_jars',
 'magazines',
 'metal_food_cans',
 'newspaper',
 'paper',
 'paper_cups',
 'plastic_containers',
 'plastic_cup_lids',
 'plastic_detergent_bottles',
 'plastic_shopping_bags',
 'plastic_soda_bottles',
 'plastic_straws',
 'plastic_trash_bags',
 'plastic_water_bottles',
 'shoes',
 'styrofoam_cups',
 'styrofoam_food_containers'
]

# Create a mapping from class index to label, replacing underscores with spaces
class_mapping = {i: label.replace('_', ' ') for i, label in enumerate(class_labels)}

# set some pre-defined configurations for the page, such as the page title, logo-icon, page loading state (whether the page is loaded automatically or you need to perform some action for loading)
st.set_page_config(
    page_title="Waste No More",
    page_icon = ":R3:",
    initial_sidebar_state = 'auto',
    layout = 'centered'
)

# hide deprication warnings which directly don't affect the working of the application
import warnings
warnings.filterwarnings("ignore")


# Inject custom CSS to hide Streamlit style elements and center content
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.centered {
    display: block;
    margin-left: auto;
    margin-right: auto;
    width: 50%; /* Adjust width as necessary */
}
.success-message {
    text-align: center;
    font-size: 18px; /* Optional: Adjust font size */
}
</style>
"""


# hide the CSS code from the screen as they are embedded in markdown text. Also, allow streamlit to unsafely process as HTML
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# Streamlit app layout
#st.title("Image Upload and Preprocessing")

st.header("""Hey there!  Yes you! Have you got some waste to let me try?  Let's play a game where you give me a waste product and I identify it
         """)


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

# Define the image preprocessing function for EfficientNetB0
def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (224, 224))  # Resize to match model input shape
    return np.expand_dims(resized_frame, axis=0)  # Add batch dimension

def detect_objects():
    model = load_model_function(pklFilePath)
    
    # Open the webcam
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW) 

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame for prediction
        processed_frame = preprocess_frame(frame)

        # Make predictions using the model
        predictions = model.predict(processed_frame)
        predicted_class_index = np.argmax(predictions, axis=1)[0]

        # Convert to original label using the mapping
        original_label = class_mapping.get(predicted_class_index, "Unknown Class")

        cv2.putText(frame, f'Predicted: {original_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Display the resulting frame
        cv2.imshow('Real-Time Object Detection', frame)

        # Break the loop if 'q' is pressed
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break
    # Release the capture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


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


 # Option to choose streaming method
data_input_option = st.radio("Choose Image source:", ("Image files", "Zip folder", "livestream through OpenCV Window"))

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
                with st.spinner("Thinking hard..."):
                    predictions = model.predict(processed_images)
                    st.empty()
                    # Clear original images display after prediction button is clicked
                    #st.empty()  # Clear previous output
                    # Display predictions beneath each image
                    for i, (image, prediction) in enumerate(zip(original_images, predictions)):
                        predicted_class_index = np.argmax(prediction)  # Get index of highest probability
                        
                        # Convert to original label using the mapping
                        original_label = class_mapping.get(predicted_class_index, "Unknown Class")
                        
                        # Resize image for display (smaller size)
                        small_image = image.resize((150, 150))  # Resize to 150x150 for display purposes
                        
                        # Display resized image and prediction below it
                        st.image(small_image, use_column_width=False)

                        st.success(f'Hope it is not wait too long! I categorise this picture under: {original_label}')
                        
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

                    #reset any other states if needed
                    #if 'zip_uploader_key' in st.session_state:
                    #    st.session_state.zip_uploader_key += 1 

# Handle ZIP folder upload
elif data_input_option == "Zip folder":

    # Optionally, allow users to upload a ZIP folder (as before)
    zip_file = st.file_uploader("Or upload a ZIP folder of images", 
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

                # Preprocess the extracted images using the defined function
                if st.button("Predict from ZIP"):
                    zip_processed_images = preprocess_images(extracted_images)
    
                    with st.spinner("Making predictions..."):
                        zip_predictions = model.predict(zip_processed_images)
                        st.empty()
                        for i, (zip_image, zip_prediction) in enumerate(zip(zip_processed_images, zip_predictions)):

                            zip_predicted_class_index = np.argmax(zip_prediction)  # Get index of highest probability
                            
                            zip_original_label = class_mapping.get(zip_predicted_class_index, "Unknown Class")

                            # Resize image for display (smaller size)
                            zip_small_image = Image.fromarray((zip_image).astype(np.uint8))  # Convert back to PIL Image

                            zip_small_image = zip_small_image.resize((150, 150))  # Resize to 150x150 for display purposes                    
                            # Display resized image and prediction below it
                            st.image(zip_small_image, use_column_width=False)

                            st.success(f'Hope it is not wait too long! I categorise this picture under: {zip_original_label}')

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
        
        except Exception as e:
            st.error(f"Error processing ZIP file: {str(e)}")

elif data_input_option == "Livestream through OpenCV Window": 
    if st.button("Start OpenCV Streaming"):
        detect_objects() 
