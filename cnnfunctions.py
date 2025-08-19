
# Import Statements
import zipfile
import pathlib
import csv
import xml.etree.ElementTree as ET
from typing import Tuple, List
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import ollama



def extract_ekg_and_data(file_path):
    """
    The goal of this function is to take a file_path string argumentand extract an EKG file.
    From that EKG file create a ekg dataframe and pull relevant information from ekg dataframe.

    - Returns: EKG_df (dataframe of graphable values), meta_data(dictionary) 
    - Process:
    1) Unzip the file to output director
    2) Find most recent .csv in electrocardiogram folder
    3) Change most recent EKG.CSV to EKG dataframe 
    4) Extract EKG values from EKG_df to be graphed 
    5) Create meta_data dictionary from relevant EKG_df values in head of ekg_df
    6) Return EKG values to be graphed in dataframe format, return meta_data dictionary
    """
    zip_path = pathlib.Path(file_path)
    extract_dir = zip_path.parent

    # --- 1) Unzip the data to output directory---
    print("Unzipping health data...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    base_dir = extract_dir /'apple_health_export'
    ekg_csv_dir = base_dir /'electrocardiograms'
   


    # --- Handle errors ---
    if not ekg_csv_dir.is_dir():
        print("Error: 'electrocardiograms' directory not found.")
        return None

    # --- 2) Find the most recent EKG .csv file ---
    csv_files = list(ekg_csv_dir.glob('*.csv'))
    if not csv_files:
        print("Error: No EKG .csv files found.")
        return None
        
    csv_files.sort() # Sorts the list alphabetically by filename
    latest_csv_path = csv_files[-1] # Get the very last item from the sorted list

    #print(f"Found latest EKG file: {latest_csv_path.name}")
    #print(f"Will proceed to create a dataframe of EKG data from {latest_csv_path.name}")

    # --- 3) Change EKG.CSV to dataframe ---
    ekg_df = pd.read_csv(latest_csv_path)

    # --- 4) Extract EKG values to be graphed ---
    ekg_value_df = ekg_df.iloc[9:, 0] 

    # --- 5) Create meta_data dictionary --- 
    date = ekg_df.iloc[1,1]
    apple_classification = ekg_df.iloc[2,1]
    associated_symptoms = str(ekg_df.iloc[3,1])

    meta_data = {
        "date" : str(date),
        "rhythm" : str(apple_classification),
        "symptom" : str(associated_symptoms)
    }
    # --- 6) Return EKG values to be graphed in dataframe format, return meta_data dictionary ---

    return ekg_value_df, meta_data


def make_ekg_image(ekg_value_df, meta_data):
    """
    This function takes a dataframe of ekg values and meta_data,
    graphs the values, and creates 3 separate 10-second .png files.
    It returns the file names in list format.
    """
    meta_data_dic = meta_data
    ekg_df = ekg_value_df.astype(float)

    output_dir = 'apple_health_export/watch_ekgs'
    os.makedirs(output_dir, exist_ok=True)
    
    saved_image_paths = [] # This list will have .png file paths appended in the loop below

    # --- Constants ---
    SAMPLES_PER_SECOND = 512
    length_ekg_split = 10  # 10 seconds per image
    samples_per_strip = SAMPLES_PER_SECOND * length_ekg_split # 5120 samples per ekg strip
    total_samples = len(ekg_df)
    max_time = 10.4
    min_time = -0.4
    max_microvolts = 800
    min_microvolts = -600

    # Calculate how many 10-second strips we can make (should be 3)
    num_strips = total_samples // samples_per_strip

    # --- Loop to create 3 separate images ---
    for i in range(num_strips):
        strip_num = i + 1  # This will be 1, 2, 3 for the filenames

        # --- 1. Slice the data for the current 10-second strip ---
        start_index = i * samples_per_strip
        end_index = (i + 1) * samples_per_strip
        
        # Select the 5120 samples for this strip
        strip_data = ekg_df.iloc[start_index:end_index]

        # --- 2. Create the filename inside the loop ---
        file_name = f"ekg_watch_{meta_data_dic['date']}_rhythm_{meta_data_dic['rhythm']}_{strip_num}.png"
        full_path = os.path.join(output_dir, file_name)

        # --- 3. Graphing Code for the strip ---
        # Create a new 0-10 second time axis for each plot
        num_samples = len(strip_data)
        time_axis = np.arange(num_samples) / SAMPLES_PER_SECOND

        # Create the plot figure
        fig, ax = plt.subplots(figsize=(14, 4))

        # Plot strip data
        ax.plot(time_axis, strip_data, zorder=1)

        # Set titles, labels, and limits for a 10-second strip
        ax.set_title(f'EKG Data (Lead 1) -{strip_num}')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Voltage (microvolts)') # Apple Watch data is in microvolts
        plt.xlim([min_time, max_time])
        plt.ylim([min_microvolts, max_microvolts])
        plt.xticks(np.arange(0, 11, 1))
        plt.grid(False)

        # 4. Save the figure and close it
        plt.savefig(full_path)
        plt.close(fig) # Crucial to free up memory in a loop
        
        #print(f"Saved {file_name}")
        saved_image_paths.append(full_path)

    return saved_image_paths

def CNNpredict_from_image(ekg_saved_image_paths, model_path, class_names):
    """
    This loads a pretrained Keras CNN model and predicts a class for 
    each EKG image in the ekg_image_path_list. 

    Arguments: 
        - ekg_image_path_list (list): A list of file paths for the .png images of EKGs from apple watch
        - model_path (str): The file path to the saved .keras model
        - class_names (list): This is a list of strings that map the identified class label (0-7) to heart rhythm (SR, Afib, AF, etc)

    Returns:
        - A list of dictionaries where each dictionary contains the file and its prediction
    """
    ekg_image_path = ekg_saved_image_paths
    
    # --- 1) Load the CNN model ---
    # Handle errors loading model
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return []
    
    # --- 2) Define constants for image preprocessing to match images model was trained ---
    IMG_HEIGHT = 128
    IMG_WIDTH = 512

    predictions = []       # This list will be returned at the end of the function

    # --- 3) loop through images in image_list ---
    for ekg_image_path in ekg_image_path:
        # --- Preprocess ---
        ekg = tf.io.read_file(ekg_image_path)
        ekg = tf.io.decode_png(ekg, channels=3)
        ekg = tf.image.resize(ekg,[IMG_HEIGHT, IMG_WIDTH])   # This resizes the image to match the sizes of images CNN was trained on
        ekg = ekg / 255.0                                    # This normalizes pixel values from 0-1
        ekg = tf.expand_dims(ekg, axis=0)                    # adds a batch size to dimensions. (1, 128, 512, 3) = (batch_size, height, width, channels)

        prediction_scores = model.predict(ekg)               # These are scores for each classification 
        predicted_index = np.argmax(prediction_scores)       # Find the highest score and returns the index of classifcation
        predicted_label = class_names[predicted_index]       # Maps the predicted index to heart rythm in class_names list

        result = {                                           # This creates a dictionary that will be stored in the list predictions that will be returned
            "file": os.path.basename(ekg_image_path),
            "prediction": predicted_label
        }

        predictions.append(result)

    return predictions

def format_report(apple_data, cnn_prediction_label):
    """
    Formats the Apple and CNN data into a single string report.

    Args:
        apple_data (dict): The dictionary of metadata from the Apple EKG.
        cnn_prediction_label (str): The final, confident prediction from the CNN.

    Returns:
        A single, multi-line string containing the formatted report.
    """
    # Using .get() is safer than ['key'] as it won't crash if a key is missing
    date = apple_data.get('date', 'N/A')
    apple_rhythm = apple_data.get('rhythm', 'N/A')
    symptoms = apple_data.get('symptom', 'N/A')

    report_string = f"""
========== REPORT ==========

--- Apple Data ---
------------------

Date : {date}
Apple Rhythm ID: {apple_rhythm}
Reported Symptoms: {symptoms}

--- CNN EKG Predictions ---
-----------------------------
CNN Rhythm ID: {cnn_prediction_label}

========== END REPORT ==========
"""
    return report_string



def get_llm_interpretation(report_string, model_name='ekgllm'):
    """
    Sends a report string to a local Ollama model and gets an interpretation.
    Assumes the model has a built-in system prompt.
    """
    
    
    try:
        # The system prompt is no longer needed here; it's in the Modelfile
        response = ollama.chat(
            model=model_name,
            messages=[
                {'role': 'user', 'content': report_string},
            ]
        )
        return response['message']['content']
        
    except Exception as e:
        return f"Error communicating with Ollama: {e}"




 
        
        

  







    