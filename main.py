from cnnfunctions import (extract_ekg_and_data, make_ekg_image, CNNpredict_from_image, 
                          format_report, get_llm_interpretation)  #import functions from cnnfunctions.py
import sys
import tensorflow as tf
import os


# This prevents TensorFlow from allocating all VRAM at once
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"TensorFlow memory growth enabled for {len(gpus)} GPU(s).")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

if len(sys.argv) != 2:
    message = """ 
    When using Command line call main.py and export.zip (health file you export from apple health)
    Format CLI Usage: python3 main.py <path_to_zip_file>
    example: python3 main.py Health1.zip
    """
    print(message)
    sys.exit(1)



model_path = 'EKG_CNN.keras'   # this variable will be used to upload pretrained CNN below
class_names = [
    'Atrial Flutter', 
    'Atrial Fibrillation',
    'Sinus Arrhythmia',
    'Sinus Bradycardia',
    'Sinus Rhythm',
    'Sinus Tachycardia',
    'Supraventricular Tachycardia'
    ]
# Model prediction Index/Identification
    #AF -> 0
    #AFIB -> 1
    #SA -> 2
    #SB -> 3
    #SR -> 4
    #ST -> 5
    #SVT -> 6
LLM_MODEL = 'ekgllama3'   # This is an ollama llm based on llama3 that I put a system prompt into. Find system prompt in ekgmodelfile

file_path = sys.argv[1]

extracted_data = extract_ekg_and_data(file_path)  # This will pull file, EKG values and meta_data from the apple.zip file
    
if extracted_data:
    ekg_values, apple_labels = extracted_data           # This unpacks the variable extracted_data to provide ekg_values and apple_labels
    image_paths = make_ekg_image(ekg_value_df = ekg_values, meta_data = apple_labels)  # This will plot EKG values and create .png and return the path to the file

    if image_paths:
        cnn_predictions = CNNpredict_from_image(image_paths, model_path, class_names)  # This will apply the created images to the CNN

        if cnn_predictions:
            all_labels = [result['prediction'] for result in cnn_predictions] # This creates a list of the three predicted labels from the 3 10 second EKGs
            prediction = str(all_labels) # This will print if CNN has more than 1 identified prediction from the 3 ekgs
          
            if len(set(all_labels)) == 1:    # Checks to see if all labels are the same by putting them in a set
                prediction = all_labels[0]   # create single prediction variable 
            

            report_to_send = format_report(apple_labels, prediction)  # Generates the initial report to send to LLM 
                                                                      # LLM has system prompt designed for the report input

            llm_response = get_llm_interpretation(report_to_send, model_name=LLM_MODEL) # Calls function for cnn functions.py
                                                                                        # inputs report created above
                                                                                        # calls our custom llm defined in a variable above LLM_MODEL

            disclaimer = """
I am not a LLM for medical diagnosis. I cannot diagnose heart attacks. 
I am created for an EKG project. Please consult a doctor with concerns. 
Nothing I say should be taken as medical advice.
            """

            final_report = f"""
========== REPORT ==========

--- Apple Data ---
------------------

Date : {apple_labels.get('date', 'N/A')}
Reported Symptoms: {apple_labels.get('symptom', 'N/A')}

Apple Rhythm ID: {apple_labels.get('rhythm', 'N/A')}

--- CNN EKG Predictions ---
---------------------------
CNN Rhythm ID: {prediction}

--- EKG LLM Interpretation ---
------------------------------
{llm_response}

{disclaimer}

========== END REPORT ==========
"""
            print(final_report)
            





