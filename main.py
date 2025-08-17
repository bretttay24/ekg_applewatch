from cnnfunctions import extract_ekg_and_data, make_ekg_image, CNNpredict_from_image, open_file  #import functions from cnnfunctions.py
import sys
import tensorflow as tf

if len(sys.argv) != 2:
    message = """ 
    When using Command line call main.py and appleexport.zip (health file you export from apple health)
    Format CLI Usage: python3 main.py <path_to_zip_file>
    example: python3 main.py apple_health_exports/Health1.zip
    """
    print(message)
    sys.exit(1)



model_path = 'EKG_CNN.keras'   # this variable will be used to upload pretrained CNN below
class_names = ['Atrial Flutter', 'Atrial Fibrillation', 'Sinus Arrhythmia', 'Sinus Bradycardia', 'Sinus Rhythm', 'Sinus Tachycardia', 'Supraventricular Tachycardia']
# Model prediction Index/Identification
    #AF -> 0
    #AFIB -> 1
    #SA -> 2
    #SB -> 3
    #SR -> 4
    #ST -> 5
    #SVT -> 6

file_path = sys.argv[1]

extracted_data = extract_ekg_and_data(file_path)  # This will pull file, EKG values and meta_data from the apple.zip file
    
if extracted_data:
    ekg_values, apple_labels = extracted_data           # This unpacks the variable extracted_data to provide ekg_values and apple_labels
    
    image_paths = make_ekg_image(ekg_value_df = ekg_values, meta_data = apple_labels)  # This will plot EKG values and create .png and return the path to the file

    if image_paths:
        cnn_predictions = CNNpredict_from_image(image_paths, model_path, class_names)  # This will apply the created images to the CNN

        if cnn_predictions:
            all_labels = [result['prediction'] for result in cnn_predictions] # This creates a list of the three predicted labels from the 3 10 second EKGs
            print("\n\n========== REPORT ==========")
            print("\n--- Apple Data ---")
            print("------------------")
            print(f"\nDate : {apple_labels['date']}")
            print(f"Apple Rhythm ID: {apple_labels['rhythm']}")
            print(f"Reported Symptoms: {apple_labels['symptom']}")
            print("\n--- CNN EKG Predictions ---")
            print("-----------------------------")
            if len(set(all_labels)) == 1:    # Checks to see if all labels are the same by putting them in a set
                prediction = all_labels[0]   # create single prediction variable 
                print(f"CNN Rhythm ID: {prediction}")

            else:
                for result in cnn_predictions:
                    print(f"\nFile:{result['file']}")
                    print(f"CNN Rhythm Identification: {result['prediction']}")
            print("\n========== END REPORT ==========")






