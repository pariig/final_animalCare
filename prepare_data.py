import pandas as pd
import json
import os

def clean_and_convert():
    df = pd.read_csv('final_merged_pet_disease_data.csv')

    # Fill missing values for both interventions
    df['Primary Intervention (First Aid)'] = df['Primary Intervention (First Aid)'].fillna("No specific first aid listed. Keep the animal calm and warm.")
    df['Secondary Intervention (Medical Therapy)'] = df['Secondary Intervention (Medical Therapy)'].fillna("Professional veterinary assessment required for medical therapy.")

    data_list = []
    for _, row in df.iterrows():
        data_list.append({
            "symptoms": str(row['symptoms']),
            "disease": str(row['disease_name']),
            "primary": str(row['Primary Intervention (First Aid)']),
            "secondary": str(row['Secondary Intervention (Medical Therapy)'])
        })

    os.makedirs('data', exist_ok=True)
    with open('data/processed_data.json', 'w') as f:
        json.dump(data_list, f, indent=4)
    
    print(f"✅ Success! Processed {len(data_list)} symptoms for advanced mapping.")

if __name__ == "__main__":
    clean_and_convert()
# def clean_and_convert():
#     # Load the CSV
#     df = pd.read_csv('final_merged_pet_disease_data.csv')

#     # Fill missing First Aid with general advice
#     df['Primary Intervention (First Aid)'] = df['Primary Intervention (First Aid)'].fillna(
#         "Maintain hygiene, keep the animal warm/calm, and contact an NGO or vet."
#     )

#     intents = []
#     # We use 'disease_category' as the tag to give the model more samples per tag
#     grouped = df.groupby('disease_category')

#     for category, group in grouped:
#         # Patterns: all symptoms listed under this category
#         patterns = group['symptoms'].dropna().unique().tolist()
        
#         # Responses: all unique first-aid steps for this category
#         responses = group['Primary Intervention (First Aid)'].dropna().unique().tolist()

#         intents.append({
#             "tag": str(category),
#             "patterns": patterns,
#             "responses": responses
#         })

#     # Save to the data folder
#     os.makedirs('data', exist_ok=True)
#     with open('data/intents.json', 'w') as f:
#         json.dump({"intents": intents}, f, indent=4)
    
#     print(f" Success! Processed {len(df)} rows into {len(intents)} categories.")

# if __name__ == "__main__":
#     clean_and_convert()