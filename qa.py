import json
from preprocess import remove_s, remove_newline
from tqdm import tqdm
import google.generativeai as genai
from google.colab import userdata

# API Key Configuration
GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY_1')  # Replace with your API key if not stored in Colab
genai.configure(api_key=GOOGLE_API_KEY)

# Input files
caption_file = 'captions.json' 
question_file = 'questions.json'


def get_answer(prompt):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    print(response.text)
    return response.text

# Load caption data (with numbered keys)
with open(caption_file) as file:
    descriptions = json.load(file)

# Load question data and process
with open(question_file) as f:
    data = json.load(f)
    new_data = []

    for key, datum in tqdm(data.items()):
        img_id = datum["imageId"]
        question = datum["question"]
        answer = datum["answer"]

        # Check for matching description based on img_id
        for key_desc, desc_datum in descriptions.items():
            if img_id == desc_datum["imageId"]:
                text = desc_datum["description"]
                text = remove_s(text)
                text = remove_newline(text)

                # Construct the prompt
                prompt = (f"Answer the question in maximum two words based on the text. "
                          f"Consider the type of question in your answer. For example, if it is a yes/no question, answer should be yes or no. "
                          f"Text: {text} "
                          f"Question: {question}")

                # Call Gemini API
                prediction = get_answer(prompt)

                # Add data to results
                new_datum = {
                    'question_id': key,
                    'img_id': img_id,
                    'question': question,
                    'answer': answer,
                    'prediction': prediction,
                    'text': text
                }

                new_data.append(new_datum)

# Save predictions to JSON file
output_file = f"predicted_answers.json"

# Save predictions to the file
with open(output_file, 'w') as outfile:
    json.dump(new_data, outfile, indent=4, sort_keys=True)

print(f"Predictions saved to {output_file}")
