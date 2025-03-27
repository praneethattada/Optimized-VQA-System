

import os
import json
import tqdm
import numpy as np
import spacy

# Load SpaCy model
nlp = spacy.load("en_core_web_md")

# Define files (in Colab, we'll create these files directly)
question_file = 'questions.json'
prediction_file = 'predicted_answers.json'


# Function to check similarity
def check_similar_in_meaning(answer1, answer2, similarity_threshold=0.3):
    """
    Checks semantic similarity between two sentences using SpaCy vectors.
    """
    # Create SpaCy Doc objects
    doc1 = nlp(answer1)
    doc2 = nlp(answer2)

    # Calculate similarity using SpaCy's built-in method
    similarity = doc1.similarity(doc2)

    # Return True if similarity exceeds the threshold
    return similarity >= similarity_threshold

# Load data
with open(question_file) as file:
    annotations = json.load(file)

with open(prediction_file) as f:
    data = json.load(f)

# Evaluation logic
correct_count = 0
total_question_count = 0

for datum in tqdm.tqdm(data):
    question_id = datum['question_id']
    prediction = datum['prediction'].strip()
    label = datum['answer'].strip()

    # Use question ID to fetch metadata
    meta = annotations[question_id]
    ground_truth = meta["answer"].strip()

    # Check similarity
    if check_similar_in_meaning(prediction, ground_truth):
        correct_count += 1

    total_question_count += 1

# Print evaluation results
print(f"Total answers: {total_question_count}, matched answers: {correct_count}. "
      f"Accuracy: {100 * correct_count / total_question_count:.2f}%")
