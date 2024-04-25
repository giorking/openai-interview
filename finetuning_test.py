from run_test_finetuning import run_test
import pandas as pd
from call_embedding_api import get_embeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Test CSV file name
test_csv = 'airline_test.csv'

result = run_test('airline_test.csv', 'airline_test_finetuning.csv', model='ft:gpt-3.5-turbo-0125:openai-candidate::9HwRn064')

print(f"The exact match percent for the finetuned model is: {result}")

# Load data
data = pd.read_csv('airline_test_finetuning.csv')

# Initialize a list to store similarity scores
similarity_scores = []

# Process each row in the dataframe
for index, row in data.iterrows():
    if (index + 1) % 1 == 50 or (index + 1) == len(data):
        print(f"Processing row {index + 1}/{len(data)}")
    
    # Strip the [''] around the airline names and handle multiple items
    airlines_cleaned = row['airlines'].strip("[]").replace("'", "").replace(" ", "") # Ground truth data
    airline_names_cleaned = row['airline_names'].strip("[]").replace("'", "").replace(" ", "") # Generated data
    
    # Get embeddings for airline_names and result
    airlines_embedding = get_embeddings(airlines_cleaned)
    airline_names_embedding = get_embeddings(airline_names_cleaned)

    # Convert embeddings to numpy arrays
    airlines_embedding_array = np.array(airlines_embedding).reshape(1, -1)
    airline_names_embedding_array = np.array(airline_names_embedding).reshape(1, -1)

    # Calculate cosine similarity
    similarity_score = cosine_similarity(airlines_embedding_array, airline_names_embedding_array)[0][0]
    
    # Append the similarity score to the list
    similarity_scores.append(similarity_score)

# Add the similarity scores as a new column to the dataframe
data['similarity_score'] = similarity_scores

# Calculate the average of the similarity scores
average_similarity = data['similarity_score'].mean()

# Print the average similarity score
print(f"Average similarity score: {average_similarity}")

# Save the updated dataframe to a new CSV file
data.to_csv('airline_test_finetuning_embeddings_similarity.csv', index=False)


