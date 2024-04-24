import pandas as pd
from load_data import load_data_from_csv
from call_api import initiate_chat_system, get_chat_completion

# Initiate the chat system
initiate_chat_system()

# Load the data from CSV
data = load_data_from_csv('airline_test.csv')

# Define a function to find airline names in a tweet
def find_airline_names(tweet):
    print(tweet)
    chat_completion = get_chat_completion(user_content="Please find the airline names in this tweet: "+tweet)
    print(chat_completion)
    return chat_completion

# Apply the function to each tweet in the DataFrame
data['airline_names'] = data['tweet'].apply(find_airline_names)

# Export the updated DataFrame to a new CSV file
data.to_csv('airline_test_zero_shot.csv', index=False)
