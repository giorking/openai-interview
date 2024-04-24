import pandas as pd
from load_data import load_data_from_csv
from call_api import initiate_chat_system, get_chat_completion

def evaluate_test(data, ground_truth="airlines", test_results="airline_names"):
    # Ensure the columns exist in the DataFrame
    if ground_truth not in data.columns or test_results not in data.columns:
        raise ValueError("Specified columns do not exist in the DataFrame")

    # Calculate the number of matches
    matches = sum(data[ground_truth].astype(str) == data[test_results].astype(str))
    
    # Calculate the total number of entries
    total_entries = len(data)
    
    # Calculate the match ratio
    match_ratio = matches / total_entries if total_entries > 0 else 0
    
    return match_ratio

def run_test(input_csv, output_csv, user_content_header="Please find the airline names in this tweet: "):
    # Initiate the chat system
    initiate_chat_system()

    # Load the data from CSV
    data = load_data_from_csv(input_csv)

    # Calculate the total number of entries
    total_entries = len(data)
    # Define a function to find airline names in a tweet
    def find_airline_names(tweet):
        chat_completion = get_chat_completion(user_content=user_content_header + tweet)
        return chat_completion

    # Apply the function to each tweet in the DataFrame with the index
    data['airline_names'] = data['tweet'].apply(find_airline_names)

    # Export the updated DataFrame to a new CSV file
    data.to_csv(output_csv, index=False)

    # Evaluate the test
    match_ratio = evaluate_test(data)
    return match_ratio
