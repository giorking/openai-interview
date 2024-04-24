import pandas as pd
from load_data import load_data_from_csv
from call_api import initiate_chat_system, get_chat_completion

def evaluate_test(data, ground_truth="airlines", test_results="airline_names"):
    # Ensure the columns exist in the DataFrame
    if ground_truth not in data.columns or test_results not in data.columns:
        raise ValueError("Specified columns do not exist in the DataFrame")

    # Sort the lists in each row of the specified columns and compare
    # Convert the arrays to strings to compare values
    data['result'] = data.apply(
        lambda row: sorted(eval(str(row[ground_truth]))) == sorted(eval(str(row[test_results]))),
        axis=1
    ).astype(int)
    
    return data

def calculate_performance(data):
    # Calculate the total number of entries
    total_entries = len(data)
    
    # Sum the 'result' column
    sum_results = data['result'].sum()
    
    # Calculate the performance ratio
    performance_ratio = sum_results / total_entries if total_entries > 0 else 0
    
    return performance_ratio

def run_test(input_csv, output_csv, user_content_header="Please find the airline names in this tweet: "):
    # Initiate the chat system
    initiate_chat_system()

    # Load the data from CSV
    data = load_data_from_csv(input_csv)

    # Calculate the total number of entries
    total_entries = len(data)
    
    # Define a function to find airline names in a tweet
    def find_airline_names(tweet, index):
        if index % 50 == 0 or index == total_entries - 1:
            print(f"Processing {index + 1}/{total_entries} entries...")
        chat_completion = get_chat_completion(user_content=user_content_header + tweet)
        # airline_names will always be an array of strings
        airline_names = chat_completion.split(',')
        return airline_names

    # Apply the function to each tweet in the DataFrame with the index
    data['airline_names'] = data.apply(lambda row: find_airline_names(row['tweet'], row.name), axis=1)

    # Evaluate the test and export the updated DataFrame to a new CSV file
    evaluated_data = evaluate_test(data)
    evaluated_data.to_csv(output_csv, index=False)
    performance_ratio = calculate_performance(evaluated_data)
    return performance_ratio
