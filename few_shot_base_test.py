from run_test import run_test
import pandas as pd
import warnings
from zero_shot_test_details import test_details

# Test cases
few_shot_test_cases = {"1-shot Test": 1, "3-shot Test": 3, "5-shot Test": 5, "10-shot Test": 10}

# Test CSV file name
test_csv = 'airline_test.csv'

# Prepare a DataFrame to store results
results_df = pd.DataFrame(columns=['Test Name', 'Performance'])

# Suppress future warnings related to pandas concat behavior
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load previous zero shot results to find the best performing test
previous_results_df = pd.read_csv('zero_shot_results.csv')
# Handle potential errors in performance values that are not numeric
previous_results_df['Performance'] = pd.to_numeric(previous_results_df['Performance'], errors='coerce')

# Get the performance number for "Baseline Test" and add it to the results DataFrame
baseline_test_name = "Baseline Test"
baseline_test_performance = previous_results_df.loc[previous_results_df['Test Name'] == baseline_test_name, 'Performance'].values[0]
baseline_test_result = pd.DataFrame({'Test Name': [baseline_test_name], 'Performance': [baseline_test_performance]})
results_df = pd.concat([results_df, baseline_test_result], ignore_index=True)

# Load the first 10 rows of the test_csv into an array to use for few-shot tests
test_data_array = pd.read_csv(test_csv, nrows=10).values

# Run all tests defined in few_shot_test_cases
for test_name, num_rows in few_shot_test_cases.items():
    print(f"Running {test_name}...")
    # Prepare system_content with sample data
    sample_data = "\n".join([f"User: {row[0]}\nResponse: {', '.join(eval(row[1]))}" for row in test_data_array[:num_rows]])
    system_content = f"You are a helpful assistant that finds airline names in tweets.\n\nBelow are some examples of how to respond to tweets you are given.\n{sample_data}"

    try:
        # Use the baseline test details for all few-shot tests
        best_test_details = test_details[baseline_test_name]
        run_test_kwargs = {
            "input_csv": test_csv,
            "output_csv": "airline_test_few_shot_" + str(num_rows) + ".csv",
            "system_content": system_content
        }
        if 'model' in best_test_details and best_test_details['model']:
            run_test_kwargs['model'] = best_test_details['model']
        if 'temperature' in best_test_details and best_test_details['temperature'] is not None:
            run_test_kwargs['temperature'] = best_test_details['temperature']
        if 'user_content_header' in best_test_details and best_test_details['user_content_header']:
            run_test_kwargs['user_content_header'] = best_test_details['user_content_header']

        result = run_test(**run_test_kwargs)
        print(f"{test_name} performance:", result)
        new_row = pd.DataFrame({'Test Name': [test_name], 'Performance': [result]})
    except Exception as e:
        print(f"Error during {test_name}:", e)
        new_row = pd.DataFrame({'Test Name': [test_name], 'Performance': [str(e)]})
    results_df = pd.concat([results_df, new_row], ignore_index=True)

# Export results to CSV
results_df.to_csv('few_shot_base_results.csv', index=False)

