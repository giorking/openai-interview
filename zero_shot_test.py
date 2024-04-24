from run_test import run_test
import pandas as pd
import warnings
from zero_shot_test_details import test_details

# Test CSV file name
test_csv = 'airline_test_short.csv'

# Prepare a DataFrame to store results
results_df = pd.DataFrame(columns=['Test Name', 'Performance'])

# Suppress future warnings related to pandas concat behavior
warnings.simplefilter(action='ignore', category=FutureWarning)

# Run all tests defined in test_details
for test_name, details in test_details.items():
    print(f"Running {test_name}...")
    try:
        if 'user_content_header' in details:
            result = run_test(test_csv, details['csv_file'], model=details['model'], temperature=details['temperature'], user_content_header=details['user_content_header'])
        else:
            result = run_test(test_csv, details['csv_file'], model=details['model'], temperature=details['temperature'])
        print(f"{test_name} performance:", result)
        new_row = pd.DataFrame({'Test Name': [test_name], 'Performance': [result]})
    except Exception as e:
        print(f"Error during {test_name}:", e)
        new_row = pd.DataFrame({'Test Name': [test_name], 'Performance': [str(e)]})
    results_df = pd.concat([results_df, new_row], ignore_index=True)

# Export results to CSV
results_df.to_csv('zero_shot_results.csv', index=False)

