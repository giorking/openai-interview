from run_test import run_test
import pandas as pd
import warnings

# Test CSV file name
test_csv = 'airline_test.csv'

# Prepare a DataFrame to store results
results_df = pd.DataFrame(columns=['Test Name', 'Performance'])

# Suppress future warnings related to pandas concat behavior
warnings.simplefilter(action='ignore', category=FutureWarning)

# Baseline test
# Model: gpt-3.5-turbo
# Temperature: 0.0 (using log probability to hit thresholds)
# Prompt: Please find the airline names in this tweet: 
print("Running baseline test...")
try:
    result_01 = run_test(test_csv, 'airline_test_zero_shot.csv')
    print("Baseline test performance:", result_01)
    new_row = pd.DataFrame({'Test Name': ['Baseline Test'], 'Performance': [result_01]})
except Exception as e:
    print("Error during baseline test:", e)
    new_row = pd.DataFrame({'Test Name': ['Baseline Test'], 'Performance': [str(e)]})
results_df = pd.concat([results_df, new_row], ignore_index=True)

# GPT-4 test
# Model: gpt-4
# Temperature: 0.0
# Prompt: Please find the airline names in this tweet: 
print("Running GPT-4 test...")
try:
    result_02 = run_test(test_csv, 'airline_test_zero_shot_gpt4.csv', model='gpt-4')
    print("GPT-4 test performance:", result_02)
    new_row = pd.DataFrame({'Test Name': ['GPT-4 Test'], 'Performance': [result_02]})
except Exception as e:
    print("Error during GPT-4 test:", e)
    new_row = pd.DataFrame({'Test Name': ['GPT-4 Test'], 'Performance': [str(e)]})
results_df = pd.concat([results_df, new_row], ignore_index=True)

# High temperature test
# Model: gpt-3.5-turbo
# Temperature: 1.0
# Prompt: Please find the airline names in this tweet: 
print("Running high temperature test...")
try:
    result_04 = run_test(test_csv, 'airline_test_zero_shot_high_temperature.csv', temperature=1.0)
    print("High temperature test performance:", result_04)
    new_row = pd.DataFrame({'Test Name': ['High Temperature Test'], 'Performance': [result_04]})
except Exception as e:
    print("Error during high temperature test:", e)
    new_row = pd.DataFrame({'Test Name': ['High Temperature Test'], 'Performance': [str(e)]})
results_df = pd.concat([results_df, new_row], ignore_index=True)

# Additional guidance 1 test
# Model: gpt-3.5-turbo
# Temperature: 0.0
# Prompt: Please find the airline names in this tweet. List the full name of the airline. Don't use abbreviations. Separate airline names with commas. Tweet:
print("Running additional guidance 1 test...")
try:
    result_05 = run_test(test_csv, 'airline_test_zero_shot_guidance_1.csv', user_content_header="Please find the airline names in this tweet. List the full name of the airline. Don't use abbreviations. Separate airline names with commas. Tweet:")
    print("Additional guidance 1 test performance:", result_05)
    new_row = pd.DataFrame({'Test Name': ['Additional Guidance 1 Test'], 'Performance': [result_05]})
except Exception as e:
    print("Error during additional guidance 1 test:", e)
    new_row = pd.DataFrame({'Test Name': ['Additional Guidance 1 Test'], 'Performance': [str(e)]})
results_df = pd.concat([results_df, new_row], ignore_index=True)

# Additional guidance 2 test
# Model: gpt-3.5-turbo
# Temperature: 0.0
# Prompt: Please find the airline names in this tweet. List the full name of the airline. Don't use abbreviations. Separate airline names with commas. Include airlines ONLY. For airlines like US Airways, include the space. Tweet:
print("Running additional guidance 2 test...")
try:
    result_06 = run_test(test_csv, 'airline_test_zero_shot_guidance_2.csv', user_content_header="Please find the airline names in this tweet. List the full name of the airline. Don't use abbreviations. Separate airline names with commas. Include airlines ONLY. For airlines like US Airways, include the space. Tweet:")
    print("Additional guidance 2 test performance:", result_06)
    new_row = pd.DataFrame({'Test Name': ['Additional Guidance 2 Test'], 'Performance': [result_06]})
except Exception as e:
    print("Error during additional guidance 2 test:", e)
    new_row = pd.DataFrame({'Test Name': ['Additional Guidance 2 Test'], 'Performance': [str(e)]})
results_df = pd.concat([results_df, new_row], ignore_index=True)

# Export results to CSV
results_df.to_csv('zero_shot_results.csv', index=False)

