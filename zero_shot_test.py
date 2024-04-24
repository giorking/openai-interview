from run_test import run_test
import pandas as pd

# Prepare a DataFrame to store results
results_df = pd.DataFrame(columns=['Test Name', 'Performance'])

# Baseline test
# Model: gpt-3.5-turbo
# Temperature: 0.0 (using log probability to hit thresholds)
# Prompt: Please find the airline names in this tweet: 
print("Running baseline test...")
result_01 = run_test('airline_test_short.csv', 'airline_test_zero_shot.csv')
print("Baseline test performance:", result_01)
results_df = results_df.append({'Test Name': 'Baseline Test', 'Performance': result_01}, ignore_index=True)

# GPT-4 test
# Model: gpt-4-turbo
# Temperature: 0.0
# Prompt: Please find the airline names in this tweet: 
print("Running GPT-4 test...")
result_02 = run_test('airline_test_short.csv', 'airline_test_zero_shot_gpt4.csv', model='gpt-4-turbo')
print("GPT-4 test performance:", result_02)
results_df = results_df.append({'Test Name': 'GPT-4 Test', 'Performance': result_02}, ignore_index=True)

# Low temperature test
# Model: gpt-3.5-turbo
# Temperature: 0.1
# Prompt: Please find the airline names in this tweet: 
print("Running low temperature test...")
result_03 = run_test('airline_test_short.csv', 'airline_test_zero_shot_low_temperature.csv', temperature=0.1)
print("Low temperature test performance:", result_03)
results_df = results_df.append({'Test Name': 'Low Temperature Test', 'Performance': result_03}, ignore_index=True)

# High temperature test
# Model: gpt-3.5-turbo
# Temperature: 0.8
# Prompt: Please find the airline names in this tweet: 
print("Running high temperature test...")
result_04 = run_test('airline_test_short.csv', 'airline_test_zero_shot_high_temperature.csv', temperature=0.8)
print("High temperature test performance:", result_04)
results_df = results_df.append({'Test Name': 'High Temperature Test', 'Performance': result_04}, ignore_index=True)

# Additional guidance test
# Model: gpt-3.5-turbo
# Temperature: 0.0
# Prompt: Please find the airline names in this tweet. List the full name of the airline. Don't use abbreviations. Separate airline names with commas. Tweet:
print("Running additional guidance test...")
result_05 = run_test('airline_test_short.csv', 'airline_test_zero_shot_guidance.csv', user_content_header="Please find the airline names in this tweet. List the full name of the airline. Don't use abbreviations. Separate airline names with commas. Tweet:")
print("Additional guidance test performance:", result_05)
results_df = results_df.append({'Test Name': 'Additional Guidance Test', 'Performance': result_05}, ignore_index=True)

# Export results to CSV
results_df.to_csv('zero_shot_results.csv', index=False)

