from run_test import run_test
import pandas as pd

result = run_test('airline_test.csv', 'airline_test_zero_shot_guidance_2.csv', model='gpt-3.5-turbo', temperature=0.0, user_content_header="Please find the airline names in this tweet. List the full name of the airline. Don't use abbreviations. Separate airline names with commas. Include airlines ONLY. For airlines like US Airways, include the space. Tweet:")

print(result)

