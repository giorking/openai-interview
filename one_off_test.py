from run_test import run_test
import pandas as pd

result = run_test('airline_test.csv', 'airline_test_zero_shot-gpt3.csv', model='gpt-3.5-turbo')

print(result)

