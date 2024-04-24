test_details = {
    "Baseline Test": {
        "model": "gpt-3.5-turbo",
        "temperature": 0.0,
        "csv_file": "airline_test_zero_shot.csv"
    },
    "GPT-4 Test": {
        "model": "gpt-4",
        "temperature": 0.0,
        "csv_file": "airline_test_zero_shot_gpt4.csv"
    },
    "High Temperature Test": {
        "model": "gpt-3.5-turbo",
        "temperature": 1.0,
        "csv_file": "airline_test_zero_shot_high_temperature.csv"
    },
    "Additional Guidance 1 Test": {
        "model": "gpt-3.5-turbo",
        "temperature": 0.0,
        "csv_file": "airline_test_zero_shot_guidance_1.csv",
        "user_content_header": "Please find the airline names in this tweet. List the full name of the airline. Don't use abbreviations. Separate airline names with commas. Tweet:"
    },
    "Additional Guidance 2 Test": {
        "model": "gpt-3.5-turbo",
        "temperature": 0.0,
        "csv_file": "airline_test_zero_shot_guidance_2.csv",
        "user_content_header": "Please find the airline names in this tweet. List the full name of the airline. Don't use abbreviations. Separate airline names with commas. Include airlines ONLY. For airlines like US Airways, include the space. Tweet:"
    }
}
