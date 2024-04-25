import pandas as pd
from call_finetuning_api import upload_training_data, create_finetune_model
from call_api import default_system_content, default_user_content

def convert_csv_to_jsonl(csv_file_path, jsonl_file_path):
    # Load the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Prepare data for chat completion format
    df['messages'] = df.apply(lambda row: [
        {"role": "system", "content": default_system_content},
        {"role": "user", "content": f"{default_user_content} {row['tweet']}"},
        {"role": "assistant", "content": row['airlines']}
    ], axis=1)
    
    # Select the relevant column and save to jsonl
    df[['messages']].to_json(jsonl_file_path, orient='records', lines=True)
    print(f"Data from {csv_file_path} has been converted and saved to {jsonl_file_path}")

# Specify the file paths
csv_file_path = 'airline_train.csv'
jsonl_file_path = 'airline_train.jsonl'

# Call the function to convert and save the data
convert_csv_to_jsonl(csv_file_path, jsonl_file_path)

# Upload the training data and get the file ID
file_id = upload_training_data(jsonl_file_path)

# Create a fine-tuning model using the uploaded file ID
create_finetune_model(file_id)

print("The training data has been uploaded to OpenAI. Please check your email for confirmation and further instructions.")
