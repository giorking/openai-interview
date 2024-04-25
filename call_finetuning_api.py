from openai import OpenAI

def upload_training_data(file_path="airline_train.jsonl"):
    client = OpenAI()
    response = client.files.create(
        file=open(file_path, "rb"),
        purpose="fine-tune"
    )
    return response.id
    
def create_finetune_model(file_id):
    client = OpenAI()
    client.fine_tuning.jobs.create(
        training_file=file_id,
        model='gpt-3.5-turbo'
    )

