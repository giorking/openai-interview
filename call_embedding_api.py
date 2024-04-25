from openai import OpenAI

def get_embeddings(text):
    client = OpenAI()
    response = client.embeddings.create(
        input=text,
        model='text-embedding-3-small'
    )
    return response.data[0].embedding

