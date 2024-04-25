from openai import OpenAI

default_model = "gpt-3.5-turbo"
default_temperature = 0
default_system_content = "You are a helpful assistant"
default_user_content = "Please find the airline names in this tweet:"

def initiate_chat_system(model=default_model, temperature=default_temperature, system_content=default_system_content):
    client = OpenAI(
        # Defaults to os.environ.get("OPENAI_API_KEY")
    )

    chat_completion = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_content},
        ]
    )

def get_chat_completion(model=default_model, temperature=default_temperature, user_content=default_user_content):
    client = OpenAI(
        # Defaults to os.environ.get("OPENAI_API_KEY")
    )

    chat_completion = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "user", "content": user_content}
        ]
    )
    
    # Return message contents
    return chat_completion.choices[0].message.content
