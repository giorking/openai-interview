from openai import OpenAI

def initiate_chat_system(model="gpt-3.5-turbo", temperature=0, system_content="You are a helpful assistant"):
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

def get_chat_completion(model="gpt-3.5-turbo", temperature=0, user_content="Please find the airline names in this tweet"):
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
