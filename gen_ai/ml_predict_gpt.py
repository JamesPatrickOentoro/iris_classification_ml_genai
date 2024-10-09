from openai import OpenAI
client = OpenAI(
 api_key="",
)
chat_completion = client.chat.completions.create(
 model='gpt-3.5-turbo',messages=[{'role': 'user', 'content': 'Hello world'}]
)