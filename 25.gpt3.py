import openai

# Set your OpenAI API key
api_key = "YOUR_API_KEY"
openai.api_key = api_key

# Define your prompt
prompt = "Once upon a time,"

# Generate text based on the prompt
response = openai.Completion.create(
    engine="text-davinci-002",  # Choose the engine
    prompt=prompt,
    max_tokens=100  # Adjust the desired length of generated text
)

# Print the generated text
print(response.choices[0].text.strip())
