import ollama

response = ollama.chat(
    model='minicpm-v',
    messages=[{
        'role': 'user',
        'content': 'Describe this image in detail.',
        'images': ['captured_image.jpg']
    }]
)

print(response)