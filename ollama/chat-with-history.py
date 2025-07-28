from ollama import chat


messages = [
  {
    'role': 'system',
    "content": "Tu es un programme utile et fiable.",
  },
]

while True:
  user_input = input('Chat with history: ')
  response = chat(
    'mistral-nemo',
    messages=messages
    + [
      {'role': 'user', 'content': user_input},
    ],
  )

  # Add the response to the messages to maintain the history
  messages += [
    {'role': 'user', 'content': user_input},
    {'role': 'assistant', 'content': response.message.content},
  ]
  print(response.message.content + '\n')
