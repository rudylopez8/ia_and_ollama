import ollama
for _ in range(1, 10):
	res = ollama.chat(
		model="minicpm-v",
		messages=[
			{
				'role': 'user',
				'content': 'Décrit cette image en français',
				'images': ['./img2.png']
			}
		]
	)

	print(res['message']['content'])
