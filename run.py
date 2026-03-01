from twin_lib import ask_question

q = 'calculate this 5 + 7 + 3 - 2 / 34 + 1 * 4 + 4 - 8 - 5 + 523 / 4 + 1032 + 52342 / 54 * 1321 / 43 * 453 + 423453 / 453 * 43 * 52342 / 54 + 52342 / 54 * 43242 / 5423 - 5342 + 542352 / 532 + 5442'

answer = ask_question(
    question=q,
    model_name="ministral-3:latest",
    ollama_url="http://localhost:11434"
)

print(answer)