
from llm_service import generate_answer

context = """
Employees can work remotely three days per week.
The remaining two days are required in the office.
"""

question = "How many days can employees work from home?"

answer = generate_answer(question, context)

print("\nAI ANSWER:")
print(answer)
