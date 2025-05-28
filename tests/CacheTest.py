import sys
import os

sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ..app.LlmService import LlmService

# initialize LLM service

service = LlmService()

query = "What is the capital of Portugal?"

result = service.query(query)

print(result)
