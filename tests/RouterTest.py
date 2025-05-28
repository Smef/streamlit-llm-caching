import sys
import os

sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from app.Router import route_query

# initialize LLM service


query = "What is the revenue of Lyft?"

result = route_query(query)

print(result)
