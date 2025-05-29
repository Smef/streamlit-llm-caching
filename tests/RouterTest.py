import sys
import os

from dotenv import load_dotenv

load_dotenv()

sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from app.Agent import Agent

# initialize LLM service


query = "Where can I find the latest financial reports for the last 10 years?"

agent = Agent()

result = agent.route_query(query)

print(result)
