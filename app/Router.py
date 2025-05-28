from openai import OpenAIError
import re
import json  # Regular expressions for cleaning or preprocessing inputs (if needed)

from .OpenAiClient import get_open_ai_client


def route_query(user_query: str):

    open_ai_client = get_open_ai_client()

    router_system_prompt = f"""
    As a professional query router, your objective is to correctly classify user input into one of three categories based on the source most relevant for answering the query:
    1. "OPENAI_QUERY": If the user's query appears to be answerable using information from OpenAI's official documentation, tools, models, APIs, or services (e.g., GPT, ChatGPT, embeddings, moderation API, usage guidelines).
    2. "10K_DOCUMENT_QUERY": If the user's query pertains to a collection of documents from the 10k annual reports, datasets, or other structured documents, typically for research, analysis, or financial content.
    3. "INTERNET_QUERY": If the query is neither related to OpenAI nor the 10k documents specifically, or if the information might require a broader search (e.g., news, trends, tools outside these platforms), route it here.

    Your decision should be made by assessing the domain of the query.

    Always respond in this valid JSON format:
    {{
        "action": "OPENAI_QUERY" or "10K_DOCUMENT_QUERY" or "INTERNET_QUERY",
        "reason": "brief justification",
        "answer": "AT MAX 5 words answer. Leave empty if INTERNET_QUERY"
    }}

    EXAMPLES:

    - User: "How to fine-tune GPT-3?"
    Response:
    {{
        "action": "OPENAI_QUERY",
        "reason": "Fine-tuning is OpenAI-specific",
        "answer": "Use fine-tuning API"
    }}

    - User: "Where can I find the latest financial reports for the last 10 years?"
    Response:
    {{
        "action": "10K_DOCUMENT_QUERY",
        "reason": "Query related to annual reports",
        "answer": "Access through document database"
    }}

    - User: "Top leadership styles in 2024"
    Response:
    {{
        "action": "INTERNET_QUERY",
        "reason": "Needs current leadership trends",
        "answer": ""
    }}

    - User: "What's the difference between ChatGPT and Claude?"
    Response:
    {{
        "action": "INTERNET_QUERY",
        "reason": "Cross-comparison of different providers",
        "answer": ""
    }}

    Strictly follow this format for every query, and never deviate.
    User: {user_query}
    """

    try:
        # Query the GPT-4 model with the router prompt and user input
        response = open_ai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": router_system_prompt}],
        )

        # Extract and parse the model's JSON response
        task_response = response.choices[0].message.content
        json_match = re.search(r"\{.*}", task_response, re.DOTALL)
        json_text = json_match.group()
        parsed_response = json.loads(json_text)
        return parsed_response

    # Handle OpenAI API errors (e.g., rate limits, authentication)
    except OpenAIError as api_err:
        return {
            "action": "INTERNET_QUERY",
            "reason": f"OpenAI API error: {api_err}",
            "answer": "",
        }

    # Handle case where model response isn't valid JSON
    except json.JSONDecodeError as json_err:
        return {
            "action": "INTERNET_QUERY",
            "reason": f"JSON parsing error: {json_err}",
            "answer": "",
        }

    # Catch-all for any other unforeseen issues
    except Exception as err:
        return {
            "action": "INTERNET_QUERY",
            "reason": f"Unexpected error: {err}",
            "answer": "",
        }
