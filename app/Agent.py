from openai import OpenAIError
import re
import json

from app.CacheService import CacheService
from app.OpenAiClient import get_open_ai_client
from app.LlmService import LlmService
from app.DocumentQuery import search_document

class Agent:

    cache_service = CacheService()

    def query(self, query_string: str):

        if query_string is None:
            return "Please ask a question"

        llm_service = LlmService()


        # get the embedding for the query
        query_embedding_array = llm_service.embed_query(query_string)


        # check to see if there's a matching answer already
        answer = self.cache_service.find_similar_answer(query_embedding_array)

        # return a real answer if we have one
        if answer is not None:
            return answer

        # we didn't find an answer

        #  Terminal color codes to make the printed output easier to read and visually structured
        CYAN = "\033[96m"
        GREY = "\033[90m"
        BOLD = "\033[1m"
        RESET = "\033[0m"


        # figure out where we need to route the query
        try:
            response = self.route_query(query_string)
        except Exception as route_err:
            # If something goes wrong while classifying the query, show an error message
            print(f"{BOLD}{CYAN}🤖 BOT RESPONSE:{RESET}\n")
            print(f"Routing error: {route_err}\n")
            return

        # Extract the routing decision and the reason behind it
        action = response.get("action")  # e.g., "OPENAI_QUERY"
        reason = response.get("reason")  # e.g., "Related to OpenAI tools"

        # Step 3: Show the selected route and why it was chosen
        print(f"{GREY}📍 Selected Route: {action}")
        print(f"📝 Reason: {reason}")
        print(f"⚙️ Processing query...{RESET}\n")


        if action == '10K_DOCUMENT_QUERY':
            answer = search_document(query_string, query_embedding_array)
        elif action == 'INTERNET_QUERY':
            response = llm_service.query_llm_for_answer(query_string)
            answer = response.output_text
        else:
            answer = "Sorry, I don't know how to answer that yet."



        # store the answer in the cache
        self.cache_service.add_answer_to_cache(answer, query_embedding_array)

        return answer

    def route_query(self, user_query: str):

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

    def get_cache_hit_count(self):
        return self.cache_service.get_cache_hit_count()

