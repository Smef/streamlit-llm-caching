import os
import numpy as np
from nomic import embed
from openai import OpenAI


class LlmService:

    client = OpenAI(
        # This is the default and can be omitted
        api_key=os.getenv("OPENAI_API_KEY"),
    )


    dimensions = 768


    def embed_query(self, query_string):

        print ("Generating embedding for query: " + query_string)
        embed_results = embed.text(
            texts=[query_string],
            model='nomic-embed-text-v1.5',
            task_type="search_query",
            inference_mode='local',
            dimensionality=self.dimensions,
        )

        query_embedding = embed_results['embeddings']
        embedding_array = np.array(query_embedding)

        return embedding_array

    def query_llm_for_answer(self, query_string, context=None):

        if context is not None:
            instructions = f"""Based on the following context, respond to the user's query. If the context doesn't provide enough information to answer the question, just say that you don't know. \n\n
                            ===BEGIN CONTEXT=== \n\n
                            {context}\n\n 
                            ===END CONTEXT==="""
        else:
            instructions = "Respond to the user's query."


        response = self.client.responses.create(
            model="gpt-4o",
            max_output_tokens=250,
            instructions=instructions,
            input=query_string,
        )


        return response
