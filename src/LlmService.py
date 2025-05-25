import faiss
from sentence_transformers import SentenceTransformer
from nomic import embed
import numpy as np
from openai import OpenAI
import os

from dotenv import load_dotenv

load_dotenv()


class LlmService:

    dimensions = 768

    cache = []

    client = OpenAI(
        # This is the default and can be omitted
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    def __init__(self, json_file='../tmp/cache.json'):

        self.index = faiss.IndexFlatL2(self.dimensions)

        example_query = "What is the capital of France?"

        embedding_array = self.embed_query(example_query)

        self.index.add(embedding_array)
        self.cache.append('Paris')

        example_query = "What is the capital of Canada?"

        embedding_array = self.embed_query(example_query)

        self.index.add(embedding_array)
        self.cache.append('Ottowa')


        example_query = "What is the capital of Germany?"

        embedding_array = self.embed_query(example_query)

        self.index.add(embedding_array)
        self.cache.append('Munich')



        index_size = self.index.ntotal
        print("Total Elements in Index: " + str(index_size))

        print(self.index.is_trained)



    # cache an entry
    def add_answer_to_cache(self, answer_str: str, query_embedding: np.array):


        self.index.add(query_embedding)


    def query(self, query_string):

        if query_string is None:
            return "Please ask a question"


        # get the embedding for the query
        query_embedding_array = self.embed_query(query_string)

        # check to see if there's a matching answer already
        answer = self.find_similar_answer(query_embedding_array)

        # return a real answer if we have one
        if answer is not None:
            return answer

        # we didn't find an answer

        # get one from the LLM
        response = self.query_llm_for_answer(query_string)

        answer = response.output_text

        # store the answer in the cache
        self.add_answer_to_cache(answer, query_embedding_array)

        return answer


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

    def query_llm_for_answer(self, query_string):
        # do a query

        response = self.client.responses.create(
            model="gpt-4o",
            max_output_tokens=250,
            # instructions="You are a coding assistant that talks like a pirate.",
            input=query_string,
        )


        return response

    def find_similar_answer(self, query_embedding_array):

        # only return the nearest result
        # search results are a 2-d array of distances and indexes
        search_results = self.index.search(query_embedding_array, k=1 )

        closest_match_index = search_results[1][0][0]
        closest_match_distance = search_results[0][0][0]
        closest_match_score = 1 - closest_match_distance

        # check if our response is similar enough
        min_similarity_threshold = 0.9
        if closest_match_score < min_similarity_threshold:
            print("Cache miss")
            return None


        # get the answer from the cache
        closest_match_answer = self.cache[closest_match_index]

        print("Cache hit!")

        return closest_match_answer
