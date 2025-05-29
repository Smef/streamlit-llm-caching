import faiss
import numpy as np
from app.LlmService import LlmService


class CacheService:

    cache = []

    index = None


    dimensions = 768


    cache_hit_count = 0

    llm_serivce = None

    def __init__(self):

        if self.llm_serivce is None:
            self.llm_service = LlmService()

        if self.index is None:
            self.index = faiss.IndexFlatL2(self.dimensions)

            example_query = "What is the capital of France?"

            embedding_array = self.llm_service.embed_query(example_query)

            self.index.add(embedding_array)
            self.cache.append('Paris')

            example_query = "What is the capital of Canada?"

            embedding_array = self.llm_service.embed_query(example_query)

            self.index.add(embedding_array)
            self.cache.append('Ottowa')


            example_query = "What is the capital of Germany?"

            embedding_array = self.llm_service.embed_query(example_query)

            self.index.add(embedding_array)
            self.cache.append('Munich')



        index_size = self.index.ntotal
        print("Total Elements in Index: " + str(index_size))

        print(self.index.is_trained)

    def get_cache_hit_count(self):
        return self.cache_hit_count

    # cache an entry
    def add_answer_to_cache(self, answer_str: str, query_embedding: np.array):
        self.index.add(query_embedding)
        self.cache.append(answer_str)


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
        self.cache_hit_count += 1

        return closest_match_answer

