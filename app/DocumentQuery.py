from qdrant_client import QdrantClient   # Qdrant is used as the vector store to retrieve documents based on similarity
from app.LlmService import LlmService


def search_document(query_string, query_embedding_array):

    client = QdrantClient(path="/Users/smef/Code/Personal/ai-agent-course/streamlit-llm-caching/database/qdrant_data")


    # default to just the 10k docs
    action = '10K_DOCUMENT_QUERY'

    # Define mapping of routing labels to their respective Qdrant collections
    collections = {
        "OPENAI_QUERY": "opnai_data",  # Collection of OpenAI documentation embeddings
        "10K_DOCUMENT_QUERY": "10k_data"  # Collection of 10-K financial document embeddings
    }

    try:
        # Ensure that the provided action is valid
        if action not in collections:
            return "Invalid action type for retrieval."

        # Step 2: Retrieve top-matching chunks from the relevant Qdrant collection
        try:
            text_hits = client.query_points(
                collection_name=collections[action],  # Choose the right collection based on routing
                query=query_embedding_array[0],  # The embedding of the user's query
                limit=3  # Fetch top 3 relevant chunks
            ).points
        except Exception as qdrant_err:
            return f"Vector DB query error: {qdrant_err}"  # Handle Qdrant access issues

        # Extract the raw content from the retrieved vector hits
        contents = [point.payload['content'] for point in text_hits]

        # If no relevant content is found, return early
        if not contents:
            return "No relevant content found in the database."

        # Step 3: Pass the retrieved context to the RAG model to generate a response
        try:
            llmService = LlmService()
            answer = llmService.query_llm_for_answer(query_string, context=contents)
            return answer.output_text
        except Exception as rag_err:
            return f"RAG response error: {rag_err}"  # Handle generation failures

    # Catch any unforeseen errors in the overall process
    except Exception as err:
        return f"Unexpected error: {err}"
