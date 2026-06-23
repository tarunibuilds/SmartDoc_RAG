from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# --------------------------------------------------
# Semantic Search Function
# --------------------------------------------------
def search_documents(query_embedding, document_embeddings, documents, top_k=3):
    """
    Returns top_k most similar documents using cosine similarity
    """

    similarities = cosine_similarity(
        [query_embedding],
        document_embeddings
    )[0]


    ranked_results = sorted(
        zip(similarities, documents),
        key=lambda x: x[0],
        reverse=True
    )


    return ranked_results[:top_k]


