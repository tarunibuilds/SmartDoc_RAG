from sentence_transformers import SentenceTransformer


# Load pretrained model once
model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embeddings(texts):
    """
    Generate embeddings for a list of texts (chunks)
    """

    if isinstance(texts, str):
        texts = [texts]

    embeddings = model.encode(
        texts,
        show_progress_bar=False
    )

    return embeddings

