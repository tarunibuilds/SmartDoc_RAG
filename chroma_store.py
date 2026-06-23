from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


# --------------------------------------------------
# Load Embeddings
# --------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)


# --------------------------------------------------
# Chroma DB Setup
# --------------------------------------------------
DB_PATH = "chroma_db"

db = Chroma(
    collection_name="docs",
    embedding_function=embeddings,
    persist_directory=DB_PATH
)


# --------------------------------------------------
# Add Documents to DB
# --------------------------------------------------
def add_documents(docs):

    documents = []

    for d in docs:

        documents.append(
            Document(
                page_content=d["text"],
                metadata=d["metadata"]
            )
        )

    db.add_documents(documents)

    print(f"✅ Stored {len(documents)} chunks")




