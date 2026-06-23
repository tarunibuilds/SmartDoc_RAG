from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from llm_groq import get_answer


# --------------------------------------------------
# Load Embeddings
# --------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


# --------------------------------------------------
# Load Chroma DB
# --------------------------------------------------
db = Chroma(
    collection_name="docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)


# --------------------------------------------------
# PDF Question Answering (With Difficulty + Page Info)
# --------------------------------------------------
def ask_question(question, level="Medium", k=5):

    docs = db.similarity_search(
        question,
        k=k,
        filter={"source": "pdf"}
    )

    if not docs:
        return "Answer not found in the uploaded PDF.", [], None, None


    # Combine context
    context = "\n\n".join([doc.page_content for doc in docs])


    # Prompt with difficulty
    prompt = f"""
Answer in {level} level.

Use only the context.

Context:
{context}

Question:
{question}
"""


    answer = get_answer(context, prompt)


    # Take first doc for source info
    first_doc = docs[0]

    page_no = first_doc.metadata.get("page", "N/A")

    paragraph = first_doc.page_content[:300] + "..."


    return answer, docs, page_no, paragraph


# --------------------------------------------------
# Viva Question Generator
# --------------------------------------------------
def generate_viva_questions():

    docs = db.similarity_search(
        "important concepts",
        k=8,
        filter={"source": "pdf"}
    )

    if not docs:
        return "No PDF content found."


    context = "\n\n".join([doc.page_content for doc in docs])


    prompt = f"""
You are a teacher preparing viva questions.

Based ONLY on the content below,
generate 8 important viva questions.

CONTENT:
{context}

Give questions in numbered format.
"""


    questions = get_answer(context, prompt)

    return questions


# --------------------------------------------------
# Notes Generator (From Audio Transcript)
# --------------------------------------------------
def generate_notes(transcript):

    prompt = f"""
You are a teacher.

Convert the following lecture transcript into
clear and structured study notes.

Make it easy to understand.

TRANSCRIPT:
{transcript}

Give notes in bullet points.
"""


    notes = get_answer(transcript, prompt)

    return notes


# --------------------------------------------------
# MCQ Generator (From PDF Content)
# --------------------------------------------------
def generate_mcqs():

    docs = db.similarity_search(
        "important concepts for exam questions",
        k=8,
        filter={"source": "pdf"}
    )

    if not docs:
        return None


    context = "\n\n".join([doc.page_content for doc in docs])


    prompt = f"""
You are an expert teacher.

From the following study material,
generate exactly 5 multiple choice questions.

Rules:
- Each question must have 4 options (A, B, C, D)
- Only ONE correct answer
- Provide correct answer at the end
- Keep questions exam-oriented

Format strictly like this:

Q1. Question text
A) Option
B) Option
C) Option
D) Option
Answer: B

CONTENT:
{context}
"""


    mcq_text = get_answer(context, prompt)

    return mcq_text






















