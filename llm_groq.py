import os
from groq import Groq


# --------------------------------------------------
# Load API Key
# --------------------------------------------------
API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found. Set it first.")


client = Groq(api_key=API_KEY)


# --------------------------------------------------
# Get Answer from Groq
# --------------------------------------------------
def get_answer(context, question):

    prompt = f"""
You are an AI tutor helping a student prepare for viva exams.

Use ONLY the following content to answer clearly and accurately.

CONTENT:
{context}

QUESTION:
{question}

Give a simple and well-explained answer.
"""


    response = client.chat.completions.create(

        # ✅ UPDATED MODEL (WORKING)
        model="llama-3.1-8b-instant",

        messages=[
            {"role": "system", "content": "You are a helpful tutor."},
            {"role": "user", "content": prompt}
        ],

        temperature=0.3,
        max_tokens=500
    )


    return response.choices[0].message.content.strip()



