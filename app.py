import streamlit as st
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from faster_whisper import WhisperModel

from rag_pipeline import (
    ask_question,
    generate_viva_questions,
    generate_notes,
    generate_mcqs
)


# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="SmartDoc-RAG",
    layout="wide"
)


# --------------------------------------------------
# Global Theme
# --------------------------------------------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(to bottom, #020617, #020617);
}

.block-container {
    padding-top: 2.5rem;
    padding-bottom: 2.5rem;
}

.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    padding: 9px 18px;
    font-weight: 600;
    border: none;
    transition: all 0.25s ease;
}

.stButton > button:hover {
    background-color: #1e40af;
    transform: scale(1.05);
}

.smart-card {
    background: linear-gradient(135deg, #e0f2fe, #dbeafe);
    padding: 22px 26px;
    border-radius: 14px;
    box-shadow: 0px 4px 10px rgba(37,99,235,0.15);
    margin: 28px 0px;
    font-size: 21px;
    font-weight: 600;
    color: #1e3a8a;
    border-left: 5px solid #2563eb;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Load Models
# --------------------------------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


@st.cache_resource
def load_db(embeddings):
    return Chroma(
        collection_name="docs",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )


@st.cache_resource
def load_whisper():
    return WhisperModel("small", device="cpu", compute_type="int8")


embeddings = load_embeddings()
db = load_db(embeddings)
whisper_model = load_whisper()


# --------------------------------------------------
# Session State
# --------------------------------------------------
if "user_profile" not in st.session_state:
    st.session_state["user_profile"] = {
        "name": "Taruni Middela",
        "course": "MSc AI & ML",
        "role": "Student"
    }

if "mcq_scores" not in st.session_state:
    st.session_state["mcq_scores"] = []

if "transcript" not in st.session_state:
    st.session_state["transcript"] = ""

if "mcq_data" not in st.session_state:
    st.session_state["mcq_data"] = None

if "mcq_answers" not in st.session_state:
    st.session_state["mcq_answers"] = {}


# --------------------------------------------------
# Card Function
# --------------------------------------------------
def section_card(title):
    st.markdown(f"""
    <div class="smart-card">{title}</div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# Hero Header
# --------------------------------------------------
st.markdown("""
<div style="
    background: linear-gradient(to right, #0f172a, #1e3a8a);
    padding: 38px;
    border-radius: 18px;
    color: white;
    text-align: center;
    margin-bottom: 30px;
">
    <h1>📘 SmartDoc-RAG</h1>
    <h3>AI-Powered Learning Assistant</h3>
    <p>PDF Analysis • Audio Transcription • Viva Preparation • MCQ Evaluation</p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
name = st.session_state["user_profile"]["name"]
course = st.session_state["user_profile"]["course"]
role = st.session_state["user_profile"]["role"]

st.sidebar.markdown(f"""
<div style="
    background:#020617;
    padding:18px;
    border-radius:14px;
    text-align:center;
    margin-bottom:20px;
">
<b style="color:#60a5fa;">👤 {name}</b><br>
🎓 {course}<br>
📚 {role}
</div>

<div style="
    background:#020617;
    padding:14px;
    border-radius:14px;
    margin-bottom:15px;
">
<b style="color:#93c5fd;">🎯 SmartDoc</b><br>
<span style="color:#c7d2fe;">Learning Dashboard</span>
</div>
""", unsafe_allow_html=True)


mode = st.sidebar.radio(
    "📌 Select Module",
    ["🏠 PDF Analysis", "🎤 Audio Processing", "📝 Assessment"]
)


# --------------------------------------------------
# PDF MODULE
# --------------------------------------------------
if mode == "🏠 PDF Analysis":

    section_card("📄 PDF Upload & Analysis")

    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])

    with col2:
        level = st.selectbox("Difficulty Level", ["Easy", "Medium", "Advanced"])


    if uploaded_pdf:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_pdf.read())
            pdf_path = tmp.name


        loader = PyPDFLoader(pdf_path)
        pages = loader.load()


        splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=150
        )

        docs = splitter.split_documents(pages)


        for i, doc in enumerate(docs):
            doc.metadata["source"] = "pdf"
            doc.metadata["chunk_id"] = i


        with st.spinner("📄 Indexing PDF..."):
            db.add_documents(docs)
            db.persist()


        st.success(f"✅ Indexed {len(docs)} chunks")


        section_card("🤖 Query Interface")

        question = st.text_input("Enter your question")


        if st.button("Retrieve Answer"):

            if question.strip():

                with st.spinner("🔍 Searching..."):
                    answer, sources, page_no, para = ask_question(
                        question, level
                    )

                st.subheader("📘 Answer")
                st.write(answer)

                st.subheader("📄 Source")
                st.write("Page:", page_no)


        section_card("🎯 Viva Preparation")

        if st.button("Generate Viva Questions"):

            with st.spinner("🎯 Generating..."):
                viva_qs = generate_viva_questions()

            st.write(viva_qs)



# --------------------------------------------------
# AUDIO MODULE
# --------------------------------------------------
elif mode == "🎤 Audio Processing":

    section_card("🎤 Audio Transcription & Notes")

    uploaded_audio = st.file_uploader(
        "Upload Audio",
        type=["wav", "mp3", "m4a"]
    )


    if uploaded_audio:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_audio.read())
            audio_path = tmp.name


        if st.button("Start Transcription"):

            with st.spinner("🎤 Transcribing..."):

                segments, info = whisper_model.transcribe(audio_path)

                text = ""

                for seg in segments:
                    text += seg.text + " "

                st.session_state["transcript"] = text


                st.text_area("Transcript", text, height=250)

                st.success("✅ Completed")


        if st.session_state["transcript"]:

            section_card("📚 Notes Generation")

            if st.button("Generate Notes"):

                with st.spinner("📝 Creating..."):

                    notes = generate_notes(
                        st.session_state["transcript"]
                    )

                st.write(notes)

                st.download_button(
                    "⬇️ Download Notes",
                    notes,
                    "audio_notes.txt",
                    "text/plain"
                )



# --------------------------------------------------
# MCQ MODULE (FIXED)
# --------------------------------------------------
elif mode == "📝 Assessment":

    section_card("📝 MCQ Assessment & Evaluation")


    if st.button("Generate Test"):

        with st.spinner("📘 Preparing test..."):

            mcq_text = generate_mcqs()


        if mcq_text:

            st.session_state["mcq_data"] = mcq_text
            st.session_state["mcq_answers"] = {}

            st.success("✅ Test Generated")


    # Show Questions
    if st.session_state["mcq_data"]:

        mcqs = st.session_state["mcq_data"].split("Q")[1:]

        correct_answers = {}


        for i, block in enumerate(mcqs, 1):

            lines = block.strip().split("\n")

            question = lines[0]
            options = lines[1:5]
            answer = lines[-1].replace("Answer:", "").strip()

            correct_answers[i] = answer


            st.markdown(f"**Q{i}. {question}**")

            choice = st.radio(
                "",
                options,
                key=f"mcq_{i}"
            )

            st.session_state["mcq_answers"][i] = choice

            st.divider()


        # ✅ Submit Button INSIDE MCQ MODULE (FIXED)
        if st.button("Submit Test"):

            score = 0
            total = len(correct_answers)


            for i in correct_answers:

                selected = st.session_state["mcq_answers"].get(i, "")

                if selected.startswith(correct_answers[i]):
                    score += 1


            percent = (score / total) * 100

            st.session_state["mcq_scores"].append(percent)


            best = max(st.session_state["mcq_scores"])
            avg = sum(st.session_state["mcq_scores"]) / len(st.session_state["mcq_scores"])
            tests = len(st.session_state["mcq_scores"])


            section_card("📈 Performance Summary")

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric("Tests", tests)

            with c2:
                st.metric("Best", f"{best:.1f}%")

            with c3:
                st.metric("Average", f"{avg:.1f}%")


            section_card("📊 Assessment Result")

            st.success(f"Score: {score} / {total}")
            st.info(f"Percentage: {percent:.2f}%")


            if percent >= 80:
                st.success("🌟 Excellent!")
            elif percent >= 60:
                st.warning("👍 Good")
            else:
                st.error("⚠️ Improve")



# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("""
<div style="
    margin-top:55px;
    padding:22px;
    background:#020617;
    border-radius:12px;
    text-align:center;
    color:#cbd5f5;
">
<b>SmartDoc-RAG</b><br>
MSc AI & ML • Taruni Middela © 2026
</div>
""", unsafe_allow_html=True)
























