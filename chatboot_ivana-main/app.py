import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# --- RAG Setup ---
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')
PDF_FILE = os.path.join(DATA_PATH, 'i_s.pdf')
INDEX_PATH = os.path.join(DATA_PATH, 'faiss_index')

vector_db = None

def init_rag():
    global vector_db
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Check if index already exists to save time/memory
    if os.path.exists(INDEX_PATH):
        print("Loading existing FAISS index...")
        try:
            vector_db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            print("Index loaded successfully.")
            return
        except Exception as e:
            print(f"Error loading index: {e}. Recreating...")

    # If no index, load PDF and create it
    if os.path.exists(PDF_FILE):
        print(f"Creating new knowledge base from {PDF_FILE}...")
        try:
            loader = PyPDFLoader(PDF_FILE)
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(documents)
            
            vector_db = FAISS.from_documents(splits, embeddings)
            vector_db.save_local(INDEX_PATH)
            print("Vector database created and saved locally.")
        except Exception as e:
            print(f"Error initializing RAG: {e}")
    else:
        print(f"WARNING: Knowledge base PDF not found at {PDF_FILE}")

# Initialize RAG on startup
init_rag()

# Configure Groq LLM via LangChain
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Define the System Prompt
system_prompt = (
    "Eres un tutor experto en Ingeniería de Software. Tu objetivo es ayudar a estudiantes "
    "a comprender conceptos fundamentales y avanzados de esta área de manera clara, "
    "didáctica y explicativa.\n\n"
    "Utiliza el siguiente contexto recuperado para responder a la pregunta del usuario. "
    "Si la información no está en el contexto, usa tu conocimiento general pero prioriza "
    "lo que dice el documento proporcionado.\n\n"
    "Contexto:\n"
    "{context}\n\n"
    "Pregunta:\n"
    "{text}"
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt)
])

# Create the LangChain chain
chain = prompt_template | llm | StrOutputParser()

@app.route('/')
def index():
    """Render the main chat interface."""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Endpoint to handle chat messages."""
    data = request.json
    user_message = data.get('message')

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        # Retrieve context if vector_db is available
        context = ""
        if vector_db:
            docs = vector_db.similarity_search(user_message, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])

        # Generate response using LangChain
        response = chain.invoke({"text": user_message, "context": context})
        return jsonify({"response": response})
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return jsonify({"error": "Hubo un error al procesar tu solicitud. Asegúrate de que la API Key de Groq sea válida."}), 500

if __name__ == '__main__':
    # Ensure the GROQ_API_KEY is set
    if not os.getenv("GROQ_API_KEY"):
        print("WARNING: GROQ_API_KEY not found in environment variables.")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
