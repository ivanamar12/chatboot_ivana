import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure Groq LLM via LangChain
# The model can be changed here (e.g., 'llama3-70b-8192' or 'mixtral-8x7b-32768')
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
    "Debes ser capaz de responder preguntas sobre:\n"
    "- Ciclo de vida del desarrollo de software (SDLC)\n"
    "- Metodologías ágiles (Scrum, Kanban)\n"
    "- Análisis y diseño de sistemas\n"
    "- Diagramas UML\n"
    "- Arquitectura de software y patrones de diseño\n"
    "- Pruebas de software (Unitarias, Integración, etc.)\n"
    "- Control de versiones (Git)\n"
    "- Ingeniería de requisitos y calidad del software\n\n"
    "Usa un tono profesional pero accesible. Si el usuario hace una pregunta fuera de este ámbito, "
    "recuérdale amablemente que tu especialidad es la Ingeniería de Software."
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{text}")
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
        # Generate response using LangChain
        response = chain.invoke({"text": user_message})
        return jsonify({"response": response})
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return jsonify({"error": "Hubo un error al procesar tu solicitud. Asegúrate de que la API Key de Groq sea válida."}), 500

if __name__ == '__main__':
    # Ensure the GROQ_API_KEY is set
    if not os.getenv("GROQ_API_KEY"):
        print("WARNING: GROQ_API_KEY not found in environment variables.")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
