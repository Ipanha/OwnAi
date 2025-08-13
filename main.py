
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from google.api_core.exceptions import ResourceExhausted, NotFound

# LangChain + Gemini imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain

# --- Step 1: Load environment variables ---
load_dotenv()

# Check if the Google API key is set
if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError("GOOGLE_API_KEY not found in .env file.")

# --- Step 2: Create the FastAPI application ---
app = FastAPI(
    title="CheckinAi Service",
    description="An AI service for CheckinMe customer support.",
    version="1.0.0"
)

# --- Step 3: Add CORS Middleware ---
origins = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Step 4: Define data models for API requests and responses ---
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

vector_store = None

# --- Helper function for language detection ---
def is_khmer(text: str) -> bool:
    """Checks if a string contains any Khmer Unicode characters."""
    for char in text:
        # Khmer Unicode range is U+1780 to U+17FF
        if '\u1780' <= char <= '\u17FF':
            return True
    return False

# --- Step 5: Load the knowledge base on server startup ---
@app.on_event("startup")
def load_knowledge_base():
    global vector_store
    KNOWLEDGE_BASE_PATH = "checkinme_knowledge.pdf"

    if not os.path.exists(KNOWLEDGE_BASE_PATH):
        raise FileNotFoundError(f"Knowledge base file not found at '{KNOWLEDGE_BASE_PATH}'")

    print(f"Loading knowledge base from: {KNOWLEDGE_BASE_PATH}...")
    loader = PyPDFLoader(KNOWLEDGE_BASE_PATH)
    documents = loader.load()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(documents, embeddings)
    print("Knowledge base loaded successfully.")

# --- Step 6: Define the main API endpoint ---
@app.post("/ask", response_model=AnswerResponse)
async def ask_ai(request: QuestionRequest):
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Knowledge base is not loaded or failed to load.")

    question = request.question
    print(f"Received question: {question}")

    try:
        relevant_docs = vector_store.similarity_search(question, k=3)

        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
        
        chain = load_qa_chain(llm, chain_type="stuff")
        
        # **THE FIX IS HERE:** Automatically detect language and set the prompt.
        final_prompt = question
        if is_khmer(question):
            print("Khmer language detected. Answering in Khmer.")
            final_prompt = f"Please answer the following question in the Khmer language, based on the context provided: {question}"
        else:
            print("English language detected. Answering in English.")
        
        result = chain.invoke({"input_documents": relevant_docs, "question": final_prompt})

        return {"answer": result['output_text']}

    except NotFound:
        raise HTTPException(status_code=404, detail="The specified AI model was not found. Please check the model name.")
    except ResourceExhausted:
        raise HTTPException(status_code=429, detail="The AI is busy due to rate limits, please try again in a minute.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

