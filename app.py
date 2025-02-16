import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
from dotenv import load_dotenv
import sys

# Ensure the correct SQLite version is used
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    print("pysqlite3 is not installed, using system sqlite3")


from chromadb.config import Settings
import chromadb

# Initialize ChromaDB with settings
chroma_client = chromadb.EphemeralClient()



# Load environment variables
load_dotenv()
GENAI_API_KEY = os.getenv("GENAI_API_KEY")
genai.configure(api_key=GENAI_API_KEY)

# Initialize ChromaDB
DB_PATH = "./chroma_db"
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_or_create_collection(name="qa_collection")

# Load CSV Data
csv_file = "datafile.csv"

# Function to generate embeddings using Gemini
def generate_embedding(text):
    """Generate embedding vector using Gemini API."""
    try:
        response = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        return response["embedding"]
    except Exception as e:
        st.error(f"Embedding Error: {str(e)}")
        return None

# Function to store data in ChromaDB
def store_data_in_chroma():
    """Stores CSV data in ChromaDB if it does not exist."""
    if collection.count() == 0:
        st.write("🔄 Generating embeddings and storing data in ChromaDB...")
        df = pd.read_csv(csv_file)  # Load data

        for index, row in df.iterrows():
            question = row["question"]
            answer = row["answer"]
            embedding = generate_embedding(question)
            
            if embedding:
                collection.add(
                    ids=[str(index)], 
                    embeddings=[embedding], 
                    metadatas=[{"question": question, "answer": answer}]
                )
        st.write("✅ Embeddings stored successfully!")

# Function to retrieve the most relevant answer
def get_best_answer(query):
    """Finds the most relevant answer from ChromaDB."""
    query_embedding = generate_embedding(query)
    
    if query_embedding is None:
        return "❌ Error generating embedding for your query."
    
    results = collection.query(query_embeddings=[query_embedding], n_results=1)
    
    if not results["metadatas"] or results["distances"][0][0] > 0.3:
        return "😕 Sorry, I couldn't find a relevant answer."
    
    return results["metadatas"][0][0]["answer"]

# Function to refine the response using Gemini
def refine_with_gemini(context):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(f"Refine and simplify this response: {context}")
    return response.text if response.text else context

# Call function to populate database on startup
store_data_in_chroma()


st.markdown("## 💬 AI Chatbot Assistant 🤖")


# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


user_input = st.text_input("Ask a question:")

if st.button("Send"):
    if user_input:
        best_answer = get_best_answer(user_input)
        refined_answer = refine_with_gemini(best_answer)
        
        # Store conversation in history
        st.session_state.chat_history.append(f"**You:** {user_input}")
        st.session_state.chat_history.append(f"**Chatbot:** {refined_answer}")

# Display chat history
for message in st.session_state.chat_history:
    st.write(message)
