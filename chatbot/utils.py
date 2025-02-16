import google.generativeai as genai

def generate_embedding(text):
    """Generate embeddings using Gemini AI."""
    try:
        response = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        return response["embedding"]
    except Exception as e:
        print("Error generating embedding:", str(e))
        return None

def refine_with_gemini(context):
    """Refine AI response using Gemini AI."""
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(f"Refine and simplify this response: {context}")
    return response.text if response.text else context
