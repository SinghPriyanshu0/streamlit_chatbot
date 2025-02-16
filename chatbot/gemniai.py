import google.generativeai as genai

def chat_with_gemini(user_input):
    """Generates a response using Google Gemini AI."""
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(user_input)
    return response.text

