import streamlit as st
import openai
import anthropic
import google.generativeai as genai
import deepseek

# Set up API keys
openai.api_key = st.secrets["OPENAI_API_KEY"]
anthropic.api_key = st.secrets["ANTHROPIC_API_KEY"]
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
deepseek.api_key = st.secrets["DEEPSEEK_API_KEY"]

# Initialize models
gemini_model = genai.GenerativeModel('gemini-pro')
claude_model = anthropic.Client()
openai_model = "gpt-4"  # or "gpt-3.5-turbo"
deepseek_model = deepseek.Client()

# Streamlit app
st.title("Multi-Chatbot Comparison App")
st.write("Enter your search query below to compare results from different chatbot models.")

# User input
user_input = st.text_input("Enter your search query:")

if user_input:
    st.write(f"**Your Query:** {user_input}")
    st.write("**Results:**")

    # Function to call OpenAI
    def call_openai(query):
        response = openai.ChatCompletion.create(
            model=openai_model,
            messages=[{"role": "user", "content": query}]
        )
        return response['choices'][0]['message']['content']

    # Function to call Claude
    def call_claude(query):
        response = claude_model.completion(
            prompt=f"\n\nHuman: {query}\n\nAssistant:",
            max_tokens_to_sample=1000
        )
        return response['completion']

    # Function to call Gemini
    def call_gemini(query):
        response = gemini_model.generate_content(query)
        return response.text

    # Function to call DeepSeek
    def call_deepseek(query):
        response = deepseek_model.generate(query)
        return response['choices'][0]['text']

    # Get responses from all models
    openai_response = call_openai(user_input)
    claude_response = call_claude(user_input)
    gemini_response = call_gemini(user_input)
    deepseek_response = call_deepseek(user_input)

    # Display results in columns
    col1, col2 = st.columns(2)
    with col1:
        st.write("**OpenAI (GPT-4):**")
        st.write(openai_response)
    with col2:
        st.write("**Claude (Anthropic):**")
        st.write(claude_response)

    col3, col4 = st.columns(2)
    with col3:
        st.write("**Google Gemini:**")
        st.write(gemini_response)
    with col4:
        st.write("**DeepSeek:**")
        st.write(deepseek_response)

    # Optional: Add a comparison section
    st.write("**Comparison:**")
    comparison_data = {
        "Model": ["OpenAI", "Claude", "Gemini", "DeepSeek"],
        "Response": [openai_response, claude_response, gemini_response, deepseek_response]
    }
    st.table(comparison_data)