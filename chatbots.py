import streamlit as st
import openai
import anthropic
import google.generativeai as genai
import requests

# Set up API keys
openai.api_key = st.secrets["OPENAI_API_KEY"]
anthropic.api_key = st.secrets["ANTHROPIC_API_KEY"]
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]

# Initialize models
gemini_model = genai.GenerativeModel('gemini-pro')
claude_model = anthropic.Client()
openai_model = "gpt-4"  # or "gpt-3.5-turbo"

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

    # Function to call DeepSeek API
    def call_deepseek(query):
        url = "https://api.deepseek.ai/v1/chat/completions"  # Updated API endpoint
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            return f"Error calling DeepSeek API: {str(e)}"

    # Get responses from all models with error handling
    responses = {}
    
    with st.spinner('Getting responses from all models...'):
        # OpenAI
        try:
            responses['OpenAI'] = call_openai(user_input)
        except Exception as e:
            responses['OpenAI'] = f"Error with OpenAI: {str(e)}"
        
        # Claude
        try:
            responses['Claude'] = call_claude(user_input)
        except Exception as e:
            responses['Claude'] = f"Error with Claude: {str(e)}"
        
        # Gemini
        try:
            responses['Gemini'] = call_gemini(user_input)
        except Exception as e:
            responses['Gemini'] = f"Error with Gemini: {str(e)}"
        
        # DeepSeek
        try:
            responses['DeepSeek'] = call_deepseek(user_input)
        except Exception as e:
            responses['DeepSeek'] = f"Error with DeepSeek: {str(e)}"

    # Display results in columns
    col1, col2 = st.columns(2)
    with col1:
        st.write("**OpenAI (GPT-4):**")
        st.write(responses['OpenAI'])
    with col2:
        st.write("**Claude (Anthropic):**")
        st.write(responses['Claude'])

    col3, col4 = st.columns(2)
    with col3:
        st.write("**Google Gemini:**")
        st.write(responses['Gemini'])
    with col4:
        st.write("**DeepSeek:**")
        st.write(responses['DeepSeek'])

    # Add a comparison table
    st.write("**Comparison Table:**")
    comparison_data = {
        "Model": list(responses.keys()),
        "Response": list(responses.values())
    }
    st.table(comparison_data)

    # Add response lengths comparison
    st.write("**Response Lengths (characters):**")
    lengths_data = {
        "Model": list(responses.keys()),
        "Length": [len(str(response)) for response in responses.values()]
    }
    st.table(lengths_data)