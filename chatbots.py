import streamlit as st
from openai import OpenAI  # Updated OpenAI import
import anthropic
import google.generativeai as genai
import requests
import time

# Set up API keys
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
claude = anthropic.Client(api_key=st.secrets["ANTHROPIC_API_KEY"])
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]

# Initialize models
gemini_model = genai.GenerativeModel('gemini-pro')
openai_model = "gpt-3.5-turbo"

# Streamlit app
st.title("Multi-Chatbot Comparison App")
st.write("Enter your search query below to compare results from different chatbot models.")

# User input
user_input = st.text_input("Enter your search query:")

if user_input:
    st.write(f"**Your Query:** {user_input}")
    st.write("**Results:**")

    # Function to call OpenAI (updated for OpenAI API v1.0.0)
    def call_openai(query):
        response = openai_client.chat.completions.create(
            model=openai_model,
            messages=[{"role": "user", "content": query}]
        )
        return response.choices[0].message.content

    # Function to call Claude (updated for latest Anthropic API)
    def call_claude(query):
        message = claude.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            messages=[
                {
                    "role": "user",
                    "content": query
                }
            ]
        )
        return message.content[0].text

    # Function to call Gemini
    def call_gemini(query):
        response = gemini_model.generate_content(query)
        return response.text

    # Function to call DeepSeek API (updated with correct endpoint)
    def call_deepseek(query):
        url = "https://api.deepseek.com/v1/chat/completions"  # Corrected endpoint
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
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            return f"Error calling DeepSeek API: {str(e)}"

    # Get responses from all models with error handling and timing
    responses = {}
    response_times = {}
    
    with st.spinner('Getting responses from all models...'):
        # OpenAI
        try:
            start_time = time.time()
            responses['OpenAI'] = call_openai(user_input)
            response_times['OpenAI'] = time.time() - start_time
        except Exception as e:
            responses['OpenAI'] = f"Error with OpenAI: {str(e)}"
            response_times['OpenAI'] = 0
        
        # Claude
        try:
            start_time = time.time()
            responses['Claude'] = call_claude(user_input)
            response_times['Claude'] = time.time() - start_time
        except Exception as e:
            responses['Claude'] = f"Error with Claude: {str(e)}"
            response_times['Claude'] = 0
        
        # Gemini
        try:
            start_time = time.time()
            responses['Gemini'] = call_gemini(user_input)
            response_times['Gemini'] = time.time() - start_time
        except Exception as e:
            responses['Gemini'] = f"Error with Gemini: {str(e)}"
            response_times['Gemini'] = 0
        
        # DeepSeek
        try:
            start_time = time.time()
            responses['DeepSeek'] = call_deepseek(user_input)
            response_times['DeepSeek'] = time.time() - start_time
        except Exception as e:
            responses['DeepSeek'] = f"Error with DeepSeek: {str(e)}"
            response_times['DeepSeek'] = 0

    # Display results in a single row with borders
    st.subheader("Comparison of Responses")
    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])  # Each column takes up a width of 2

    with col1:
        st.markdown("<div style='border: 1px solid black; padding: 10px;'>", unsafe_allow_html=True)
        st.write("**OpenAI (GPT-4):**")
        st.write(responses['OpenAI'])
        st.write(f"Response time: {response_times['OpenAI']:.2f}s")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div style='border: 1px solid black; padding: 10px;'>", unsafe_allow_html=True)
        st.write("**Claude (Anthropic):**")
        st.write(responses['Claude'])
        st.write(f"Response time: {response_times['Claude']:.2f}s")
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div style='border: 1px solid black; padding: 10px;'>", unsafe_allow_html=True)
        st.write("**Google Gemini:**")
        st.write(responses['Gemini'])
        st.write(f"Response time: {response_times['Gemini']:.2f}s")
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        st.markdown("<div style='border: 1px solid black; padding: 10px;'>", unsafe_allow_html=True)
        st.write("**DeepSeek:**")
        st.write(responses['DeepSeek'])
        st.write(f"Response time: {response_times['DeepSeek']:.2f}s")
        st.markdown("</div>", unsafe_allow_html=True)

    # Add a comparison table
    st.write("**Comparison Metrics:**")
    comparison_data = {
        "Model": list(responses.keys()),
        "Response Length": [len(str(response)) for response in responses.values()],
        "Response Time (s)": [f"{time:.2f}" for time in response_times.values()]
    }
    st.table(comparison_data)