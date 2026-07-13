from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
import streamlit as st
from langchain_core.prompts import PromptTemplate,load_prompt

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="conversational",
    huggingfacehub_api_token=os.getenv(
        "HUGGINGFACEHUB_API_TOKEN"
    )
)

model = ChatHuggingFace(llm=llm)


st.header("Game Explanation Generator")

game_input = st.selectbox( "Select Game Name", ["Chess", "Ludo", "Poker", "Dota 2", "Counter-Strike", "Clash Royale"] )

game_style = st.selectbox("Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] )

game_length = st.selectbox("Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )


template = load_prompt('template_game.json')


if st.button('Summarize'):
    chain = template | model
    result = chain.invoke({
        'game_input':game_input,
        'game_style':game_style,
        'game_length':game_length
    })
    st.write(result.content)