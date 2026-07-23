from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="conversational",
    huggingfacehub_api_token=os.getenv(
        "HUGGINGFACEHUB_API_TOKEN"
    )
)



model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template='Give me top 5 best player of {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'topic': 'football'})

print(result)
