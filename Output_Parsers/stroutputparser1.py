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

#template-1 
template1 = PromptTemplate(
    template = "write a detailed report on {topic}",
    input_variables= ['topic']
)

# tempalte-2
template2 = PromptTemplate(
    template = "Write a 5 line of summary on a following text. /n {text}",
    input_variables= ['text']
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic': "Franz Kafka"})
print(result)