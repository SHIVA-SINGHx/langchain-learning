from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader


load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="conversational",
    huggingfacehub_api_token=os.getenv(
        "HUGGINGFACEHUB_API_TOKEN"
    )
)


model1 = ChatHuggingFace(llm=llm) # model-1


prompt = PromptTemplate(
    template='Write a summary for the following poem -\n {poem} ',
    input_variables=['poem']
)

parser = StrOutputParser()


loader = TextLoader('cricket.txt', encoding='utf-8')

docs = loader.load()



print(type(docs))
print(len(docs))

print(docs[0].page_content)
print(docs[0].metadata)

chain = prompt | model1 | parser

print(chain.invoke({'poem': docs[0].page_content}))

