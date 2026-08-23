from langchain_community.document_loaders import WebBaseLoader
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


model1 = ChatHuggingFace(llm=llm) # model-1


prompt = PromptTemplate(
    template='Answer the following question \n {question} from the following text - \n {text} ',
    input_variables=['question', 'text']
)

parser = StrOutputParser()

url = 'https://www.flipkart.com/apple-iphone-17-black-256-gb/p/itm6eb39da622cdd'

loader = WebBaseLoader(url)

docs = loader.load()

chain = prompt | model1 | parser



print(chain.invoke({'question': 'What is the color of that product?', 'text':docs[0].page_content}))

# print(len(docs))
# print(docs[0].page_content)


