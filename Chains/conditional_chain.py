from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="conversational",
    huggingfacehub_api_token=os.getenv(
        "HUGGINGFACEHUB_API_TOKEN"
    )
)

model1 = ChatHuggingFace(llm=llm)
model2 = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.8)

parser1 = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')
    
    
parser2 = PydanticOutputParser(pydantic_object=Feedback)


prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)

classfier_chain = prompt1 | model1 | parser2 



prompt2 = PromptTemplate(
    template='write an appropriate response to this positive feedaback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

# RUNNABLE BRANCH 
branch_main = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model1 | parser1),
    (lambda x:x.sentiment == 'negative', prompt3 | model1 | parser1),
    RunnableLambda(lambda x: "Invalid sentiment")
    
)

chain = classfier_chain | branch_main

print(chain.invoke({'feedback': 'This is a terrible phone'}))