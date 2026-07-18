from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="conversational",
    huggingfacehub_api_token=os.getenv(
        "HUGGINGFACEHUB_API_TOKEN"
    )
)

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name: str = Field(description='Name of the person')
    age: int = Field(gt=19, description= 'Age of the person')
    city: str = Field(description="City of the person")
    
    
parser = PydanticOutputParser(pydantic_object=Person)


template = PromptTemplate(
    template= 'Generate the name, age and city of a fictional {place} person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'place': 'Delhi'})

print(result)
