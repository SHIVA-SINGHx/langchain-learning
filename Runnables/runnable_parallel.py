from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableSequence, RunnableParallel



load_dotenv()

prompt1 = PromptTemplate(
    template="Generate a X post about {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template= " Generate a linkedin post about {topic}",
    input_variables= ['topic']
)

model2 = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.8) 

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'X': RunnableSequence(prompt1, model2, parser),
    'Linkedin': RunnableSequence(prompt2, model2, parser)
}
)

result = parallel_chain.invoke({'topic': "AI"})

print(result['X'])
print(result['Linkedin'])