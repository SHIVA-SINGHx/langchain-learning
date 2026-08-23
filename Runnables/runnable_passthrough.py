from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough


load_dotenv()



prompt1 = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template= "Explain the following joke - {text}",
    input_variables= ['text']
)

model2 = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.8) 


parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt1, model2, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(prompt1, model2, parser)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = final_chain.invoke({'topic': 'football'})

print(result)

