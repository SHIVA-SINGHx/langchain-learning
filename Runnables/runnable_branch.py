from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch


## RunnableBranch is basically conditinal chains you know if else 

load_dotenv()



prompt1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template= "Summarize the following text \n {text}",
    input_variables= ['text']
)


model2 = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.8) 


parser = StrOutputParser()

report_gen_chain = RunnableSequence(prompt1, model2, parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split())>200, RunnableSequence(prompt2, model2, parser)), # if condition
    RunnablePassthrough()   # defualt condition
)

final_chain = RunnableSequence(report_gen_chain, branch_chain)

result = final_chain.invoke({'topic': 'Cloud Computing'})

print(result)