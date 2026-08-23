from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda


## RunnableLambda here we can pass lambda function or normal function into RunnableLambda

load_dotenv()


def word_counter(text):
    return len(text.split())



prompt1 = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=['topic']
)


model2 = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.8) 


parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt1, model2, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'words_count': RunnableLambda(word_counter)
})


final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = final_chain.invoke({'topic': 'AI'})



print(result['joke'])
print(result['words_count'])