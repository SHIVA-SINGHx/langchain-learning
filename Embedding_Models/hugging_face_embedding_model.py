from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# text = "What is the capital of UK?"

documents = [
    "What is capital of UK?",
    "What is capital of France?",
    "What is capital of Germany?",
]

# vector = embeddings.embed_query(text) ## this is for single query
vectors = embeddings.embed_documents(documents)  ## this is for multiple documents

print(str(vectors))