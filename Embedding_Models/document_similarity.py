from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity


load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "Lionel Messi is a Goat..",
    "Pelé, also known as the 'God of Football', holds many goal-scoring records..",
    "Kylian Mbappé is known for his elegant dribbling and record-breaking hat-tricks.",
    "Jude Bellingham is an England midfielder famous for his calm demeanor and finishing skills.",
    "Erling Haaland is a Norwegian striker known for his unorthodox movement and power."
]

query = "Who is Lionel Messi?"

query_vector = embeddings.embed_query(query)  # this is for single query
vectors = embeddings.embed_documents(documents) # this is for multiple documents


similarities = cosine_similarity([query_vector], vectors) # here we are comparing the query vector with all the document vectors to get similarity scores

index, score = sorted(list(enumerate(similarities[0])), key=lambda x: x[1], reverse=True)[0]  # getting the index and score of the most similar document to the query

print(query)
print(documents[index])
print("similarity score:", score)