from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

loader = PyPDFLoader('dl-curriculum.pdf')

docs = loader.load()

text = """
Performance Bottlenecks: Can become a bottleneck or single point of failure under high traffic if not properly designed.
Increased Latency: Additional processing like routing, authentication, and transformations can slow down requests.
Complexity: Managing and configuring the gateway becomes difficult with many services and endpoints.
Security Risks: Misconfiguration can lead to vulnerabilities like improper authentication or data exposure.
Scalability Challenges: Scaling the gateway efficiently in dynamic, high-demand environments can be challenging.
"""


spiltter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0,
    separator='' 
)

# result= spiltter.split_text(text)
result2= spiltter.split_documents(docs)


# print(result)
print(result2[0].page_content)


