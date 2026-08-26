from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
Performance Bottlenecks: Can become a bottleneck or single point of failure under high traffic if not properly designed.
Increased Latency: Additional processing like routing, authentication, and transformations can slow down requests.
Complexity: Managing and configuring the gateway becomes difficult with many services and endpoints.
Security Risks: Misconfiguration can lead to vulnerabilities like improper authentication or data exposure.
Scalability Challenges: Scaling the gateway efficiently in dynamic, high-demand environments can be challenging.
"""

splliter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 0
)

result = splliter.split_text(text)


print(len(result))
print(result)
