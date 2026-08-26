from langchain_community.document_loaders import CSVLoader

loader = CSVLoader('free.csv')

docs = loader.load()


print(len(docs))
print(docs[2])