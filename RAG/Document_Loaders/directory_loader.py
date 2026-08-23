from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader


# In Directoryloader wen can import any type of file it could be pdf, text, or anything jstt change the sign



loader = DirectoryLoader(
    path='books',
    glob='*.pdf',
    loader_cls= PyPDFLoader
)

docs = loader.lazy_load()

## we use lazy_load() instead of load() cuz it's fast and optiomal lazy_laod() does load one by one and otherhand laod() function load at once that's why it takes alot of time... 

for document in docs:
    print(document.metadata)


# print(docs[0].page_content)
# print(docs[1].metadata)
