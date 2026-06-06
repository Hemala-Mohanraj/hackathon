from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "vector_db/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

def search_document(query):

    docs = db.similarity_search(query, k=2)

    context = ""

    for doc in docs:
        context += doc.page_content + "\n"

    return context