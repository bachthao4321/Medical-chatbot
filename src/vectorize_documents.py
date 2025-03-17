from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


#Load raw PDF
DATA_PATH = "./data/"

def load_pdf_files(data):
    loader = DirectoryLoader(data,                  
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

documents = load_pdf_files(data=DATA_PATH)
# print("Length of PDF pages: ", len(documents))


# Creare Chunks
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    text_chunks = text_splitter.split_documents(extracted_data)  
    return text_chunks     

text_chunks = create_chunks(extracted_data=documents)
# print("Length of Text Chunks: ", len(text_chunks))  


# Create Vector Embeddings
def get_embedding_model():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

embedding_model = get_embedding_model()

# Store embeddings in FAISS                        
vectorstore = FAISS.from_documents(text_chunks, embedding_model)
vectorstore.save_local('./faiss_medical')                  
                            