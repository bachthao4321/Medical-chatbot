from langchain.chains.retrieval_qa.base import RetrievalQA 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from src.load_llm import llm
from src.prompt_template import prompt
from flask import Flask, render_template, jsonify, request

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


app = Flask(__name__)
# Connect LLM with FAISS and Create chain
DB_FAISS_PATH = "faiss_medical"

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm= llm,
    chain_type="stuff",
    retriever= db.as_retriever(search_kwargs={'k':3}),
    return_source_documents = True,
    chain_type_kwargs={'prompt':prompt}
)

# Invoke with a single query
# user_query = input("Write Query Here: ")
# response = qa_chain.invoke({"query":user_query})

# print("RESULT: ", response["result"])
# print("SOURCE DOCUMENTS: ", response["source_documents"])



@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get("message", "").strip()
        if not user_message:
            return jsonify({"response": "Xin vui lòng nhập câu hỏi."})

        # Gọi chatbot để lấy phản hồi
        response = qa_chain.invoke({"query": user_message})

        # Lấy kết quả từ phản hồi
        bot_reply = response.get("result", "Xin lỗi, tôi không hiểu câu hỏi của bạn.")

        return jsonify({"response": bot_reply})
    
    except Exception as e:
        return jsonify({"response": f"Lỗi: {str(e)}"})


