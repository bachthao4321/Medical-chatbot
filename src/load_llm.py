import os
from langchain_huggingface import HuggingFaceEndpoint

# Setup LLM ( Mistral with HuggingFace)
HF_TOKEN = os.environ.get('HF_TOKEN')
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token":HF_TOKEN,
                      "max_length":"256"}
    )
    return llm

llm = load_llm(HUGGINGFACE_REPO_ID)

