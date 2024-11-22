from flask import Flask, request, jsonify
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM  # Remplacez par la nouvelle classe

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader


app = Flask(__name__)

# Embeddings Ollama
embeddings = OllamaEmbeddings(base_url="http://localhost:7869", model='nomic-embed-text:latest')

llm = OllamaLLM(
    base_url="http://localhost:7869",
    model="llama3.2:1b",
    temperature=0.7,                  
    max_tokens=2048,                   
    top_p=0.9,                         
    frequency_penalty=0.2,           
    presence_penalty=0.6, 
)

def load_and_split_documents(base_dir: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    docs = []  
    for filename in os.listdir(base_dir):
        file_path = os.path.join(base_dir, filename)
        
        if filename.endswith('.txt'):
            loader = TextLoader(file_path)
            docs_temp = loader.load()
            for doc in docs_temp:
                doc.metadata['source'] = filename
            docs.extend(docs_temp)

        elif filename.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            docs_temp = loader.load()
            for doc in docs_temp:
                doc.metadata['source'] = filename
            docs.extend(docs_temp)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = text_splitter.split_documents(docs)
    
    return splits

base_directory = './rag/src/base'
splits = load_and_split_documents(base_directory)

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="chroma_db"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)


def run_rag(question: str):
    print('chaine')
    rag_chain = (
        {"context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
         "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    return rag_chain.invoke(question)

@app.route('/rag', methods=['POST'])
def rag_api():
    data = request.json
    question = data.get("question")
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    try:
        print('run')
        answer = run_rag(question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
