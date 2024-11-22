import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import bs4
import boto3
from io import BytesIO
from io import StringIO 


# prompt
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)


# Model LLM 
llm = Ollama(
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

def download_file_from_s3(bucket_name: str, file_key: str):
    try:
        s3_client = boto3.client(
            's3', 
            region_name='eu-west-3' 
        )
        file_obj = BytesIO()
        s3_client.download_fileobj(bucket_name, file_key, file_obj)
        file_obj.seek(0) 
        return file_obj
    except Exception as e:
        return None
    

def load_and_split_documents_from_s3(s3_bucket: str, s3_prefix: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    s3_client = boto3.client(
        's3', 
        region_name='eu-west-3' 
    )
    docs = []

    try:
        objects = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=s3_prefix)
        if 'Contents' not in objects:
            return []

        for obj in objects['Contents']:
            file_key = obj['Key']
            file_obj = download_file_from_s3(s3_bucket, file_key)

            if file_obj is None:
                continue 
            if file_key.endswith('.txt'):
                file_content = file_obj.getvalue().decode('utf-8')  
                string_io_file = StringIO(file_content)
                loader = TextLoader(string_io_file)
                docs_temp = loader.load()
                for doc in docs_temp:
                    doc.metadata['source'] = file_key
                docs.extend(docs_temp)

            elif file_key.endswith('.pdf'):
                loader = PyPDFLoader(file_obj)
                docs_temp = loader.load()
                for doc in docs_temp:
                    doc.metadata['source'] = file_key
                docs.extend(docs_temp)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        splits = text_splitter.split_documents(docs)
        return splits

    except Exception as e:
        return []

# Configuration S3
s3_bucket_name = 'aws-s3-test-rag'  # Remplacez par votre bucket S3
s3_prefix = 'base/'   # Remplacez par le préfixe approprié

# Charger et diviser les documents depuis S3
# splits = load_and_split_documents_from_s3(s3_bucket=s3_bucket_name, s3_prefix=s3_prefix)




# Embed
print("Configuration des embeddings avec nomic-embed-text...")
embeddings = OllamaEmbeddings(
    base_url="http://localhost:7869",
    model='nomic-embed-text:latest'
)

print("Création du Vectorstore avec nomic-embed-text...")
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="chroma_db"  # Optionnel: pour sauvegarder la base de données
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

#### RETRIEVAL and GENERATION ####
# Prompt
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)



# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Poser une question
question = "What are some variations of strawberry cakes, and how can they be customized for different occasions?"
print(f"Posing question: {question}")
answer = rag_chain.invoke(question)

# Afficher la réponse
print("\n### Réponse ###")
print(answer)