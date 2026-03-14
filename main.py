import os
import wikipedia
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def sync_wikipedia():
    print('Syncing Wikipedia data.')
    try:
        page = wikipedia.page("Cristiano Ronaldo", auto_suggest=False)
        with open(f"{DATA_DIR}/wikipedia_cr7.txt", "w", encoding="utf-8") as f:
            f.write(page.content)
    except Exception as e:
        print(f"Wikipedia sync skipped: {e}")

sync_wikipedia()

print("Loading knowledge base.")

loaders = {
    ".txt": lambda path: TextLoader(path, encoding="utf-8", autodetect_encoding=True),
    ".pdf": PyPDFLoader
}

docs = []
for file in os.listdir(DATA_DIR):
    file_path = os.path.join(DATA_DIR, file)
    ext = os.path.splitext(file)[1].lower()
    if ext in loaders:
        try:
            loader_func = loaders[ext]
            loader = loader_func(file_path)
            docs.extend(loader.load())
        except Exception as e:
            print(f"Failed to load {file}: {e}")

if not docs:
    print("No documents were loaded. Check your 'data' folder.")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250, separators=["\n\n", "\n", ". ", " ", ""])
final_docs = text_splitter.split_documents(docs)

embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="llama3.2:3b", temperature=0.1)

print("Building the vector engine.")
vectorstore = FAISS.from_documents(final_docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

system_prompt = (
    "You are a CR7 expert. Answer the question based ONLY on the context.\n"
    "If the answer is not there in the context, say I don't know."
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

print("\n CR7 Bot is Ready! (Type 'exit' to stop)")
while True:
    query = input("Ask about CR7: ")
    if query.lower() in ['exit', 'quit']: break
    
    response = rag_chain.invoke({"input": query})
    print(f"\n[Bot]: {response['answer']}\n")