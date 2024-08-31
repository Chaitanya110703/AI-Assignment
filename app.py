from flask import Flask, request, jsonify
import os
from langchain_cohere import CohereEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.vectorstores import InMemoryVectorStore

app = Flask(__name__)


if not os.getenv("COHERE_API_KEY"):
    os.environ["COHERE_API_KEY"] = "********************************"


os.environ['USER_AGENT'] = 'myagent'


url = "https://brainlox.com/courses/category/technical"
loader = WebBaseLoader(url)
documents = loader.load()




embeddings = CohereEmbeddings(
    model="embed-english-v3.0",
)


vectorstore = InMemoryVectorStore.from_documents(
    documents,
    embedding=embeddings,
)

# Create a retriever
retriever = vectorstore.as_retriever()

@app.route('/query', methods=['POST'])
def query_vectorstore():
   
    query = request.json.get('query')
    
    
    retrieved_documents = retriever.get_relevant_documents(query)
    
    
    if not retrieved_documents:
        return jsonify({"message": "No relevant documents found."}), 404
    
   
    content = retrieved_documents[0].page_content
    return jsonify({"query": query, "response": content})

if __name__ == '__main__':
    app.run(debug=True)
