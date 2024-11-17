#pip install faiss-cpu sentence-transformers transformers

# Required libraries
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer

# Step 1: Initialize components
# Load a SentenceTransformer for dense embeddings
retriever_model = SentenceTransformer("all-MiniLM-L6-v2")

# Create a FAISS index for dense vector retrieval
embedding_dim = retriever_model.get_sentence_embedding_dimension()
index = faiss.IndexFlatL2(embedding_dim)

# Initialize a GPT-based model (e.g., GPT-2 or GPT-3, but here we use GPT-2)
generator = pipeline("text-generation", model="gpt2", tokenizer="gpt2")

# Example knowledge base (documents)
documents = [
    "Python is a popular programming language for machine learning.",
    "FAISS is a library for efficient similarity search and clustering of dense vectors.",
    "Transformers are pre-trained models used in natural language processing tasks."
]

# Step 2: Index initial documents
def index_documents(documents, index, retriever_model):
    embeddings = retriever_model.encode(documents)
    index.add(embeddings)
    return embeddings

# Index the initial documents
index_documents(documents, index, retriever_model)

# Step 3: Search the knowledge base
def retrieve_documents(query, index, retriever_model, top_k=2):
    query_embedding = retriever_model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    return [documents[i] for i in indices[0]]

# Step 4: Add new knowledge dynamically
def add_document(new_document, index, retriever_model):
    new_embedding = retriever_model.encode([new_document])
    index.add(new_embedding)
    documents.append(new_document)

# Step 5: Generate responses using GPT and retrieved documents
def generate_response(query):
    # Retrieve relevant documents based on the query
    retrieved_docs = retrieve_documents(query, index, retriever_model)
    
    if retrieved_docs:
        # If there are relevant documents, use them as context for GPT
        context = " ".join(retrieved_docs)
        prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    else:
        # If no relevant documents, use GPT's generalization
        prompt = f"Question: {query}\nAnswer:"
    
    # Generate a response from GPT
    response = generator(prompt, max_length=100, num_return_sequences=1)[0]["generated_text"]
    return response

# Example Usage
query = "What is FAISS?"
print("Response before adding new knowledge:")
print(generate_response(query))

# Add new knowledge to the system
add_document("FAISS supports both CPU and GPU for vector similarity search.", index, retriever_model)

print("\nResponse after adding new knowledge:")
print(generate_response(query))
