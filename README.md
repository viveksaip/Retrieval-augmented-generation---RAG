# Retrieval-augmented-generation---RAG

## How It Works
Initialization:
* Use SentenceTransformers to encode documents into dense vectors.
* Use FAISS to manage a vector index for retrieval.

## Dynamic Updates:
New documents are encoded and added to the FAISS index.
The original list of documents is updated for reference.

## Inference:
A query retrieves relevant documents from the FAISS index.
The generator uses the retrieved documents as context to generate a response.


## Output Example
### Response before adding new knowledge:
FAISS is a library for efficient similarity search and clustering of dense vectors.

### Response after adding new knowledge:
FAISS supports both CPU and GPU for vector similarity search. FAISS is a library for efficient similarity search.
