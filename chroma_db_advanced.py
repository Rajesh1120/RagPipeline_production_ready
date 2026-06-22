from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
import tempfile # used for creating temp files and directories and automatically cleanup after program terminate
from dotenv import load_dotenv

load_dotenv()

embeddings_model=OpenAIEmbeddings(model="text-embedding-3-small")
SAMPLE_DOCS = [
    Document(
        page_content="LangChain is a framework for developing applications powered by language models.",
        metadata={"source": "langchain_docs", "topic": "overview"},
    ),
    Document(
        page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs.",
        metadata={"source": "langgraph_docs", "topic": "overview"},
    ),
    Document(
        page_content="Vector stores are databases optimized for storing and searching embeddings.",
        metadata={"source": "vector_guide", "topic": "database"},
    ),
    Document(
        page_content="RAG combines retrieval with generation for more accurate LLM responses.",
        metadata={"source": "rag_guide", "topic": "architecture"},
    ),
    Document(
        page_content="Embeddings convert text into numerical vectors for semantic similarity.",
        metadata={"source": "embeddings_guide", "topic": "fundamentals"},
    ),
    Document(
        page_content="Chroma is an open-source embedding database for AI applications.",
        metadata={"source": "chroma_docs", "topic": "database"},
    ),
    Document(
        page_content="FAISS is a library for efficient similarity search developed by Facebook.",
        metadata={"source": "faiss_docs", "topic": "database"},
    ),
    Document(
        page_content="Pinecone is a managed vector database service for production workloads.",
        metadata={"source": "pinecone_docs", "topic": "database"},
    ),
]

def chroma_basics():
    with tempfile.TemporaryDirectory() as tempdir:
        #create vector store from documents(modern way to use it.
        # it will 3 steps in one 1. creating embeddings, 2. create vectorDB, 3. insert documents)

        vector_store=Chroma.from_documents(
            documents=SAMPLE_DOCS,
            embedding=embeddings_model,
            persist_directory=tempdir # store the documents and embeddings in the dir
        )

        print(
            f"Vector store created {vector_store._collection.count()} documents and persisted."
        )

        # perform similarity search
        query= "what is LangChain?"
        results=vector_store.similarity_search(query, k=2)
        print(f"Top 2 results for query '{query}':")
        for i, doc in enumerate(results):
            print(
                    f"Result {i+1}: {doc.page_content} (Source: {doc.metadata['source']})"
                )
            
def similarity_search_with_scores():
    with tempfile.TemporaryDirectory() as tempdir:
        #create vector from documents
        vector_store= Chroma.from_documents(
            documents=SAMPLE_DOCS,
            embedding=embeddings_model,
            persist_directory=tempdir
        )
         # perform similarity search with scores
        query = "Explain vector stores."
        results_with_scores = vector_store.similarity_search_with_score(query, k=2)

        print(f"Top 2 results with scores for query '{query}':")

        for i, (doc, score) in enumerate(results_with_scores):
            final_score = 1 / (1 + score)  # Convert distance to similarity
            print(
                f"Result {i+1}: {doc.page_content} (Score: {final_score:.4f}, Source: {doc.metadata['source']})"
            )

def metadata_filtering():
    with tempfile.TemporaryDirectory() as tmpdir:
        # create vector store from documents
        vectorstore = Chroma.from_documents(
            documents=SAMPLE_DOCS, embedding=embeddings_model, persist_directory=tmpdir
        )

        query = "What databases are available?"

        # without metadata filtering
        results = vectorstore.similarity_search(query, k=5)
        print(f"Results without metadata filtering for query '{query}':")
        for i, doc in enumerate(results):
            print(
                f"Result {i+1}: {doc.page_content} (Source: {doc.metadata['source']})"
            )

        # with metadata filtering
        filter_criteria = {"topic": "database"}
        filtered_results = vectorstore.similarity_search(
            query, k=5, filter=filter_criteria
        )
        print(f"\nResults with metadata filtering for query '{query}':")
        for i, doc in enumerate(filtered_results):
            print(
                f"Result {i+1}: {doc.page_content} (Source: {doc.metadata['source']})"
            )




if __name__=="__main__":
    # chroma_basics()
    # similarity_search_with_scores()
    metadata_filtering()