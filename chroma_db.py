import chromadb 

#create a chroma client

chroma_client = chromadb.Client()

#create a collection .

#collections are where you will store your embeddings, documents, 
#and also any additional metadata, so collections index your embeddings or documents 
#and enable efficient retrieval and also filtering

collection= chroma_client.create_collection(name="my_collection")


# Adding some test documents to the collection

# So here, what is happening is the chroma will store your text and handle embedding and indexing automatically. 
# You can also optimize the embedding model.
#  You must provide unique string IDs for your documents. 


# use the  collection.upsert(); it don't add multiple times when every you run the program
collection.add(
    ids=['id1','id2'],
    documents=[
        "This is a document about pineapple",
        "This is a document about oranges"
    ]
)


# Query the collection 

#You can query the collections with a list of query tests, 
# and chroma will return the most similar requests.


results= collection.query(
        query_texts=["This is a query document about hawaii"], # Chroma will embed this for you
        n_results=2 # how many results to return
)

print(results)