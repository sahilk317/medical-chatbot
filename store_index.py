import os 
from dotenv import load_dotenv
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.helper import Helper
from src.prompt import system_prompt



class Store_Index:

    def __init__(self):
        load_dotenv()
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        # if not self.pinecone_api_key:
            # raise ValueError("PINECONE_API_KEY not found in environment.")
        self.helper = Helper()
        self.embeddings = self.helper.download_embeddings()

    def Create_Index(self, index_name):
        pc = Pinecone(api_key=self.pinecone_api_key)

        if not pc.has_index(index_name):

            self.documents = self.helper.load_pdf_files('./data')
            self.minimal_docs = self.helper.filter_to_minimal_docs(self.documents)
            self.splitted_text = self.helper.split_text(self.minimal_docs)

            pc.create_index(
                name=index_name,
                dimension=384,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )

        # Only embed and upload if index is new
            docsearch = PineconeVectorStore.from_documents(
                documents=self.splitted_text,
                embedding=self.embeddings,
                index_name=index_name
            )
        
            
        return docsearch


    def list_indexes(self):
        pc = Pinecone(api_key=self.pinecone_api_key)
        return [index.name for index in pc.list_indexes()]
    
    def Load_Index(self, index_name):
        return PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=self.embeddings
    )