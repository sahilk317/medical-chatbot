from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


class Helper:

    def __init__(self):
        pass

    def load_pdf_files(self,data):
        loader = DirectoryLoader(
            path=data,
            glob='*.pdf',
            loader_cls=PyPDFLoader    
        )
        
        documents = loader.load()
        return documents
    
    def filter_to_minimal_docs(self,docs):
        minimal_docs = []
        for doc in docs:
            src = doc.metadata.get('source')
            minimal_docs.append(
                Document(
                    page_content=doc.page_content,
                    metadata = {'source' : src}
                )
            )
        return minimal_docs
    
    def split_text(self,docs):
        splitters = RecursiveCharacterTextSplitter(
            chunk_size = 500,
            chunk_overlap = 20
        )
        splitted_text = splitters.split_documents(docs)
        return splitted_text
    
    def download_embeddings(self):
        model_name = 'all-MiniLM-L6-v2'
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"}
        )

        return embeddings