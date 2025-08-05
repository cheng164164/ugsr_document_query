import os
import logging
import concurrent.futures
import azure.functions as func
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from azure.storage.blob import ContainerClient
from openai import AzureOpenAI
from .config import INDEX_CONFIGS, ENV_VARS, SCHEMA_MAPPING_DICT
from .util import set_env_vars, clean_metadata
from .indexer import *


logging.basicConfig(level=logging.INFO)
executor = concurrent.futures.ThreadPoolExecutor()


def main(req: func.HttpRequest) -> func.HttpResponse:

    logging.info("HTTP trigger received: launching indexing job in background...")
    # Launch index job in background thread
    executor.submit(run_index_job)

    return func.HttpResponse("Indexing started.", status_code=202)


def run_index_job():   
    try:
        # Set environment variables from config.py
        set_env_vars(ENV_VARS=ENV_VARS)

        connection_string = os.getenv('AZURE_BLOB_CONN_STRING')
        azure_doc_intell_endpoint = os.getenv('AZURE_DOC_INTELL_ENDPOINT')
        azure_doc_intell_key = os.getenv('AZURE_DOC_INTELL_KEY')
        azure_oai_deployment_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        azure_oai_key = os.getenv('AZURE_OPENAI_KEY')
        azure_oai_deployment_model = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
        search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        search_key = os.getenv("AZURE_SEARCH_KEY")
        azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        azure_oai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        azure_oai_embedding_deployment = os.getenv('AZURE_EMBEDDING_DEPLOYMENT_NAME', 'text-embedding-3-small')

        # print(f"AZURE_BLOB_CONN_STRING: {os.getenv('AZURE_BLOB_CONN_STRING')}")
        # print(f"AZURE_OPENAI_KEY: {os.getenv('AZURE_OPENAI_KEY')}")
        # print(f"AZURE_SEARCH_ENDPOINT: {os.getenv('AZURE_SEARCH_ENDPOINT')}")
        # print(f"EMBEDDING_DEPLOYMENT: {os.getenv('AZURE_EMBEDDING_DEPLOYMENT_NAME')}")
        # print(f"DOC_INTELL_ENDPOINT: {azure_doc_intell_endpoint}")
        # print(f"DOC_INTELL_KEY: {azure_doc_intell_key}")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        embedder = AzureOpenAIEmbeddings(
            azure_deployment=azure_oai_embedding_deployment,
            openai_api_key=azure_oai_key,
            openai_api_version=azure_openai_api_version,
            azure_endpoint=azure_oai_endpoint
        )

        # openai model used to generate embeddings
        embedder_client = AzureOpenAI(
            azure_endpoint=azure_oai_endpoint,
            api_key=azure_oai_key,
            api_version=azure_openai_api_version
        )

        for config in INDEX_CONFIGS:
            logging.info(f"=== Creating index: {config['index_name']} ===")
            metadata_df = indexer.read_metadata_from_blob(connection_string, config['metadata_container'], config['metadata_blob'])
            metadata_df = clean_metadata(metadata_df, SCHEMA_MAPPING_DICT, config['index_name'])
            indexer.create_index(config['index_name'], search_key, search_endpoint)

            # Get total files in the document container
            container_client = ContainerClient.from_connection_string(connection_string, config['document_container'])
            blob_list = [b for b in container_client.list_blobs() if b.name != "index_log.csv"]
            total_files = len(blob_list)
            batch_size = 50
            total_batches = (total_files + batch_size - 1) // batch_size
            logging.info(f"Found {total_files} blobs in container '{config['document_container']}'")
            
            for batch_number in range(total_batches):
                logging.info(f"--- Running batch {batch_number + 1} of {total_batches} for index {config['index_name']} ---")
                indexer.data_chunck_embed_upload_batch(
                    splitter, embedder, embedder_client, connection_string, config['document_container'], metadata_df,
                    config['index_name'], azure_doc_intell_endpoint, azure_doc_intell_key, azure_oai_deployment_endpoint, azure_oai_key, 
                    azure_oai_deployment_model, using_embedder=True, batch_number=batch_number, batch_size=batch_size
                )
            logging.info(f"Index {config['index_name']} created and populated.") 

        logging.info(f"All index creation jobs completed.")
        

    except Exception as e:
        logging.exception(f"❌ Exception in background indexing job: {str(e)}")