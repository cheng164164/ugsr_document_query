import os
import base64
import logging
import json
import azure.functions as func
from index_creation.config import ENV_VARS, INDEX_CONFIGS, SCHEMA_MAPPING_DICT
from index_creation.util import set_env_vars, clean_metadata
from index_creation.indexer import *
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from openai import AzureOpenAI
from azure.storage.blob import ContainerClient
from datetime import datetime

logging.info("🔄 indexer_queue_trigger module loaded.")

def main(msg: func.QueueMessage) -> None:
    logging.info("⚙️ Queue trigger function started.")
    try:
        raw = msg.get_body().decode("utf-8")
        logging.info(f"📩 Raw queue message: {raw}")
        payload = json.loads(raw)

        if payload.get("action") != "start_indexing":
            logging.warning("⚠️ Invalid action")
            return

        run_index_job()

    except Exception as e:
        logging.exception(f"❌ Failed to process message: {e}")
        # Log failed message to blob
        log_failed_message_to_blob(
            msg_body=raw,
            reason=str(e),
            storage_conn_str=os.getenv("AZURE_BLOB_CONN_STRING"),
            container="failed-index-jobs",
            blob_prefix="failed-index-jobs"
        )


def run_index_job():
    try:
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

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        embedder = AzureOpenAIEmbeddings(
            azure_deployment=azure_oai_embedding_deployment,
            openai_api_key=azure_oai_key,
            openai_api_version=azure_openai_api_version,
            azure_endpoint=azure_oai_endpoint
        )

        embedder_client = AzureOpenAI(
            azure_endpoint=azure_oai_endpoint,
            api_key=azure_oai_key,
            api_version=azure_openai_api_version
        )

        for config in INDEX_CONFIGS:
            try:
                logging.info(f"=== Creating index: {config['index_name']} ===")
                metadata_df = read_metadata_from_blob(connection_string, config['metadata_container'], config['metadata_blob'])
                metadata_df = clean_metadata(metadata_df, SCHEMA_MAPPING_DICT, config['index_name'])
                logging.info(f"Metadata loaded with {len(metadata_df)} rows.")

                create_index(config['index_name'], search_key, search_endpoint)

                container_client = ContainerClient.from_connection_string(connection_string, config['document_container'])
                blob_list = [b for b in container_client.list_blobs() if b.name != "index_log.csv"]
                total_files = len(blob_list)
                batch_size = 10
                total_batches = (total_files + batch_size - 1) // batch_size
                logging.info(f"Found {total_files} blobs in container '{config['document_container']}'")

                for batch_number in range(2):
                    logging.info(f"--- Running batch {batch_number + 1} of {total_batches} for index {config['index_name']} ---")
                    data_chunck_embed_upload_batch(
                        splitter, embedder, embedder_client, connection_string, config['document_container'], metadata_df,
                        config['index_name'], azure_doc_intell_endpoint, azure_doc_intell_key, azure_oai_deployment_endpoint, azure_oai_key,
                        azure_oai_deployment_model, using_embedder=True, batch_number=batch_number, batch_size=batch_size
                    )

                logging.info(f"✅ Finished: {config['index_name']}")

            except Exception as index_error:
                logging.exception(f"❌ Failed processing index {config['index_name']}: {str(index_error)}")

        logging.info("🎉 All index creation jobs completed.")

    except Exception as e:
        logging.exception(f"❌ Error in run_index_job: {str(e)}")



def log_failed_message_to_blob(msg_body: str, reason: str, storage_conn_str: str, container: str, blob_prefix: str):
    try:
        blob_service = BlobServiceClient.from_connection_string(storage_conn_str)
        container_client = blob_service.get_container_client(container)
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        blob_name = f"{blob_prefix}/failed_message_{timestamp}.json"

        failed_log = {
            "timestamp": timestamp,
            "reason": reason,
            "message": msg_body
        }

        container_client.upload_blob(blob_name, json.dumps(failed_log), overwrite=True)
        logging.info(f"📝 Logged failed message to blob: {blob_name}")
    except Exception as log_err:
        logging.error(f"❌ Failed to log message to blob: {log_err}")
