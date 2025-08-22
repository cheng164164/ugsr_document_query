import os
import logging
import json
import base64
import azure.functions as func
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from azure.storage.blob import ContainerClient
from openai import AzureOpenAI
from .config import INDEX_CONFIGS, ENV_VARS, SCHEMA_MAPPING_DICT
from .util import set_env_vars, clean_metadata
from .indexer import *
from azure.storage.queue import QueueClient
from math import ceil

set_env_vars(ENV_VARS)
logging.basicConfig(level=logging.INFO)
logging.info("🔁 index_creation HTTP trigger module loaded.")

enable_grouping = True

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("🚀 HTTP trigger function started.")
    try:
        conn_str = os.getenv("AzureWebJobsStorage")
        connection_string = os.getenv("AZURE_BLOB_CONN_STRING")

        queue_name = "indexing-requests"
        queue_client = QueueClient.from_connection_string(conn_str, queue_name)

        batch_size = 100  # adjust as needed

        if enable_grouping:
            logging.info("🧩 Group-by-group mode is ENABLED.")

            # Determine first group
            all_groups = sorted(set(cfg["group"] for cfg in INDEX_CONFIGS))
            first_group = all_groups[0]

            # Save initial group state to blob
            save_group_state(first_group, all_groups, connection_string)

            # Enqueue only first group
            for config in [cfg for cfg in INDEX_CONFIGS if cfg["group"] == first_group]:
                enqueue_init_and_batches(config, queue_client, connection_string, batch_size)


        else:
            logging.info("📦 Group-by-group mode is DISABLED. Enqueueing all libraries.")
            for config in INDEX_CONFIGS:
                enqueue_init_and_batches(config, queue_client, connection_string, batch_size)

        logging.info("✅ Indexing messages enqueued successfully.")
        return func.HttpResponse("Message(s) enqueued.", status_code=202)

    except Exception as e:
        logging.exception("❌ Failed to enqueue message.")
        return func.HttpResponse(f"Error: {str(e)}", status_code=500)


def enqueue_init_and_batches(config, queue_client, blob_conn_str, batch_size):
    index_name = config["index_name"]

    container_client = ContainerClient.from_connection_string(blob_conn_str, config['document_container'])
    blob_list = [b for b in container_client.list_blobs() if b.name != "index_log.csv"]
    total_files = len(blob_list)
    total_batches = ceil(total_files / batch_size)

    # Send init message
    init_msg = json.dumps({"action": "init_index", 
                           "index_name": index_name, 
                           "enable_grouping": enable_grouping, 
                           "total_batches": total_batches})
    queue_client.send_message(base64.b64encode(init_msg.encode("utf-8")).decode("utf-8"))

    # Send batch messages
    for batch_number in range(total_batches):
        msg = json.dumps({
            "action": "start_indexing",
            "index_name": index_name,
            "batch_number": batch_number,
            "batch_size": batch_size,
            "total_batches": total_batches,
            "enable_grouping": enable_grouping
        })
        queue_client.send_message(base64.b64encode(msg.encode("utf-8")).decode("utf-8"), visibility_timeout=10)

    logging.info(f"📨 Enqueued {total_batches} batch(es) for index: {index_name}")


def save_group_state(current_group, all_groups, conn_str, container="index-logs"):
    try:
        state = {
            "current_group": current_group,
            "all_groups": all_groups
        }
        blob_service = BlobServiceClient.from_connection_string(conn_str)
        blob_client = blob_service.get_blob_client(container=container, blob="group_state.json")
        blob_client.upload_blob(json.dumps(state, indent=2), overwrite=True)
        logging.info("📄 Saved group_state.json to blob storage.")
    except Exception as e:
        logging.error(f"❌ Failed to save group state: {e}")
