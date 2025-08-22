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


logging.basicConfig(level=logging.INFO)
logging.info("🔁 index_creation HTTP trigger module loaded.")

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("🚀 HTTP trigger function started.")
    try:
        conn_str = os.getenv("AzureWebJobsStorage")
        connection_string = os.getenv("AZURE_BLOB_CONN_STRING")

        queue_name = "indexing-requests"
        queue_client = QueueClient.from_connection_string(conn_str, queue_name)

        batch_size = 100  # adjust as needed

        # Step 1: Determine first group
        all_groups = sorted(set(cfg["group"] for cfg in INDEX_CONFIGS))
        first_group = all_groups[0]
        save_group_state(first_group, all_groups, connection_string)

        for config in [cfg for cfg in INDEX_CONFIGS if cfg["group"] == first_group]:
            # Send init message FIRST
            index_name = config["index_name"]

            init_message = json.dumps({
                "action": "init_index",
                "index_name": index_name
            })
            queue_client.send_message(base64.b64encode(init_message.encode("utf-8")).decode("utf-8"))

            container_client = ContainerClient.from_connection_string(connection_string, config['document_container'])
            blob_list = [b for b in container_client.list_blobs() if b.name != "index_log.csv"]
            total_files = len(blob_list)
            total_batches = ceil(total_files / batch_size)

            for batch_number in range(total_batches):
                message = json.dumps({
                        "action": "start_indexing",
                        "index_name": config["index_name"],
                        "batch_number": batch_number,
                        "batch_size": batch_size,
                        "total_batches": total_batches
                    })
                encoded = base64.b64encode(message.encode("utf-8")).decode("utf-8")
                queue_client.send_message(encoded, visibility_timeout=10)

        logging.info("✅  First group message sent to queue.")
        return func.HttpResponse("Message enqueued.", status_code=202)

    except Exception as e:
        logging.exception("❌ Failed to enqueue message.")
        return func.HttpResponse(f"Error: {str(e)}", status_code=500)


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