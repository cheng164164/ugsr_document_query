import os
import base64
import logging
import json
from math import ceil
import azure.functions as func
from index_creation.config import ENV_VARS, INDEX_CONFIGS, SCHEMA_MAPPING_DICT
from index_creation.util import set_env_vars, clean_metadata
from index_creation.indexer import *
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from openai import AzureOpenAI
from azure.storage.blob import ContainerClient
from datetime import datetime
from azure.storage.queue import QueueClient
from azure.core.exceptions import ResourceNotFoundError, HttpResponseError


logging.info("🔄 indexer_queue_trigger module loaded.")

set_env_vars(ENV_VARS)

connection_string = os.getenv('AZURE_BLOB_CONN_STRING')
azure_doc_intell_endpoint = os.getenv('AZURE_DOC_INTELL_ENDPOINT')
azure_doc_intell_key = os.getenv('AZURE_DOC_INTELL_KEY')
azure_oai_deployment_endpoint = os.getenv('AZURE_OPENAI_DEPLOYMENT_ENDPOINT')
azure_oai_key = os.getenv('AZURE_OPENAI_KEY')
azure_oai_deployment_model = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
search_key = os.getenv("AZURE_SEARCH_KEY")
azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
azure_oai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
azure_oai_embedding_deployment = os.getenv('AZURE_EMBEDDING_DEPLOYMENT_NAME', 'text-embedding-3-small')


def main(msg: func.QueueMessage) -> None:
    logging.info("⚙️ Queue trigger function started.")
    try:
        raw = msg.get_body().decode("utf-8")
        payload = json.loads(raw)
        index_name = payload.get("index_name")
        action = payload.get("action")
        enable_grouping = payload.get("enable_grouping", True)

        if not action or not index_name:
            logging.warning("⚠️ Missing action or index_name in message")
            return

        config = next((c for c in INDEX_CONFIGS if c["index_name"] == index_name), None)
        if not config:
            logging.warning(f"⚠️ No config found for index: {index_name}")
            return

        if action == "init_index":
            delete_existing_log_blob(index_name, connection_string)
            create_index(index_name, search_key, search_endpoint)
            save_index_log(index_name, {"index_name": index_name, "batches": [], "total_batches": payload.get("total_batches")}, connection_string)
            logging.info(f"✅ Initialized index and log for: {index_name}")
            return

        elif action == "start_indexing":
            batch_number = payload.get("batch_number")
            batch_size = payload.get("batch_size")
            total_batches = payload.get("total_batches")

            if batch_number is None or batch_size is None:
                logging.warning("⚠️ Missing batch_number or batch_size")
                return

            log = load_index_log(index_name, connection_string)
            run_index_job(config, log, batch_number, batch_size, total_batches)
            logging.info(f"✅ Finished batch {batch_number} for index: {index_name}")

            if enable_grouping:
                group_num = config.get("group")

                if check_if_group_complete(group_num, connection_string):
                    group_state = load_group_state(connection_string)
                    if group_state["current_group"] == group_num:
                        all_groups = group_state["all_groups"]
                        current_idx = all_groups.index(group_num)

                        if current_idx + 1 < len(all_groups):
                            next_group = all_groups[current_idx + 1]
                            queue_client = QueueClient.from_connection_string(os.getenv("AzureWebJobsStorage"), "indexing-requests")

                            for next_config in [c for c in INDEX_CONFIGS if c["group"] == next_group]:
                                enqueue_init_and_batches(next_config, queue_client, connection_string, enable_grouping=True)

                            save_group_state(next_group, all_groups, connection_string)
                            logging.info(f"📦 Queued group {next_group}")
        else:
            logging.warning(f"⚠️ Unknown action: {action}")

    except Exception as e:
        logging.exception(f"❌ Failed to process message: {e}")
        log_failed_message_to_blob(raw, str(e), connection_string, "failed-index-jobs", "failed-index-jobs")

def run_index_job(config, log, batch_number, batch_size, total_batches):
    set_env_vars(ENV_VARS=ENV_VARS)
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

    metadata_df = read_metadata_from_blob(connection_string, config['metadata_container'], config['metadata_blob'])
    metadata_df = clean_metadata(metadata_df, SCHEMA_MAPPING_DICT, config['index_name'])

    container_client = ContainerClient.from_connection_string(connection_string, config['document_container'])
    blob_list = [b.name for b in container_client.list_blobs() if b.name != "index_log.csv"]

    start = batch_number * batch_size
    end = min(start + batch_size, len(blob_list))
    blobs_to_process = blob_list[start:end]

    if batch_number in {b["batch_number"] for b in log.get("batches", []) if b["status"] == "uploaded"}:
        logging.info(f"⏩ Skipping already uploaded batch {batch_number}")
        return

    metadata_df = data_chunk_embed_upload_batch(
        splitter, embedder, embedder_client, connection_string, config['document_container'], metadata_df,
        config['metadata_container'], config['metadata_blob'], config['index_name'],
        azure_doc_intell_endpoint, azure_doc_intell_key, azure_oai_endpoint, azure_oai_key,
        azure_oai_deployment_model, using_embedder=True, batch_number=batch_number,
        batch_size=batch_size, total_batches=total_batches, blob_subset=blobs_to_process
    )

    update_batch_log(config["index_name"], batch_number, "uploaded", connection_string)

def enqueue_init_and_batches(config, queue_client, conn_str, enable_grouping=True, batch_size=100):
    index_name = config["index_name"]
    container_client = ContainerClient.from_connection_string(conn_str, config['document_container'])
    blob_list = [b for b in container_client.list_blobs() if b.name != "index_log.csv"]
    total_files = len(blob_list)
    total_batches = ceil(total_files / batch_size)
    init_msg = json.dumps({
        "action": "init_index",
        "index_name": index_name,
        "enable_grouping": enable_grouping,
        "total_batches": total_batches
    })
    queue_client.send_message(base64.b64encode(init_msg.encode("utf-8")).decode("utf-8"))
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

def load_group_state(conn_str, container="index-logs"):
    blob = BlobServiceClient.from_connection_string(conn_str).get_blob_client(container, "group_state.json")
    return json.loads(blob.download_blob().readall())

def save_group_state(current_group, all_groups, conn_str, container="index-logs"):
    state = {"current_group": current_group, "all_groups": all_groups}
    blob = BlobServiceClient.from_connection_string(conn_str).get_blob_client(container, "group_state.json")
    blob.upload_blob(json.dumps(state, indent=2), overwrite=True)

def check_if_group_complete(group_number, conn_str, container="index-logs"):
    group_indexes = [cfg["index_name"] for cfg in INDEX_CONFIGS if cfg.get("group") == group_number]
    for index_name in group_indexes:
        try:
            log = load_index_log(index_name, conn_str, container)
            total_batches = log.get("total_batches", 0)
            if total_batches == 0:
                logging.warning(f"⚠️ No batches found for index {index_name}")
                return False
            uploaded_batches = {b["batch_number"] for b in log.get("batches", []) if b["status"] == "uploaded"}
            if len(uploaded_batches) < total_batches:
                return False
        except Exception as e:
            logging.warning(f"⚠️ Could not verify log for index {index_name}: {e}")
            return False
    return True


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


def save_index_log(index_name, log_data, connection_string, container="index-logs"):
    try:
        blob_name = f"{index_name}_log.json"
        blob_client = BlobServiceClient.from_connection_string(connection_string).get_blob_client(
            container=container,
            blob=blob_name
        )
        blob_client.upload_blob(json.dumps(log_data, indent=2), overwrite=True)
        logging.info(f"📄 Index log saved to: {blob_name}")
    except Exception as e:
        logging.warning(f"⚠️ Failed to save index log for {index_name}: {e}")


def delete_existing_log_blob(index_name, connection_string, container="index-logs"):
    try:
        blob_name = f"{index_name}_log.json"
        blob_client = BlobServiceClient.from_connection_string(connection_string).get_blob_client(
            container=container,
            blob=blob_name
        )
        blob_client.delete_blob()
        logging.info(f"🗑️ Existing log blob deleted: {blob_name}")
    except Exception as cleanup_err:
        logging.info(f"ℹ️ No existing log to delete or deletion skipped: {cleanup_err}")


def load_index_log(index_name, connection_string, container="index-logs"):
    """
    Loads an existing index log from blob storage, or returns a new log template if not found.
    """
    blob_name = f"{index_name}_log.json"
    blob_client = BlobServiceClient.from_connection_string(connection_string).get_blob_client(
        container=container,
        blob=blob_name
    )

    try:
        log_content = blob_client.download_blob().readall()
        log = json.loads(log_content)
        return log
    except ResourceNotFoundError:
        logging.info(f"📄 No existing log found — starting new log for: {index_name}")
        return {"index_name": index_name, "batches": []}
    

def update_batch_log(index_name, batch_number, status, connection_string, error=None, container="index-logs"):
    """
    Safely updates the index log to include or overwrite the given batch entry.
    """
    try:
        # Load the latest log
        log = load_index_log(index_name, connection_string, container=container)

        # Remove any existing entry for the batch
        log["batches"] = [b for b in log.get("batches", []) if b["batch_number"] != batch_number]

        # Add the updated entry
        entry = {"batch_number": batch_number, "status": status}
        if error:
            entry["error"] = error
        log["batches"].append(entry)

        # Save updated log
        save_index_log(index_name, log, connection_string, container=container)
        logging.info(f"📝 Updated log for batch {batch_number} in index '{index_name}'")

    except Exception as e:
        logging.error(f"❌ Failed to update batch log for index '{index_name}': {e}")
    
