import os
import logging
import json
import base64
import time
import azure.functions as func
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from azure.storage.blob import ContainerClient
from openai import AzureOpenAI
from .config import INDEX_CONFIGS, ENV_VARS, SCHEMA_MAPPING_DICT, enable_delta_updates, enable_grouping
from .util import set_env_vars, clean_metadata
from azure.storage.queue import QueueClient
from math import ceil

set_env_vars(ENV_VARS)
logging.basicConfig(level=logging.INFO)
logging.info("🔁 index_creation HTTP trigger module loaded.")

if enable_delta_updates:
    from index_creation.indexer_delta import *
else:
    from index_creation.indexer import *


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("🚀 HTTP trigger function started.")
    try:
        conn_str = os.getenv("AzureWebJobsStorage")
        connection_string = os.getenv("AZURE_BLOB_CONN_STRING")

        queue_name = os.getenv("QUEUE_NAME")
        queue_client = QueueClient.from_connection_string(conn_str, queue_name)

        batch_size =50  # adjust as needed

        if enable_grouping:
            logging.info("🧩 Group-by-group mode is ENABLED.")

            # Determine first group
            all_groups = sorted(set(cfg["group"] for cfg in INDEX_CONFIGS))
            first_group = all_groups[0]

            # Save initial group state to blob
            save_group_state(first_group, all_groups, connection_string)

            # Enqueue only first group
            for config in [cfg for cfg in INDEX_CONFIGS if cfg["group"] == first_group]:
                enqueue_init_and_batches(config, queue_client, connection_string, enable_grouping, batch_size)

            # If only one group, trigger cleanup directly
            if len(all_groups) == 1 and enable_delta_updates:
                logging.info("🧼 Triggering cleanup after single-group indexing.")
                trigger_cleanup(config)


        else:
            logging.info("📦 Group-by-group mode is DISABLED. Enqueueing all libraries.")
            for config in INDEX_CONFIGS:
                enqueue_init_and_batches(config, queue_client, connection_string, enable_grouping, batch_size)

            # Trigger cleanup for entire index if no grouping
            if enable_delta_updates:
                for config in INDEX_CONFIGS:
                    trigger_cleanup(config)

        logging.info("✅ Indexing messages enqueued successfully.")
        return func.HttpResponse("Message(s) enqueued.", status_code=202)

    except Exception as e:
        logging.exception("❌ Failed to enqueue message.")
        return func.HttpResponse(f"Error: {str(e)}", status_code=500)


def enqueue_init_and_batches(config, queue_client, blob_conn_str, enable_grouping, batch_size):
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
    logging.info(f"📨 Sent init_index message for index: {index_name}")

    # Pre-log all batches as pending
    for batch_number in range(total_batches):
        update_batch_log(
            index_name=index_name,
            batch_number=batch_number,
            status="pending",
            connection_string=blob_conn_str,
            container="index-logs",
            extra_fields={"pre_logged": True}
        )

    # Wait until index is available (polling instead of fixed sleep)
    wait_for_index_ready(index_name)

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
        queue_client.send_message(base64.b64encode(msg.encode("utf-8")).decode("utf-8"), visibility_timeout=40)

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


def wait_for_index_ready(index_name, search_endpoint=None, search_key=None, max_retries=30, delay_seconds=2):
    if not search_endpoint:
        search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    if not search_key:
        search_key = os.getenv("AZURE_SEARCH_KEY")

    client = SearchIndexClient(endpoint=search_endpoint, credential=AzureKeyCredential(search_key))

    for attempt in range(max_retries):
        try:
            client.get_index(index_name)
            logging.info(f"✅ Index {index_name} is now available.")
            return True
        except Exception as e:
            logging.info(f"⏳ Waiting for index '{index_name}' to be ready ({attempt + 1}/{max_retries})...")
            time.sleep(delay_seconds)

    logging.error(f"❌ Timed out waiting for index '{index_name}' to be ready after {max_retries * delay_seconds} seconds.")
    return False


def trigger_cleanup(config):
    try:
        from .indexer_delta import clean_file_level_deletes_after_batch
        index_name = config["index_name"]
        container = config["document_container"]
        search_client = SearchClient(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            index_name=index_name,
            credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY"))
        )
        clean_file_level_deletes_after_batch(
            connection_string=os.getenv("AZURE_BLOB_CONN_STRING"),
            container_name=container,
            search_client=search_client
        )
    except Exception as e:
        logging.error(f"❌ Failed to trigger cleanup: {e}")


def update_batch_log(index_name, batch_number, status, connection_string, error=None, container="index-logs", extra_fields=None):
    """
    Safely updates the index log to include or overwrite the given batch entry.
    """
    try:
        log = load_index_log(index_name, connection_string, container=container)

        # Safely get previous entry
        existing_entry = next((b for b in log.get("batches", []) if b["batch_number"] == batch_number), None)
        retry_count = 0
        if existing_entry:
            if status == "failed":
                retry_count = existing_entry.get("retry_count", 0) + 1
            else:
                retry_count = existing_entry.get("retry_count", 0)

        # Build log entry
        entry = {
            "batch_number": batch_number,
            "status": status,
            "retry_count": retry_count
        }
        if error:
            entry["error"] = error
        if extra_fields:
            entry.update(extra_fields)

        # ✅ Merge-safe update
        existing_batches = {b["batch_number"]: b for b in log.get("batches", [])}
        existing_batches[batch_number] = entry  # overwrite or insert
        log["batches"] = list(existing_batches.values())

        # Save back to blob
        save_index_log(index_name, log, connection_string, container=container)
        logging.info(f"📝 Safely updated log for batch {batch_number} in index '{index_name}'")

    except Exception as e:
        logging.error(f"❌ Failed to update batch log for index '{index_name}': {e}")


def save_index_log(index_name, log_data, connection_string, container="index-logs"):
    try:
        # Optionally inject group number if missing in log_data
        if "group" not in log_data:
            config = next((c for c in INDEX_CONFIGS if c["index_name"] == index_name), None)
            if config and "group" in config:
                log_data["group"] = config["group"]

        blob_name = f"{index_name}_log.json"
        blob_client = BlobServiceClient.from_connection_string(connection_string).get_blob_client(
            container=container,
            blob=blob_name
        )
        blob_client.upload_blob(json.dumps(log_data, indent=2), overwrite=True)
        logging.info(f"📄 Index log saved to: {blob_name}")
    except Exception as e:
        logging.warning(f"⚠️ Failed to save index log for {index_name}: {e}")


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
    