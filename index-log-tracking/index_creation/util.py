import os
import re
import pandas as pd
import logging
import json 
import time
import base64
from datetime import datetime
from math import ceil
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions, BlobClient, ContainerClient
from azure.core.exceptions import ResourceNotFoundError
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchClient
from .config import INDEX_CONFIGS, ENV_VARS, SCHEMA_MAPPING_DICT, enable_delta_updates, enable_grouping


def set_env_vars(ENV_VARS=None):
    """
    Set environment variables for Azure/OpenAI config.
    If a variable is set in ENV_VARS and is not a placeholder, use it as override.
    Otherwise, use the value from .env (already loaded via load_dotenv()).
    """
    if ENV_VARS is None:
        ENV_VARS = {}
    for k in ENV_VARS:
        value = ENV_VARS[k]
        # Use ENV_VARS value if not a placeholder, else keep .env value
        if value and not value.startswith('<') and not value.endswith('>'):
            os.environ[k] = value


def clean_metadata(df, schema_mapping_dict, index_name):
    """
    Cleans and standardizes metadata DataFrame columns using schema_mapping_dict.
    - Renames columns to standard schema.
    - Removes hash numbers from string columns.
    - Fills missing values with empty string.
    """
    # Rename columns based on schema mapping
    for key_std, key_df in schema_mapping_dict[index_name].items():
        if not key_df:
            df[key_std] = ""  # Add empty column if key_std is missing
            continue
        elif key_df not in df.columns:
            print("Warning: Column '{}' not found in DataFrame for index '{}'. Skipping renaming.".format(key_df, index_name))
            continue
        elif key_std in df.columns:
            continue
        df = df.rename(columns={key_df: key_std})

    # Remove hash numbers
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(remove_hash_number)
    df = df.fillna("")

    # Merge duplicate file names by concatenating differing values with ';'
    try:    
        if 'Name' in df.columns:
            group_cols = [col for col in df.columns if col != 'Name']

            def merge_rows(group):
                merged = {}
                merged['Name'] = group.name
                for col in group_cols:
                    values = set(str(val) for val in group[col] if pd.notna(val) and val != "")
                    merged[col] = '; '.join(sorted(values))
                return pd.Series(merged)

            df = df.groupby('Name').apply(merge_rows).reset_index(drop=True)
    except:
        pass
    return df


def remove_hash_number(text):
    if isinstance(text, str):
        try:
            return re.sub(r"#\d+", "", text)
        except:      
            return text
        

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


def load_group_state(conn_str, container="index-logs"):
    blob = BlobServiceClient.from_connection_string(conn_str).get_blob_client(container, "group_state.json")
    return json.loads(blob.download_blob().readall())


def save_group_state(current_group, all_groups, conn_str, container="index-logs", status=None):
    try:
        state = {
            "current_group": current_group,
            "all_groups": all_groups
        }
        if status is not None:
            # this is where "All completed" gets set
            state["status"] = status
        else:
            # Preserve previous status if available
            try:
                existing = load_group_state(conn_str, container)
                if "status" in existing:
                    state["status"] = existing["status"]
            except Exception:
                pass  # Silent fail is okay

        blob = BlobServiceClient.from_connection_string(conn_str).get_blob_client(container, "group_state.json")
        blob.upload_blob(json.dumps(state, indent=2), overwrite=True)
    except Exception as e:
        logging.error(f"❌ Failed to save group state: {e}")


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
    
    # If this is the last group, update group_state.json to mark all complete
    try:
        group_state = load_group_state(conn_str, container)
        current_group = group_state.get("current_group")
        all_groups = group_state.get("all_groups", [])

        if group_number == all_groups[-1]: # last group
            save_group_state(current_group, all_groups, conn_str, container, status="All completed")
            logging.info("🎉 All groups completed. Marked in group_state.json.")
    except Exception as e:
        logging.warning(f"⚠️ Failed to finalize group_state.json: {e}")
    return True


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


def enqueue_init_and_batches(config, queue_client, blob_conn_str, enable_grouping=True, batch_size=100):
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


def trigger_file_cleanup(config):
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


