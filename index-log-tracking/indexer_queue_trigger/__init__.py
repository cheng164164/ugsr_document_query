import os
import base64
import logging
import json
import time
from math import ceil
import azure.functions as func
from index_creation.config import ENV_VARS, INDEX_CONFIGS, SCHEMA_MAPPING_DICT, enable_grouping, enable_delta_updates
from index_creation.util import (set_env_vars, clean_metadata, delete_existing_log_blob, load_index_log, save_index_log, 
                                load_group_state, save_group_state, check_if_group_complete, update_batch_log, log_failed_message_to_blob, enqueue_init_and_batches) 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from openai import AzureOpenAI
from azure.storage.blob import ContainerClient
from datetime import datetime
from azure.storage.queue import QueueClient
from azure.core.exceptions import ResourceNotFoundError, HttpResponseError


logging.info("🔄 indexer_queue_trigger module loaded.")
set_env_vars(ENV_VARS)

if enable_delta_updates:
    from index_creation.indexer_delta import *
else:
    from index_creation.indexer import *

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
            # Re-create the index if it already exists
            create_index(index_name, search_key, search_endpoint)
            save_index_log(index_name, {
            "index_name": index_name,
            "group": config.get("group"),
            "batches": [],
            "total_batches": payload.get("total_batches")
            }, connection_string)

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

            # Find batch entry in the log
            batch_entry = next((b for b in log.get("batches", []) if b["batch_number"] == batch_number), None)
            should_run_batch = False

            if batch_entry is None:
                logging.info(f"ℹ️ No log entry found for batch {batch_number} — treating as new. Will run.")
                should_run_batch = True
            elif batch_entry["status"] == "uploaded":
                logging.info(f"⏩ Skipping already uploaded batch {batch_number} for index: {index_name}")
            elif batch_entry["status"] in {"failed", "pending"}:
                logging.info(f"🔁 Retrying previously failed/pending batch {batch_number} for index: {index_name}")
                should_run_batch = True
            else:
                logging.warning(f"⚠️ Unrecognized batch status '{batch_entry['status']}' — will retry as safety.")
                should_run_batch = True

            if should_run_batch:
                try:
                    run_index_job(config, log, batch_number, batch_size, total_batches)
                    logging.info(f"✅ Finished batch {batch_number} for index: {index_name}")

                except Exception as e:
                    logging.exception(f"❌ Error running batch {batch_number} for index: {index_name}")

                    # 🔴 Mark as failed
                    update_batch_log(
                        index_name=index_name,
                        batch_number=batch_number,
                        status="failed",
                        connection_string=connection_string,
                        container="index-logs",
                        extra_fields={"error": str(e)}
                    )

                    # Optionally log to blob
                    log_failed_message_to_blob(
                        msg_body=json.dumps(payload),
                        reason=str(e),
                        storage_conn_str=connection_string,
                        container="failed-index-jobs",
                        blob_prefix="failed-index-jobs"
                    )

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
                                logging.info(f"📨 Enqueueing index init for group {next_group}: {next_config['index_name']}")
                                enqueue_init_and_batches(next_config, queue_client, connection_string, enable_grouping=True, batch_size=batch_size)

                            save_group_state(next_group, all_groups, connection_string)
                            logging.info(f"✅ Moved to next group: {next_group}")
                            logging.info(f"📦 Queued group {next_group}")

                        elif enable_delta_updates:
                            logging.info("🧼 Triggering cleanup after final group")
                            search_client = SearchClient(endpoint=search_endpoint, index_name=index_name, credential=AzureKeyCredential(search_key))
                            clean_file_level_deletes_after_batch(connection_string, config['document_container'], search_client)

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

    # 🟡 Step 1: Mark this batch as pending
    update_batch_log(
        config["index_name"],
        batch_number,
        status="pending",
        connection_string=connection_string,
        container="index-logs",
        extra_fields={}
    )

    try:
        # ⚙️ Step 2: Run the actual indexing process
        result = data_chunk_embed_upload_batch(
            splitter, embedder, embedder_client, connection_string, config['document_container'], metadata_df,
            config['metadata_container'], config['metadata_blob'], config['index_name'],
            azure_doc_intell_endpoint, azure_doc_intell_key, azure_oai_endpoint, azure_oai_key, azure_openai_api_version,
            azure_oai_deployment_model, using_embedder=True, batch_number=batch_number,
            batch_size=batch_size, total_batches=total_batches, blob_subset=blobs_to_process
        )

        # ✅ Step 3: Mark batch as uploaded (success)

        # 📝 Extract detailed file info
        added_info = result.get("added_files", {})
        deleted_info = result.get("deleted_files", {})
        modified_info = result.get("modified_files", {})
        failed_info = result.get("failed_files", {})
        update_batch_log(
            config["index_name"],
            batch_number,
            status="uploaded",
            connection_string=connection_string,
            container="index-logs",
            extra_fields={
                "timestamp_central": result.get("timestamp_central",''),
                "uploaded_chunks": result.get("uploaded_chunks", 0),
                "deleted_chunks": result.get("deleted_chunks", 0),
                "skipped_files": result.get("skipped_files", 0),
                
                "added_files_count": added_info.get("count", 0),
                "added_files_list": added_info.get("files", []),

                "deleted_files_count": deleted_info.get("count", 0),
                "deleted_files_list": deleted_info.get("files", []),

                "modified_files_count": modified_info.get("count", 0),
                "modified_files_list": modified_info.get("files", []),

                "failed_files_count": failed_info.get("count", 0),
                "failed_files_list": failed_info.get("files", []),
            }
        )

    except Exception as e:
        logging.exception(f"❌ Failed to process batch {batch_number} for index: {config['index_name']}")
        
        # 🔴 Step 4: Mark batch as failed
        update_batch_log(
            config["index_name"],
            batch_number,
            status="failed",
            connection_string=connection_string,
            container="index-logs",
            extra_fields={"error": str(e)}
        )

        # Optionally log the failed message for recovery
        log_failed_message_to_blob(
            json.dumps({
                "index_name": config["index_name"],
                "batch_number": batch_number,
                "batch_size": batch_size,
                "total_batches": total_batches
            }),
            reason=str(e),
            storage_conn_str=connection_string,
            container="failed-index-jobs",
            blob_prefix="failed-index-jobs"
        )


