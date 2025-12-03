import os
import logging
import azure.functions as func
from dotenv import load_dotenv
from azure.storage.blob import ContainerClient
from .config import INDEX_CONFIGS, ENV_VARS, SCHEMA_MAPPING_DICT, enable_delta_updates, enable_grouping
from .util import set_env_vars, save_group_state, enqueue_init_and_batches, trigger_file_cleanup, delete_existing_log_blob
from azure.storage.queue import QueueClient


set_env_vars(ENV_VARS)
logging.basicConfig(level=logging.INFO)
logging.info("🔁 index_creation HTTP trigger module loaded.")
 

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("🚀 HTTP trigger function started.")
    try:
        conn_str = os.getenv("AzureWebJobsStorage")
        connection_string = os.getenv("AZURE_BLOB_CONN_STRING")

        queue_name = os.getenv("QUEUE_NAME")
        queue_client = QueueClient.from_connection_string(conn_str, queue_name)

        batch_size =30 # adjust as needed

        if enable_grouping:
            logging.info("🧩 Group-by-group mode is ENABLED.")

            # Determine first group
            all_groups = sorted(set(cfg["group"] for cfg in INDEX_CONFIGS))
            first_group = all_groups[0]

            # Save initial group state to blob
            delete_existing_log_blob('group_state', connection_string)  # delete existing group state blob
            save_group_state(first_group, all_groups, connection_string)

            # Enqueue only first group
            for config in [cfg for cfg in INDEX_CONFIGS if cfg["group"] == first_group]:
                enqueue_init_and_batches(config, queue_client, connection_string, enable_grouping, batch_size)

            # If only one group, trigger cleanup directly
            if len(all_groups) == 1 and enable_delta_updates:
                logging.info("🧼 Triggering cleanup after single-group indexing.")
                trigger_file_cleanup(config)


        else:
            logging.info("📦 Group-by-group mode is DISABLED. Enqueueing all libraries.")
            for config in INDEX_CONFIGS:
                enqueue_init_and_batches(config, queue_client, connection_string, enable_grouping, batch_size)

            # Trigger cleanup for entire index if no grouping
            if enable_delta_updates:
                for config in INDEX_CONFIGS:
                    trigger_file_cleanup(config)

        logging.info("✅ Indexing messages enqueued successfully.")
        return func.HttpResponse("Message(s) enqueued.", status_code=202)

    except Exception as e:
        logging.exception("❌ Failed to enqueue message.")
        return func.HttpResponse(f"Error: {str(e)}", status_code=500)

    