import os
import logging
import json
import base64
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
from azure.storage.queue import QueueClient


logging.basicConfig(level=logging.INFO)
logging.info("🔁 index_creation HTTP trigger module loaded.")


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("🚀 HTTP trigger function started.")
    try:
        conn_str = os.getenv("AzureWebJobsStorage")
        queue_name = "indexing-requests"
        queue_client = QueueClient.from_connection_string(conn_str, queue_name)

        message = json.dumps({"action": "start_indexing"})
        encoded = base64.b64encode(message.encode("utf-8")).decode("utf-8")
        queue_client.send_message(encoded)

        logging.info("✅ Base64-encoded message sent to queue.")
        return func.HttpResponse("Message enqueued.", status_code=202)

    except Exception as e:
        logging.exception("❌ Failed to enqueue message.")
        return func.HttpResponse(f"Error: {str(e)}", status_code=500)
    

