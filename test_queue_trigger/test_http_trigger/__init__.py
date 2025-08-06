import logging
import os
import base64
import json
import azure.functions as func
from azure.storage.queue import QueueClient

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("🚀 HTTP trigger function started.")
    try:
        queue_name = "indexing-requests"
        conn_str = os.getenv("AzureWebJobsStorage")
        message = json.dumps({"action": "start_indexing"})
        encoded_message = base64.b64encode(message.encode("utf-8")).decode("utf-8")

        queue_client = QueueClient.from_connection_string(conn_str, queue_name)
        queue_client.send_message(encoded_message)

        logging.info("✅ Message sent to queue.")
        return func.HttpResponse("Message sent to queue.", status_code=202)

    except Exception as e:
        logging.exception("❌ Failed to enqueue message.")
        return func.HttpResponse(f"Error: {str(e)}", status_code=500)