import logging
import azure.functions as func
import json
from datetime import datetime
from pytz import timezone, utc
from databricks import sql
import os
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        # Parse incoming data
        data = req.get_json()

        # Extract values from Power Automate body
        file_name = data.get("FileName")
        event_type = data.get("EventType")
        file_modified_at_str = data.get("FileModifiedAt")  # UTC ISO string
        file_url = data.get("URL")

        # Convert FileModifiedAt to Central Time
        file_modified_utc = datetime.strptime(file_modified_at_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=utc)
        file_modified_central = file_modified_utc.astimezone(timezone("America/Chicago"))

        # Get current log time in Central Time
        log_time_utc = datetime.utcnow().replace(tzinfo=utc)
        log_time_central = log_time_utc.astimezone(timezone("America/Chicago"))

        # Load Databricks token (try Key Vault first, fallback to local setting)
        try:
            key_vault_url = os.environ.get("KEY_VAULT_URL")
            if key_vault_url:
                secret_name = "DATABRICKS_TOKEN"
                credential = DefaultAzureCredential()
                secret_client = SecretClient(vault_url=key_vault_url, credential=credential)
                token = secret_client.get_secret(secret_name).value
            else:
                raise ValueError("KEY_VAULT_URL not set")
        except Exception as e:
            logging.warning(f"Key Vault access failed or not configured: {e}. Falling back to local token.")
            token = os.environ["DATABRICKS_TOKEN"]

        # Connect to Databricks SQL warehouse
        connection = sql.connect(
            server_hostname=os.environ["DATABRICKS_HOST"],
            http_path=os.environ["DATABRICKS_HTTP_PATH"],
            access_token=os.environ["DATABRICKS_TOKEN"]
        )

        query = """
            INSERT INTO sharepoint_logs.file_change_log (
                FileName, EventType, FileModifiedAtCentral, LoggedAtCentral, URL
            ) VALUES (?, ?, ?, ?, ?)
        """

        with connection.cursor() as cursor:
            cursor.execute(query, (
                file_name,
                event_type,
                file_modified_central.strftime("%Y-%m-%d %H:%M:%S"),
                log_time_central.strftime("%Y-%m-%d %H:%M:%S"),
                file_url
            ))

        return func.HttpResponse("Logged successfully", status_code=200)

    except Exception as e:
        logging.exception("Failed to log file event")
        return func.HttpResponse(f"Error: {str(e)}", status_code=500)
