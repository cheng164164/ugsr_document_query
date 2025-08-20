import logging
import json
import os
import re
from datetime import datetime, timedelta
from pytz import timezone, utc
import azure.functions as func
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from databricks import sql
from openai import AzureOpenAI
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from index_creation import ENV_VARS, INDEX_CONFIGS
from index_creation.util import set_env_vars
from index_creation.indexer import is_english_filename


debug = False

def document_read(sas_url, endpoint, key):
    client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    poller = client.begin_analyze_document("prebuilt-read", {"urlSource": sas_url})
    result = poller.result()
    return result.content

def obtain_version_and_publish_date(context, azure_oai_endpoint, azure_oai_key, azure_oai_model):
    client = AzureOpenAI(
        azure_endpoint=azure_oai_endpoint,
        api_key=azure_oai_key,
        api_version="2025-01-01-preview",
    )
    messages = [{
        "role": "user",
        "content": (
            "From the following document content, extract the **version number** and **publish date** "
            "if they exist. Return only in this JSON format:\n"
            "{\"version\": <version>, \"publish_date\": <date>}.\n"
            "Here is the document content: " + context[:4000]
        )
    }]
    try:
        completion = client.chat.completions.create(
            model=azure_oai_model,
            messages=messages
        )
        result = completion.choices[0].message.content.strip()
        parsed = json.loads(result)
        return parsed.get("version"), parsed.get("publish_date")
    except Exception as e:
        logging.warning(f"OpenAI extraction failed: {e}")
        return None, None

def generate_blob_sas_url(connection_string, container_name, blob_name):
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    sas_token = generate_blob_sas(
        account_name=blob_client.account_name,
        container_name=container_name,
        blob_name=blob_name,
        account_key=blob_service_client.credential.account_key,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.utcnow() + timedelta(hours=1)
    )
    return f"{blob_client.url}?{sas_token}"


def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        set_env_vars(ENV_VARS=ENV_VARS)

        # Parse incoming data
        data = req.get_json()

        # Extract values from Power Automate body
        index_name = data.get("IndexName")
        file_name = data.get("FileName")
        event_type = data.get("EventType")
        file_modified_at_str = data.get("FileModifiedAt")  # UTC ISO string
        modified_by = data.get("ModifiedBy")
        file_url = data.get("URL")

        if debug:
            logging.info("⚠️ DEBUG MODE ACTIVE — Using hardcoded test values")
            index_name = "ugsr_index"
            file_name = "Test_Document.pdf"
            event_type = "modified"
            file_modified_at_str = "2025-08-13T14:00:00Z"
            modified_by = "debug.user@example.com"
            file_url = "https://yourstorage.blob.core.windows.net/container/Test_Document.pdf"

        # Convert FileModifiedAt to Central Time
        file_modified_utc = datetime.strptime(file_modified_at_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=utc)
        file_modified_central = file_modified_utc.astimezone(timezone("America/Chicago"))

        # Get current log time in Central Time
        log_time_utc = datetime.utcnow().replace(tzinfo=utc)
        log_time_central = log_time_utc.astimezone(timezone("America/Chicago"))

        # --- Generate SAS URL for Document Intelligence ---
        blob_conn_str = os.getenv("AZURE_BLOB_CONN_STRING")
        index_config = next((cfg for cfg in INDEX_CONFIGS if cfg["index_name"] == index_name), None)
        if not index_config:
            raise ValueError(f"Invalid index_name: {index_name}")
        
        blob_container = index_config["document_container"]
        blob_name = file_name
        sas_url = generate_blob_sas_url(blob_conn_str, blob_container, blob_name)

        version, publish_date = None, None

        # --- Only analyze content if file is in English ---
        if not debug and is_english_filename(file_name):
            try:
                # --- Document Intelligence ---
                doc_content = document_read(
                    sas_url,
                    os.getenv("AZURE_DOC_INTELL_ENDPOINT"),
                    os.getenv("AZURE_DOC_INTELL_KEY")
                )

                # --- Azure OpenAI ---
                version, publish_date = obtain_version_and_publish_date(
                    doc_content,
                    os.getenv("AZURE_OPENAI_ENDPOINT"),
                    os.getenv("AZURE_OPENAI_KEY"),
                    os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
                )
            except Exception as e:
                logging.warning(f"⚠️ Failed to extract version/publish date: {e}")
                version, publish_date = None, None

        elif debug:
            logging.info("DEBUG MODE: Skipping content analysis.")
        else:
            logging.info(f"Non-English file detected. Logging only metadata: {file_name}")

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
                IndexName, FileName, EventType, FileModifiedAtCentral, LoggedAtCentral, ModifiedBy,  Version, PublishDate, URL
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        with connection.cursor() as cursor:
            cursor.execute(query, (
                index_name,
                file_name,
                event_type,
                file_modified_central.strftime("%Y-%m-%d %H:%M:%S"),
                log_time_central.strftime("%Y-%m-%d %H:%M:%S"),
                modified_by,
                version,
                publish_date,
                file_url    
            ))

        return func.HttpResponse("Logged successfully", status_code=200)

    except Exception as e:
        logging.exception("Failed to log file event")
        return func.HttpResponse(f"Error: {str(e)}", status_code=500)
