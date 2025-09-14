import azure.functions as func
import logging
import pyodbc
import json
import os
import time

MAX_RETRIES = 9
RETRY_DELAY = 1  # seconds

def connect_with_retry(conn_str):
    for attempt in range(MAX_RETRIES):
        try:
            return pyodbc.connect(conn_str, timeout=30)
        except Exception as e:
            logging.warning(f"DB connection attempt {attempt+1} failed: {e}")
            time.sleep(RETRY_DELAY)
    raise Exception("Failed to connect to DB after retries")

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)


@app.route(route="sql_log_chat", methods=["POST"])
def sql_log_chat(req: func.HttpRequest) -> func.HttpResponse:
    try:
        logging.info("Received request")


        try:
            data = req.get_json()
        except ValueError:
            logging.error("❌ Invalid JSON payload")
            return func.HttpResponse("Invalid JSON format", status_code=400)
        
        if callable(data):
            return func.HttpResponse("FATAL: data is a function", status_code=500)
        
        logging.info(f"📦 Parsed request data: {data}")

        user_id = data.get("user_id", "unknown")
        user_name = data.get("user_name", "unknown")
        direction = data.get("direction", "unknown")
        content = data.get("content", "")
        metadata = data.get("metadata", {})
        metadata_json = json.dumps(metadata)

        # # Mock insert
        # print(f"[MOCK] Would insert: user_id={user_id}, user_name={user_name}, direction={direction}, content={content}, metadata={metadata_json}")

        # # Skip actual DB call during local dev
        # return func.HttpResponse("Mocked insert - test passed", status_code=200)

        conn_str = os.getenv("AZURE_SQL_CONNECTION_STRING")
        if not conn_str:
            logging.error("❌ AZURE_SQL_CONNECTION_STRING not set in environment variables.")
            return func.HttpResponse("Server error: missing DB connection string", status_code=500)
        
        logging.info("Connecting to SQL Server...")
        conn = connect_with_retry(conn_str)
        cursor = conn.cursor()

        logging.info("Inserting into ChatHistory...")
        cursor.execute("""
            INSERT INTO ChatHistory (UserId, UserName, Direction, Content, Metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (
            user_id,
            user_name,
            direction,
            content,
            metadata_json
        ))

        conn.commit()
        cursor.close()
        conn.close()

        logging.info("Insert completed")
        return func.HttpResponse("Logged to Azure SQL", status_code=200)
        
    except Exception as e:
        logging.error(f"Exception occurred: {str(e)}")
        return func.HttpResponse(f"Error: {str(e)}", status_code=500)
