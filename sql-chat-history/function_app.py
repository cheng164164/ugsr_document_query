import azure.functions as func
import logging
import pyodbc
import json
import os

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

@app.route(route="sql_log_chat", methods=["POST"])
def sql_log_chat(req: func.HttpRequest) -> func.HttpResponse:
    try:
        logging.info("Received request")

        data = req.get_json()
        if callable(data):
            return func.HttpResponse("FATAL: data is a function", status_code=500)
        logging.info(f"Request data: {data}")

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
        logging.info("Connecting to SQL Server...")
        conn = pyodbc.connect(conn_str)
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
