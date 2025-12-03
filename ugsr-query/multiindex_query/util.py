import os
import re
import pandas as pd
import logging
from tiktoken import get_encoding
from openai import AzureOpenAI
import pyodbc
import json
import time
import requests
from azure.storage.blob import BlobServiceClient, ContentSettings
from collections import Counter
from typing import List
from .config import ENV_VARS



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


set_env_vars(ENV_VARS)

# Tokenizer for GPT-4o (O3 models)
tokenizer = get_encoding("cl100k_base")

AZURE_BLOB_CONN_STRING = os.getenv("AZURE_BLOB_CONN_STRING")
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")

# Max token budget for safety (adjustable)
MAX_TOTAL_TOKENS = 16000
EXPECTED_COMPLETION_TOKENS = 1000
MAX_INPUT_TOKENS = MAX_TOTAL_TOKENS - EXPECTED_COMPLETION_TOKENS


def title_case_filename(filename):
    try:
        name, ext = os.path.splitext(filename)
        # Special case: if the name matches all caps with periods or digits (e.g., OP2.12U)
        if re.fullmatch(r'[a-z]{2,}[0-9.]*[a-z]?', name, re.IGNORECASE):
            return f"{name.upper()}{ext}"
        
        parts = re.split(r'[\s_\-\.]+', name)
        title_cased = ' '.join(p.capitalize() if not p.isupper() else p for p in parts)
        return f"{title_cased}{ext}"
    except Exception:
        return filename


def title_case_name(name):
    try:
        return ' '.join(part.capitalize() for part in name.strip().split())
    except Exception:
        return name


def detect_specific_index(query: str, index_aliases: dict | None) -> str | None:
    """
    Detect if the query explicitly mentions one of the index aliases.
    Returns the matched index name, or None if no match or if alias check should be skipped.
    """
    # Skip if alias mapping is missing or trivial
    if not index_aliases or len(index_aliases) <= 1:
        return None
    
    q_lower = query.lower()
    for index_name, aliases in index_aliases.items():
        if not aliases:  # skip if aliases list is empty
            continue
        for alias in aliases:
            if alias.lower() in q_lower:
                return index_name
    return None


def resolve_reference_url(filename: str, original_url: str, supplement_files: dict) -> str:
    """
    Returns a reference link for the document.
    If the filename is listed in supplement_files, return the associated reference_link.
    Otherwise, return the original_url.
    """
    cleaned_filename = filename.strip().lower()
    for index_docs in supplement_files.values():
        for entry in index_docs:
            if entry.get("file_name", "").strip().lower() == cleaned_filename:
                return entry.get("reference_link", original_url)
    return original_url


def resolve_reference_name(filename: str, supplement_files: dict) -> str:
    """
    Returns a reference name for the document.
    If the filename is listed in supplement_files, return the reference file name without file extension.
    Otherwise, return the original filename.
    """
    cleaned_filename = filename.strip().lower()
    for index_docs in supplement_files.values():
        for entry in index_docs:
            if entry.get("file_name", "").strip().lower() == cleaned_filename:
                return os.path.splitext(filename)[0]
    return filename


def extract_structured_filenames(text: str, normalize: bool = True) -> List[str]:
    """
    Extracts all structured file names from the input text based on known patterns.
    Pattern examples: OP2.105U, OP20.2-F1, POL2.33U, MAN30.100, FRM56.2

    Features:
    - Matches multiple filenames per input
    - Supports lowercase or mixed case
    - Optional normalization (default: lowercase + strip trailing punctuation)
    
    Returns:
        A list of matched filenames (normalized if enabled), or empty list if none found.
    """
    # Known pattern:
    #   - 2+ letters
    #   - 1+ digits
    #   - a dot
    #   - 1+ digits
    #   - optional [A-Z] or -[A-Z0-9]+
    pattern = r'\b([A-Z]{2,}[0-9]+\.[0-9]+(?:[A-Z]|-[A-Z0-9]+)?)\b'

    matches = re.findall(pattern, text.upper())
    
    if normalize:
        # Lowercase and strip trailing punctuation like `.`, `,`, etc.
        return [m.lower().rstrip(".,;:") for m in matches]
    else:
        return matches



def count_tokens(text):
    return len(tokenizer.encode(text))


def truncate_history(history_turns, max_tokens=MAX_INPUT_TOKENS):
    total_tokens = 0
    truncated = []
    for turn in reversed(history_turns): # Newest to oldest
        tokens = count_tokens(turn)
        if total_tokens + tokens > max_tokens:
            break
        truncated.insert(0, turn)
        total_tokens += tokens
    return truncated


def get_connection(max_retries=5, delay_seconds=1):
    conn_str = os.environ.get("AZURE_SQL_CONN_STR")
    if not conn_str:
        logging.error("❌ AZURE_SQL_CONN_STR not found in environment settings")
        return None

    for attempt in range(max_retries):
        try:
            return pyodbc.connect(conn_str, timeout=50)
        except Exception as e:
            logging.warning(f"DB connection attempt {attempt+1} failed: {e}")
            time.sleep(delay_seconds)
    raise Exception("Failed to connect to DB after retries")


def save_chat(user_id, user_name, direction, content, metadata=None):
    try:
        conn = get_connection()
        if not conn:
            logging.error("❌ Failed to get DB connection in save_chat.")
            return
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO ChatHistory (UserId, UserName, Direction, Content, Metadata)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, user_name, direction, content, json.dumps(metadata or {})))
        conn.commit()
    except Exception as e:
        logging.error(f"❌ Error saving chat to DB: {e}")
    finally:
        try:
            cursor.close()
            conn.close()
        except:
            pass

def fetch_recent_history(query_history, answer_history, user_id, top_n=5):
    """
    Fetch recent conversation turns for a user.
    Returns two pipe-separated strings: q_hist and a_hist.
    """
    conn = None
    cursor = None

    try:
        conn = get_connection()
        if not conn:
            logging.error("❌ Failed to get DB connection in fetch_recent_history.")
            return "", ""

        cursor = conn.cursor()
        cursor.execute("""
            SELECT TOP (?) Direction, Content
            FROM ChatHistory
            WHERE UserId = ?
            ORDER BY Timestamp_Central ASC   -- ASC so turns stay in chronological order
        """, (top_n, user_id))
        rows = cursor.fetchall()

        queries = []
        answers = []

        for r in rows:
            if r[0] == "user":
                queries.append(r[1])
            elif r[0] == "bot":
                answers.append(r[1])

        q_hist = " | ".join(queries)
        a_hist = " | ".join(answers)
        return q_hist, a_hist

    except Exception as e:
        logging.error(f"❌ Error fetching recent history from DB: {e}")
        return "", ""

    finally:
        try:
            cursor.close()
            conn.close()
        except:
            pass


def save_json_to_blob(blob_conn_str, container_name, blob_name, data):
    service_client = BlobServiceClient.from_connection_string(blob_conn_str)
    try:
        container_client = service_client.get_container_client(container_name)

        # Ensure container exists — create if not
        if not container_client.exists():
            container_client.create_container()
            logging.info(f"📦 Created missing blob container: '{container_name}'")

        blob_client = container_client.get_blob_client(blob_name)

        blob_client.upload_blob(
            json.dumps(data, indent=2, ensure_ascii=False),
            overwrite=True,
            content_settings=ContentSettings(content_type="application/json")
        )

        logging.info(f"✅ Uploaded blob: '{blob_name}' to container: '{container_name}'")

    except Exception as e:
        logging.error(f"❌ Failed to upload blob '{blob_name}' to container '{container_name}': {e}")
        raise


def load_json_from_blob(blob_conn_str, container_name, blob_name):
    client = BlobServiceClient.from_connection_string(blob_conn_str)
    blob = client.get_blob_client(container=container_name, blob=blob_name)
    if blob.exists():
        stream = blob.download_blob()
        return json.loads(stream.readall())
    return None


def build_index_metadata_summary(index_name, sample_size=30):
    url = f"{AZURE_SEARCH_ENDPOINT}/indexes/{index_name}/docs/search?api-version=2024-07-01"
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_SEARCH_KEY
    }

    select_fields = ["terms", "topics"]
    payload = {
        "search": "*",
        "top": sample_size,
        "select": ",".join(select_fields),
        "queryType": "simple",
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code != 200:
            logging.warning(f"⚠️ Failed to query index '{index_name}': {response.text}")
            return ""
        docs = response.json().get("value", [])
        collected = {f: [] for f in select_fields}
        for doc in docs:
            for f in select_fields:
                val = doc.get(f)
                if isinstance(val, list):
                    collected[f].extend(val)
                elif isinstance(val, str):
                    collected[f].append(val)
        summary_parts = []
        for f, vals in collected.items():
            if vals:
                summary_parts.append(f"{f}: {', '.join(vals)}")

        return ". ".join(summary_parts)
    except Exception as e:
        logging.error(f"❌ Error building summary for {index_name}: {e}")
        return ""

# --- Initialize summaries (load or create) ---
def get_or_build_metadata_summaries(index_names, blob_conn_str, container_name, blob_name):
    summaries = load_json_from_blob(blob_conn_str, container_name, blob_name)
    if summaries:
        return summaries

    logging.info("⚙️ metadata_summaries.json not found. Building new summaries...")
    summaries = {index: build_index_metadata_summary(index) for index in index_names}
    save_json_to_blob(blob_conn_str, container_name, blob_name, summaries)
    logging.info("✅ Saved new metadata summaries to blob.")
    return summaries