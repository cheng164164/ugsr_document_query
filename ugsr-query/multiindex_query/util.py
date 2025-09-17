import os
import re
import pandas as pd
import logging
from tiktoken import get_encoding
from openai import AzureOpenAI
import pyodbc
import json
from datetime import datetime
import time

# Tokenizer for GPT-4o (O3 models)
tokenizer = get_encoding("cl100k_base")

# Max token budget for safety (adjustable)
MAX_TOTAL_TOKENS = 16000
EXPECTED_COMPLETION_TOKENS = 1000
MAX_INPUT_TOKENS = MAX_TOTAL_TOKENS - EXPECTED_COMPLETION_TOKENS


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


def get_connection(max_retries=8, delay_seconds=1):
    conn_str = os.environ.get("AZURE_SQL_CONN_STR")
    if not conn_str:
        logging.error("❌ AZURE_SQL_CONN_STR not found in environment settings")
        return None

    for attempt in range(max_retries):
        try:
            return pyodbc.connect(conn_str, timeout=30)
        except Exception as e:
            logging.warning(f"DB connection attempt {attempt+1} failed: {e}")
            time.sleep(delay_seconds)
    raise Exception("Failed to connect to DB after retries")


def save_chat(user_id, user_name, direction, content, metadata=None):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO ChatHistory (UserId, UserName, Direction, Content, Metadata)
        VALUES (?, ?, ?, ?, ?)
    """, (user_id, user_name, direction, content, json.dumps(metadata or {})))
    conn.commit()
    cursor.close()
    conn.close()

def fetch_recent_history(user_id, top_n=10):
    """
    Fetch recent conversation turns for a user.
    Returns two pipe-separated strings: q_hist and a_hist.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT TOP (?) Direction, Content
        FROM ChatHistory
        WHERE UserId = ?
        ORDER BY Timestamp_Central ASC   -- ASC so turns stay in chronological order
    """, (top_n, user_id))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

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