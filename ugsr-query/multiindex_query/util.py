import os
import io
import re
import pandas as pd
import logging
import base64
from tiktoken import get_encoding
from datetime import datetime, timedelta
from openai import AzureOpenAI
import pyodbc
import json
import time
import numpy as np
import requests
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions, ContentSettings, BlobClient
from collections import Counter
from typing import List, Dict, Optional
from .config import ENV_VARS
from azure.core.exceptions import ResourceNotFoundError


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

_cached_cluster_profiles = None
_cached_cluster_embeddings = None
_cached_metadata_summaries = None


def get_feature_flags(chatbot_name: str, global_feature_flags: dict, chatbot_feature_overrides: dict) -> dict:
    """
    Returns the final feature flags for the chatbot.
    
    Rules:
    - If global debug_mode=True → return global flags unchanged.
    - Otherwise → merge global flags with chatbot-specific overrides.
    - Chatbot-specific flag "debug_mode" is ALWAYS forced to False for deployment.
    """
    global_flags = global_feature_flags.copy()
    global_debug = global_flags.get("debug_mode", False)

    # If global debug mode is ON → use global flags only
    if global_debug:
        return global_flags

    # Apply chatbot overrides if they exist
    bot_overrides = chatbot_feature_overrides.get(chatbot_name, {})
    final_flags = global_flags.copy()

    for key, value in bot_overrides.items():
        final_flags[key] = value

    # Chatbot debug_mode ALWAYS false
    final_flags["debug_mode"] = False

    return final_flags


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


def title_case_name(name: str) -> str:
    try:
        if not name:
            return name

        # Split by comma or semicolon and take only the FIRST segment
        first_part = name.replace(";", ",").split(",")[0].strip()

        # Title-case each word of the selected part
        return " ".join(word.capitalize() for word in first_part.split() if word)
    except Exception:
        return name


def _normalize(text: str) -> str:
    """Lowercase and collapse whitespace."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def detect_specific_index(
    query: str,
    index_aliases: Optional[Dict[str, List[str]]]
) -> Optional[List[str]]:
    """
    Detect all indexes whose aliases appear in the query using
    simple full-term (phrase) matching with word boundaries.

    - Returns a list of matched index names (1 or more)
    - Returns None if no match
    """
    if not index_aliases or len(index_aliases) <= 1:
        return None

    q_norm = _normalize(query)
    matched_indexes: List[str] = []

    for index_name, aliases in index_aliases.items():
        if not aliases:
            continue

        for alias in aliases:
            if not alias:
                continue

            alias_norm = _normalize(alias)
            if not alias_norm:
                continue

            # Full phrase match with word boundaries
            # e.g. "global ehs" -> r"\bglobal ehs\b"
            pattern = r"\b" + re.escape(alias_norm) + r"\b"

            if re.search(pattern, q_norm):
                matched_indexes.append(index_name)
                break  # avoid adding same index multiple times

    return matched_indexes or None



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


def strip_doc_references(text: str) -> str:
    # 1) Remove any (...) that contains DocN, e.g.
    #    (Doc1), (Doc1, Doc2), (refer to Doc1 and Doc2)
    text = re.sub(r"\([^)]*Doc\d+[^)]*\)", "", text)

    # 2) Remove any [...] that contains DocN, e.g.
    #    [Doc1], [Doc1, Doc2], [refer to Doc1 and Doc2]
    text = re.sub(r"\[[^\]]*Doc\d+[^\]]*\]", "", text)

    # 3) Remove free-standing phrases like:
    #    refer to Doc1 and Doc2
    #    see Doc1, Doc2
    text = re.sub(
        r"\b(?:see|refer to|see also)\s+Doc\d+(?:\s*(?:,|and)\s*Doc\d+)*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # 4) Remove any leftover standalone DocN tokens
    text = re.sub(r"\bDoc\d+\b", "", text)

    # 5) Clean up extra blank lines
    text = re.sub(r"\n\s*\n", "\n\n", text).strip()
    return text


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
            SELECT Direction, Content
            FROM (
                SELECT TOP (?) Direction, Content, Timestamp_Central
                FROM ChatHistory
                WHERE UserId = ?
                ORDER BY Timestamp_Central DESC   -- newest first
            ) AS recent
            ORDER BY Timestamp_Central ASC        -- chronological for LLM
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
    try:
        client = BlobServiceClient.from_connection_string(blob_conn_str)
        blob = client.get_blob_client(container=container_name, blob=blob_name)
        if not blob.exists():
            return None
        return json.loads(blob.download_blob().readall())
    except Exception as e:
        logging.error(f"❌ Failed loading {blob_name}: {e}")
        return None


def load_cluster_profiles_and_embeddings(blob_conn_str):
    global _cached_cluster_profiles, _cached_cluster_embeddings

    if (_cached_cluster_profiles is not None 
        and _cached_cluster_embeddings is not None):
        return _cached_cluster_profiles, _cached_cluster_embeddings

    profiles = load_json_from_blob(
        blob_conn_str,
        "index-metadata-summary",
        "clustered_profiles.json"
    )

    embeddings = load_json_from_blob(
        blob_conn_str,
        "index-metadata-summary",
        "clustered_profile_embeddings.json"
    )

    if not profiles or not embeddings:
        logging.warning("⚠ Clustered profiles or embeddings missing, using empty dicts.")
        profiles, embeddings = {}, {}

    _cached_cluster_profiles = profiles
    _cached_cluster_embeddings = embeddings

    return profiles, embeddings


def load_index_metadata_summaries(blob_conn_str):
    """Loads metadata_summaries.json from Blob Storage
    and caches it briefly for performance."""
    global _cached_metadata_summaries


    # use cached version if recent
    if _cached_metadata_summaries is not None:
        return _cached_metadata_summaries

    summaries = load_json_from_blob(
        blob_conn_str,
        "index-metadata-summary",
        "metadata_summaries.json"
    )

    if not summaries:
        logging.warning("⚠ metadata_summaries.json missing in blob! Using empty dict.")
        summaries = {}

    _cached_metadata_summaries = summaries
    return summaries


def is_meaningful_metadata_answer(ans: str) -> bool:
    """Detects whether metadata summary is meaningful or too vague."""
    if not ans:
        return False

    lowered = ans.lower()

    # common signs of empty/unhelpful metadata summaries
    bad_signals = [
        "no relevant", 
        "cannot find", 
        "sorry", 
        "not available", 
        "no metadata", 
        "no information", 
        "i cannot help",
        "please try looking it up",
        "i don't have access"
    ]

    if any(sig in lowered for sig in bad_signals):
        return False

    # Require at least some content
    if len(ans.strip()) < 50:
        return False

    return True


def generate_blob_sas_url(connection_string, container_name, blob_name):
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

    account_key = blob_service_client.credential.account_key
    sas_token = generate_blob_sas(
        account_name=blob_client.account_name,
        container_name=container_name,
        blob_name=blob_name,
        account_key=account_key,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.utcnow() + timedelta(hours=1)
    )

    return f"{blob_client.url}?{sas_token}"


def append_images_to_answer(main_answer, show_image=False):
    """
    Shows inline Base64 images first,
    then shows clickable image links in a separate section.
    """

    if not show_image:
        return main_answer

    image_pattern = re.compile(
        r'\b([A-Za-z0-9_\-\.]+\.(?:jpg|jpeg|png|gif|bmp|webp))\b',
        re.IGNORECASE
    )
    mentioned_images = list(set(image_pattern.findall(main_answer)))
    if not mentioned_images:
        return main_answer

    # sections in correct order
    inline_section = "\n\n"
    link_section   = "\n\n**Full Resolution Images:**\n"

    for img_name in mentioned_images:
        sas_url = None

        # always produce link
        try:
            sas_url = generate_blob_sas_url(
                connection_string=AZURE_BLOB_CONN_STRING,
                container_name="ldgn-documents",
                blob_name=img_name
            )
            link_section += f"- [{img_name}]({sas_url})\n"
        except Exception as e:
            logging.warning(f"SAS URL failed for {img_name}: {e}")

        # try inline image
        try:
            blob_client = BlobClient.from_connection_string(
                conn_str=AZURE_BLOB_CONN_STRING,
                container_name="ldgn-documents",
                blob_name=img_name
            )
            img_bytes = blob_client.download_blob().readall()

            # determine mime
            ext = img_name.lower().split(".")[-1]
            mime = f"image/{'jpeg' if ext in ['jpg','jpeg'] else ext}"

            # encode base64
            b64 = base64.b64encode(img_bytes).decode("utf-8")

            # add inline displayed image
            inline_section += (
                f'\n<img src="data:{mime};base64,{b64}" '
                f'alt="{img_name}" style="max-width:420px; border-radius:6px;" />\n'
            )

        except Exception as e:
            logging.warning(f"Inline image render failed for {img_name}: {e}")

    return main_answer + inline_section


def contact_search_from_blob(
    query: str,
    blob_conn_str: str,
    container: str = "index-metadata-summary",
    prefix: str = "contacts/",
    top_k: int = 1
):
    # -------------------------
    # Helpers
    # -------------------------
    def load_blob_bytes(blob_conn_str, container, blob_path):
        blob = BlobClient.from_connection_string(blob_conn_str, container, blob_path)
        return blob.download_blob().readall()

    def normalize(text):
        return str(text).lower().strip()


    def normalized_edit_similarity(a: str, b: str) -> float:
        """
        Computes normalized Levenshtein similarity: 1 - (edit_distance / max_len)
        """
        a = a.lower().strip()
        b = b.lower().strip()
        if not a or not b:
            return 0.0

        len_a, len_b = len(a), len(b)
        dp = [[0] * (len_b + 1) for _ in range(len_a + 1)]

        for i in range(len_a + 1):
            dp[i][0] = i
        for j in range(len_b + 1):
            dp[0][j] = j

        for i in range(1, len_a + 1):
            for j in range(1, len_b + 1):
                cost = 0 if a[i-1] == b[j-1] else 1
                dp[i][j] = min(
                    dp[i-1][j] + 1,      # deletion
                    dp[i][j-1] + 1,      # insertion
                    dp[i-1][j-1] + cost  # substitution
                )

        distance = dp[len_a][len_b]
        max_len = max(len_a, len_b)
        return 1 - (distance / max_len)
        

    def field_value_fuzzy_match(value: str, query: str, threshold: float = 0.75) -> bool:
        """
        Compares full field value to each query substring (phrase) using string edit similarity.
        """
        if not value or not query:
            return False

        value = value.lower().strip()
        query_tokens = re.findall(r"\b\w+\b", query.lower())

        if not query_tokens:
            return False

        val_len = len(value.split())
        window_sizes = [val_len - 1, val_len, val_len + 1]
        window_sizes = [w for w in window_sizes if w > 0]

        for w in window_sizes:
            for i in range(len(query_tokens) - w + 1):
                span = " ".join(query_tokens[i:i+w])
                sim = normalized_edit_similarity(span, value)
                if sim >= threshold:
                    return True

        return False

    def format_records_response(matched_records):
        if not matched_records:
            return "Sorry, I couldn't find any matching contact records."

        lines = [f"**Answer:**\n\nMatches for your query:\n"]
        for idx, rec in enumerate(matched_records, 1):
            fields = rec.get("fields", {})
            source = rec.get("source_file", "Unknown")
            sheet = rec.get("sheet", "")
            first = fields.get("First Name", "")
            last = fields.get("Last Name", "")
            full_name = f"{first} {last}".strip()

            lines.append(f"---\n**Match {idx}** (From Source: `{source}`)")
            if full_name:
                lines.append(f"- Contact Name: {full_name}")
            if fields.get("Email"):
                lines.append(f"- Email: {fields['Email']}")

            ignore = {"First Name", "Last Name", "Email"}
            for k, v in fields.items():
                if k in ignore or not v:
                    continue
                lines.append(f"- {k}: {str(v).strip()}")
            lines.append("")
        return "\n".join(lines).strip()

    # -------------------------
    # Load contact records + embeddings
    # -------------------------
    try:
        embeddings = np.load(io.BytesIO(load_blob_bytes(blob_conn_str, container, prefix + "contact_record_embeddings.npy")))
        raw_records = load_blob_bytes(blob_conn_str, container, prefix + "contact_records.jsonl")
        records = [json.loads(line) for line in raw_records.decode("utf-8").splitlines() if line.strip()]
    except ResourceNotFoundError:
        return "Sorry, I cannot help with that because I could not find the contact list information."
    except Exception as e:
        return f"Unexpected error occurred while loading contact data: {str(e)}"

    # -------------------------
    # Embed the user query
    # -------------------------
    try:
        client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-12-01-preview"
        )

        emb_response = client.embeddings.create(
            model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"),
            input=query
        )
        query_vec = np.array(emb_response.data[0].embedding, dtype=np.float32)

    except Exception as e:
        return f"Failed to embed query: {str(e)}"

    # -------------------------
    # Find top-1 record by similarity
    # -------------------------
    try:
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        sims = np.dot(embeddings, query_vec) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec) + 1e-8)
        top_index = int(np.argmax(sims))
        top_record = records[top_index]

    except Exception as e:
        return f"Failed to compute similarity: {str(e)}"

    # -------------------------
    # Extract overlapping fields from query
    # -------------------------
    query_text = normalize(query)
    top_fields = top_record.get("fields", {})
    matched_fields = {}

    for key, value in top_fields.items():
        if not value or not isinstance(value, str):
            continue
        val_norm = normalize(value)
        if val_norm in query_text or any(val_norm in t for t in query_text.split()):
            matched_fields[key] = value.strip()

    # fallback: advanced fuzzy match using spans
    if not matched_fields:
        for key, value in top_fields.items():
            if not isinstance(value, str) or not value.strip():
                continue
            if field_value_fuzzy_match(value, query, threshold=0.75):
                matched_fields[key] = value.strip()

        # If still nothing matches → just return top-1
        if not matched_fields:
            return format_records_response([top_record])

    # -------------------------
    # Find all rows in same file/sheet with same column:value
    # -------------------------
    source_file = top_record.get("source_file", "")
    sheet = top_record.get("sheet", "")
    matched_rows = []

    for rec in records:
        if rec.get("source_file") != source_file or rec.get("sheet") != sheet:
            continue
        fields = rec.get("fields", {})
        match_all = all(
            normalize(fields.get(k, "")) == normalize(v)
            for k, v in matched_fields.items()
        )
        if match_all:
            matched_rows.append(rec)

    # Always include the top-1 row if it wasn't in the match list
    if top_record not in matched_rows:
        matched_rows.insert(0, top_record)

    return format_records_response(matched_rows)
