
import logging
import azure.functions as func
import os
import json
import requests
import numpy as np
from sklearn.cluster import KMeans
from azure.storage.blob import BlobServiceClient, ContentSettings
from openai import AzureOpenAI
from index_creation.config import ENV_VARS, INDEX_CONFIGS
from index_creation.util import set_env_vars
import io
import hashlib
from datetime import datetime, timezone
import pandas as pd


# Load environment variables from config.py
set_env_vars(ENV_VARS)

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
EMBED_MODEL = os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME")


# ------------------------------------------
# Azure OpenAI Embedding
# ------------------------------------------
def embed_text(text: str, max_tokens=4000):
    """
    Safely embeds long text by splitting into chunks under the embedding model token limit.
    Returns the mean-pooled embedding vector.
    """
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_API_VERSION,
    )

    # Split into rough token-sized chunks by words
    words = text.split()
    chunks = []
    current_chunk = []

    for w in words:
        current_chunk.append(w)
        if len(" ".join(current_chunk)) > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    print(f"📦 Embedding {len(chunks)} chunks due to token limits.")

    vectors = []
    for ch in chunks:
        resp = client.embeddings.create(
            model=EMBED_MODEL,
            input=ch
        )
        vec = np.array(resp.data[0].embedding, dtype=np.float32)
        vectors.append(vec)

    # Mean pooling = robust global embedding
    return np.mean(np.vstack(vectors), axis=0)


# ================================
# ✅ NEW: Batch embedding for many short texts (records)
# ================================
def embed_texts_batch(texts: list[str]) -> np.ndarray:
    """
    Batch embedding for speed. Returns (N, D) float32 array.
    """
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_API_VERSION,
    )
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts
    )
    vectors = [np.array(d.embedding, dtype=np.float32) for d in resp.data]
    return np.vstack(vectors)


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
    

# =========================================================
# ✅ NEW FEATURE: CONTACT LIST EMBEDDING GENERATION (CONFIG-DRIVEN)
# =========================================================

def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()

def _row_to_record_text(row: dict, max_chars: int) -> str:
    parts = []
    for k, v in row.items():
        if v is None:
            continue
        v_str = str(v).strip()
        if not v_str:
            continue
        parts.append(f"{str(k).strip()}: {v_str}")
    text = " | ".join(parts)
    return text[:max_chars] if len(text) > max_chars else text

def _parse_table_blob_to_records(file_bytes: bytes, blob_name: str, max_chars: int) -> list[dict]:
    """
    Reads CSV/XLSX (all sheets). Returns row records:
      {id, source_file, sheet, row_number, fields, record_text}
    """
    records: list[dict] = []
    lower = blob_name.lower()

    def add_df(df: pd.DataFrame, sheet: str | None):
        if df is None or df.empty:
            return
        df = df.replace({np.nan: ""}).fillna("").astype(str)
        df.columns = [str(c).strip() for c in df.columns]

        rows = df.to_dict(orient="records")
        for i, row in enumerate(rows, start=1):
            fields = {str(k).strip(): ("" if v is None else str(v).strip()) for k, v in row.items()}
            record_text = _row_to_record_text(fields, max_chars=max_chars)

            # stable id based on full row payload + location
            seed = json.dumps(
                {"source_file": blob_name, "sheet": sheet or "", "row_number": i, "fields": fields},
                ensure_ascii=False,
                sort_keys=True
            )
            records.append({
                "id": _sha256_text(seed),
                "source_file": blob_name,
                "sheet": sheet,
                "row_number": i,
                "fields": fields,
                "record_text": record_text
            })

    if lower.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes))
        add_df(df, sheet=None)
    elif lower.endswith(".xlsx") or lower.endswith(".xls"):
        xls = pd.ExcelFile(io.BytesIO(file_bytes))
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            add_df(df, sheet=sheet_name)
    else:
        # unsupported
        return []

    return records

def _upload_bytes_to_blob(blob_conn_str: str, container_name: str, blob_name: str, data: bytes, content_type: str):
    svc = BlobServiceClient.from_connection_string(blob_conn_str)
    container_client = svc.get_container_client(container_name)
    if not container_client.exists():
        container_client.create_container()
        logging.info(f"📦 Created missing blob container: '{container_name}'")

    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(
        data,
        overwrite=True,
        content_settings=ContentSettings(content_type=content_type)
    )
    logging.info(f"✅ Uploaded blob bytes: '{blob_name}' to container: '{container_name}'")

def build_and_save_contact_kb(blob_conn_str: str, job_cfg: dict) -> dict:
    """
    Builds row-level embedding cache from all csv/xlsx files in a blob container.
    job_cfg expects:
      source.container, source.file_types
      output.container, output.records_blob, output.embeddings_blob, output.manifest_blob
      max_record_chars
      embedding.batch_size
    """
    source = job_cfg["source"]
    output = job_cfg["output"]
    embed_cfg = job_cfg["embedding"]

    source_container = source["container"]
    file_types = tuple(source.get("file_types", [".csv", ".xlsx", ".xls"]))

    out_container = output["container"]
    records_blob = output["records_blob"]
    embeddings_blob = output["embeddings_blob"]
    manifest_blob = output["manifest_blob"]

    max_chars = int(job_cfg.get("max_record_chars", 4000))
    batch_size = int(embed_cfg.get("batch_size", 64))

    svc = BlobServiceClient.from_connection_string(blob_conn_str)
    src_client = svc.get_container_client(source_container)

    if not src_client.exists():
        logging.warning(f"⚠️ Contact source container does not exist: {source_container}")
        return {"status": "skipped", "reason": "missing_source_container", "source_container": source_container}

    blobs = list(src_client.list_blobs())
    targets = [b for b in blobs if b.name.lower().endswith(file_types)]
    logging.info(f"📒 Contact KB: found {len(targets)} files in '{source_container}'")

    all_records: list[dict] = []
    source_files = []

    for b in targets:
        try:
            data = src_client.get_blob_client(b.name).download_blob().readall()
            recs = _parse_table_blob_to_records(data, b.name, max_chars=max_chars)
            if recs:
                all_records.extend(recs)
                source_files.append({
                    "name": b.name,
                    "last_modified": b.last_modified.isoformat() if getattr(b, "last_modified", None) else None
                })
            logging.info(f"✅ Contact KB: parsed {len(recs)} rows from {b.name}")
        except Exception as e:
            logging.warning(f"⚠️ Contact KB: failed parsing {b.name}: {e}")

    if not all_records:
        logging.warning("⚠️ Contact KB: no records generated; skipping embedding build.")
        return {"status": "skipped", "reason": "no_records"}

    texts = [r["record_text"] for r in all_records]

    logging.info(f"🔢 Contact KB: embedding {len(texts)} records in batches of {batch_size}")
    vec_batches = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        vec_batches.append(embed_texts_batch(batch))
    emb_matrix = np.vstack(vec_batches).astype(np.float32)

    # Save JSONL records
    jsonl_bytes = ("\n".join(json.dumps(r, ensure_ascii=False) for r in all_records)).encode("utf-8")
    _upload_bytes_to_blob(
        blob_conn_str, out_container, records_blob,
        jsonl_bytes, content_type="application/jsonl"
    )

    # Save NPY embeddings
    buf = io.BytesIO()
    np.save(buf, emb_matrix)
    _upload_bytes_to_blob(
        blob_conn_str, out_container, embeddings_blob,
        buf.getvalue(), content_type="application/octet-stream"
    )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "embedding_model": EMBED_MODEL,
        "record_count": len(all_records),
        "vector_dim": int(emb_matrix.shape[1]),
        "source_container": source_container,
        "source_files": source_files,
        "outputs": {
            "records_jsonl": records_blob,
            "embeddings_npy": embeddings_blob
        }
    }
    save_json_to_blob(blob_conn_str, out_container, manifest_blob, manifest)

    logging.info("✅ Contact KB: build complete.")
    return {"status": "success", "record_count": len(all_records), "manifest_blob": manifest_blob}
