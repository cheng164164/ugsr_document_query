import logging
import azure.functions as func
import os
import json
import requests
import numpy as np
from sklearn.cluster import KMeans
from azure.storage.blob import BlobServiceClient, ContentSettings
from openai import AzureOpenAI
from index_creation.config import ENV_VARS, INDEX_CONFIGS, ADDITIONAL_EMBEDDINGS
from index_creation.util import set_env_vars
from helper import save_json_to_blob, embed_text, build_and_save_contact_kb


# Load environment variables from config.py
set_env_vars(ENV_VARS)

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_BLOB_CONN_STRING = os.getenv("AZURE_BLOB_CONN_STRING")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
EMBED_MODEL = os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME")

# Build index names from config
INDEX_NAMES = [cfg["index_name"] for cfg in INDEX_CONFIGS]

# Blob destination
CONTAINER_NAME = "index-metadata-summary"
BLOB_METADATA_SUMMARY = "metadata_summaries.json"
BLOB_CLUSTERED_PROFILES = "clustered_profiles.json"
BLOB_CLUSTERED_EMBEDDINGS = "clustered_profile_embeddings.json"


# ------------------------------------------
# Read ALL terms/topics from an index
# ------------------------------------------
def collect_all_phrases(index_name: str, batch_size: int = 500):
    """
    Reads ALL documents from an Azure AI Search index using $top + $skip
    paging, and extracts unique phrases from 'terms' and 'topics' fields.

    Args:
        index_name (str): Name of the Azure Search index.
        batch_size (int): Number of documents per page.

    Returns:
        list[str]: Unique collected phrases.
    """

    url = f"{AZURE_SEARCH_ENDPOINT}/indexes/{index_name}/docs/search?api-version=2024-07-01"
    headers = {"Content-Type": "application/json", "api-key": AZURE_SEARCH_KEY}

    phrases = set()
    total_docs = 0
    skip = 0
    page_number = 1

    print(f"📘 Starting full index scan for: {index_name}")

    while True:
        payload = {
            "search": "*",
            "select": "terms,topics",
            "top": batch_size,
            "skip": skip,
            "count": True,  # enables @odata.count
        }

        resp = requests.post(url, json=payload, headers=headers)
        if resp.status_code != 200:
            logging.error(f"❌ Failed to query index '{index_name}': {resp.text}")
            break

        data = resp.json()
        docs = data.get("value", [])
        page_count = len(docs)
        total_docs += page_count

        # Log the total index size (only on page 1)
        if skip == 0:
            reported_total = data.get("@odata.count")
            print(f"ℹ️ {index_name}: Azure reports {reported_total} total chunks.")

        print(f"📄 Page {page_number}: skip={skip}, returned {page_count} chunks")

        # Extract phrases from this page
        for d in docs:
            for field in ["terms", "topics"]:
                v = d.get(field)
                if isinstance(v, str):
                    phrases.add(v.strip())
                elif isinstance(v, list):
                    for item in v:
                        if item:
                            phrases.add(str(item).strip())

        # End of paging: fewer results than requested
        if page_count < batch_size:
            print(f"🏁 Completed paging for index '{index_name}'.")
            break

        # Move to next page
        skip += batch_size
        page_number += 1

    print(f"📥 {index_name}: collected {len(phrases)} unique phrases from {total_docs} chunks.")
    return list(phrases)


# ------------------------------------------
# Helper: Cluster phrases using embeddings
# ------------------------------------------
def cluster_phrases(phrases, k=120):
    if not phrases:
        return []

    logging.info(f"🔢 Embedding {len(phrases)} phrases...")
    embeddings = np.array([embed_text(p) for p in phrases], dtype=np.float32)

    if len(phrases) <= k:
        logging.info("Phrase count < K → skipping clustering.")
        return phrases

    logging.info("🤖 Running KMeans clustering...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(embeddings)
    centers = kmeans.cluster_centers_

    representatives = []
    for cid in range(k):
        idxs = np.where(labels == cid)[0]
        if len(idxs) == 0:
            continue

        centroid = centers[cid]
        best_idx = min(idxs, key=lambda i: np.linalg.norm(embeddings[i] - centroid))
        representatives.append(phrases[best_idx])

    logging.info(f"🏗️ Produced {len(representatives)} representative cluster phrases.")
    return representatives


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


# ------------------------------------------
# MAIN HTTP TRIGGER
# ------------------------------------------
def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("🚀 post_index_metadata HTTP trigger started.")

    try:
        metadata_summaries = {}
        clustered_profiles  = {}
        clustered_embeddings = {}

        for index in INDEX_NAMES:
            logging.info("=====================================")
            logging.info(f"🏁 Processing index: {index}")
            logging.info("=====================================")

            # ---------------------------------------------------------
            # 1. SAMPLING → metadata summary (NEW)
            # ---------------------------------------------------------
            summary = build_index_metadata_summary(index, sample_size=30)
            metadata_summaries[index] = summary

            # ---------------------------------------------------------
            # 2. Collect all phrases → cluster → profile text
            # ---------------------------------------------------------
            phrases = collect_all_phrases(index)
            print(f"📝 Clustering {len(phrases)} phrases for profile generation...")
            reps = cluster_phrases(phrases, k=500)

            profile_text = ". ".join(reps)
            clustered_profiles[index] = profile_text

            # ---------------------------------------------------------
            # 3. Embedding
            # ---------------------------------------------------------
            print('🔢 Embedding clustered profile text...')
            emb = embed_text(profile_text)
            clustered_embeddings[index] = emb.tolist()

        # Save outputs
        save_json_to_blob(
            AZURE_BLOB_CONN_STRING, CONTAINER_NAME,
            BLOB_METADATA_SUMMARY, metadata_summaries
        )

        save_json_to_blob(
            AZURE_BLOB_CONN_STRING, CONTAINER_NAME,
            BLOB_CLUSTERED_PROFILES, clustered_profiles
        )

        save_json_to_blob(
            AZURE_BLOB_CONN_STRING, CONTAINER_NAME,
            BLOB_CLUSTERED_EMBEDDINGS, clustered_embeddings
        )

        # =========================================================
        # ✅ NEW: Run additional embedding jobs (if configured)
        # =========================================================
        additional_results = {}
        if isinstance(ADDITIONAL_EMBEDDINGS, dict) and ADDITIONAL_EMBEDDINGS:
            for job_name, job_cfg in ADDITIONAL_EMBEDDINGS.items():
                try:
                    logging.info(f"🔧 Running additional embedding job: {job_name}")
                    # For now we support row-mode table KB jobs (contact lists)
                    if job_cfg.get("record_mode") == "row":
                        additional_results[job_name] = build_and_save_contact_kb(
                            AZURE_BLOB_CONN_STRING, job_cfg
                        )
                    else:
                        additional_results[job_name] = {
                            "status": "skipped",
                            "reason": f"Unsupported record_mode: {job_cfg.get('record_mode')}"
                        }
                except Exception as e:
                    logging.exception(f"❌ Additional embedding job failed: {job_name}")
                    additional_results[job_name] = {"status": "failed", "error": str(e)}
        else:
            logging.info("ℹ️ No ADDITIONAL_EMBEDDINGS configured; skipping extra embedding jobs.")

        return func.HttpResponse(
            json.dumps({
                "status": "success",
                "indexes_processed": INDEX_NAMES,
                "outputs": {
                    "metadata_summaries": BLOB_METADATA_SUMMARY,
                    "clustered_profiles": BLOB_CLUSTERED_PROFILES,
                    "clustered_embeddings": BLOB_CLUSTERED_EMBEDDINGS,
                    "additional_embedding_jobs": additional_results
                }
            }, indent=2),
            mimetype="application/json",
            status_code=200
        )

    except Exception as e:
        logging.exception("❌ Error in post-index-metadata function:")
        return func.HttpResponse(f"Error: {str(e)}", status_code=500)