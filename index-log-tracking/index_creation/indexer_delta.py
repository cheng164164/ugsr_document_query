import os
import uuid
import base64
import io
import json
import hashlib
from typing import Dict, List, Tuple, Iterable, Set
import traceback
import pandas as pd
import requests
import logging
from dotenv import load_dotenv
from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions, BlobClient, ContainerClient
from openpyxl import load_workbook
from datetime import datetime, timedelta
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchClient
from azure.search.documents.indexes.models import (
    SearchIndex, SearchField, SimpleField, SearchableField, VectorSearch,
    VectorSearchProfile, HnswAlgorithmConfiguration, VectorSearchAlgorithmKind,
    SemanticPrioritizedFields, SemanticConfiguration, SemanticField, SemanticSearch, SearchFieldDataType
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from openai import AzureOpenAI


def read_metadata_from_blob(connection_string, container_name, blob_name):
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

    # Determine file type
    _, file_extension = os.path.splitext(blob_name.lower())
    stream = io.BytesIO()
    blob_data = blob_client.download_blob()
    blob_data.readinto(stream)
    stream.seek(0)

    if file_extension == '.csv':
        df = pd.read_csv(stream)
        df = df.astype(str)
        return df

    elif file_extension in ['.xlsx', '.xls']:
        wb = load_workbook(stream, data_only=True)
        ws = wb.active
        rows = []
        header = [cell.value for cell in ws[1]]
        name_col_index = header.index("Name")
        header.append("url")

        add_url = 'url' not in header
        if add_url:
            header.append("url")

        for row in ws.iter_rows(min_row=2):
            row_values = [cell.value for cell in row]
            if add_url:
                name_cell = row[name_col_index]
                hyperlink = name_cell.hyperlink.target if name_cell.hyperlink else None
                row_values.append(hyperlink)
            rows.append(row_values)

        df = pd.DataFrame(rows, columns=header)
        df = df.astype(str)
        return df
    else:
        raise ValueError("Unsupported file format. Only .xlsx and .csv are supported.")


def create_index(index_name: str, search_key: str, search_endpoint: str) -> None:
    """Only create the index if it doesn't already exist. Delta-mode safe."""
    credential = AzureKeyCredential(search_key)
    index_client = SearchIndexClient(endpoint=search_endpoint, credential=credential)

    # Check if the index already exists
    try:
        index_client.get_index(index_name)
        logging.info(f"ℹ️ Index '{index_name}' already exists. Skipping creation (delta mode).")
        return
    except ResourceNotFoundError:
        logging.info(f"🆕 Index '{index_name}' does not exist. Proceeding to create it.")

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True, sortable=True, filterable=True, facetable=True),
        SimpleField(name="filename", type=SearchFieldDataType.String, filterable=True, sortable=True, facetable=True),
        SimpleField(name="url", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="title", type=SearchFieldDataType.String, filterable=True, searchable=True, facetable=True, retrievable=True),
        SearchableField(name="owner", type=SearchFieldDataType.String, filterable=True, searchable=True, facetable=True, retrievable=True),
        SearchableField(name="doc_type", type=SearchFieldDataType.String, filterable=True, searchable=True, facetable=True, retrievable=True),
        SearchableField(name="doc_category", type=SearchFieldDataType.String, filterable=True, searchable=True, facetable=True, retrievable=True),
        SearchableField(name="doc_function", type=SearchFieldDataType.String, filterable=True, searchable=True, facetable=True, retrievable=True),
        SearchableField(name="terms", type=SearchFieldDataType.String, filterable=True, searchable=True, retrievable=True),
        SearchableField(name="topics", type=SearchFieldDataType.String, filterable=True, searchable=True, retrievable=True),
        SearchableField(name="summary", type=SearchFieldDataType.String, filterable=False, facetable=False, searchable=True, retrievable=True),
        SearchableField(name="content", type=SearchFieldDataType.String, filterable=False, facetable=False, searchable=True, retrievable=True),
        SimpleField(name="chunk_id", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
        SimpleField(name="content_sha256", type=SearchFieldDataType.String, filterable=True, sortable=False),
        SimpleField(name="last_modified", type=SearchFieldDataType.DateTimeOffset, filterable=True, sortable=True),
        SearchField(name="content_embedding", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, hidden=False, vector_search_dimensions=1536, vector_search_profile_name='default'),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="default",
                kind=VectorSearchAlgorithmKind.HNSW,
                parameters={'m': 4, "efConstruction": 400, "efSearch": 500, "metric": "cosine"},
            )
        ],
        profiles=[VectorSearchProfile(name="default", algorithm_configuration_name="default")],
    )

    prioritized_fields = SemanticPrioritizedFields(
        title_field=SemanticField(field_name='title'),
        content_fields=[SemanticField(field_name='content'), SemanticField(field_name='summary')],
        keywords_fields=[
            SemanticField(field_name='terms'),
            SemanticField(field_name='topics'),
            SemanticField(field_name='doc_type'),
            SemanticField(field_name='doc_category'),
            SemanticField(field_name='doc_function'),
        ],
    )

    semantic_configuration = SemanticConfiguration(name="default-semantic", prioritized_fields=prioritized_fields)
    semantic_search = SemanticSearch(default_configuration_name="default-semantic", configurations=[semantic_configuration])

    index = SearchIndex(name=index_name, fields=fields, semantic_search=semantic_search, vector_search=vector_search)
    index_client.create_index(index)
    

def create_search_index_from_schema(index_name: str, fields: list, vector_config: dict = None, semantic_config: dict = None) -> None:
    """Create or update an Azure AI Search index with REST API using a provided schema JSON."""
    search_service = os.getenv("AZURE_SEARCH_ENDPOINT")
    search_key = os.getenv("AZURE_SEARCH_KEY")
    if not search_service or not search_key:
        raise ValueError("AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_KEY must be set in environment.")
    headers = {"Content-Type": "application/json", "api-key": search_key}
    index_definition = {"name": index_name, "fields": fields}
    if vector_config:
        index_definition["vectorSearch"] = vector_config
    if semantic_config:
        index_definition["semantic"] = semantic_config
    url = f"{search_service}/indexes/{index_name}?api-version=2023-10-01-Preview"
    response = requests.put(url, headers=headers, json=index_definition)
    if response.status_code not in [200, 201]:
        raise RuntimeError(f"Failed to create or update index: {response.status_code} {response.text}")


def list_blobs(connection_string, container_name):
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    return [b for b in container_client.list_blobs() if b.name != "index_log.csv"]

def generate_blob_sas_url(connection_string, container_name, blob_name):
    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

        if not blob_client.exists():
            print(f"❌ Blob does not exist: {blob_name}")
            return None
        
        account_name = blob_service_client.account_name
        account_key = None
        if hasattr(blob_service_client.credential, 'account_key'):
            account_key = blob_service_client.credential.account_key
        if not account_key:
            raise ValueError("❌ Could not extract account key for SAS generation.")
        sas_token = generate_blob_sas(
            account_name=blob_client.account_name,
            container_name=container_name,
            blob_name=blob_name,
            account_key=blob_service_client.credential.account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(hours=1)
        )
        return f"{blob_client.url}?{sas_token}"
    
    except Exception as e:
        print(f"❌ Error generating SAS URL for blob {blob_name}: {e}")
        return None

def document_read(sas_url, azure_doc_intell_endpoint, azure_doc_intell_key, file_extension=None):
    file_extension = file_extension.lower() if file_extension else None
    if file_extension in [".csv"]:
        df = pd.read_csv(sas_url)
        return df.to_string(index=False)

    elif file_extension in [".xls", ".xlsx"]:
        df = pd.read_excel(sas_url)
        return df.to_string(index=False)

    else:
        client = DocumentIntelligenceClient(endpoint=azure_doc_intell_endpoint, credential=AzureKeyCredential(azure_doc_intell_key))
        poller = client.begin_analyze_document("prebuilt-read", AnalyzeDocumentRequest(url_source=sas_url))
        result = poller.result()
        return result.content 


def obtain_topics(context, azure_oai_endpoint, azure_oai_key, azure_oai_deployment_model, azure_openai_api_version):
    client = AzureOpenAI(
        azure_endpoint=azure_oai_endpoint,
        api_key=azure_oai_key,
        api_version=azure_openai_api_version,
    )
    messages = [{"role": "user", "content": f"Obtain all the main topics mentioned in this document. Keep your response short and just include the topics. Also, include the scope and purpose. Here is the document Content: {context}"}]
    completion = client.chat.completions.create(
        model=azure_oai_deployment_model,
        messages=messages
    )
    return completion.choices[0].message.content

def obtain_key_terms(context, azure_oai_endpoint, azure_oai_key, azure_oai_deployment_model, azure_openai_api_version):
    client = AzureOpenAI(
        azure_endpoint=azure_oai_endpoint,
        api_key=azure_oai_key,
        api_version=azure_openai_api_version,
    )
    messages = [{"role": "user", "content": f"Obtain key terminology used in this document. Keep your response short and just include the terms. Here is the document Content: {context}"}]
    completion = client.chat.completions.create(
        model=azure_oai_deployment_model,
        messages=messages
    )
    return completion.choices[0].message.content

def obtain_summary(context, azure_oai_endpoint, azure_oai_key, azure_oai_deployment_model, azure_openai_api_version):
    client = AzureOpenAI(
        azure_endpoint=azure_oai_endpoint,
        api_key=azure_oai_key,
        api_version=azure_openai_api_version,
    )
    messages = [{"role": "user", "content": f"Obtain the summary of the document. Keep your response short under 200 words. Here is the document Content: {context}"}]
    completion = client.chat.completions.create(
        model=azure_oai_deployment_model,
        messages=messages
    )
    return completion.choices[0].message.content

def truncate_summary(text, max_chars=4000):
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    end = truncated.rfind(". ")
    return truncated[:end+1] if end > 0 else truncated


def obtain_version_and_publish_date(context, azure_oai_endpoint, azure_oai_key, azure_oai_deployment_model, azure_openai_api_version):
    from openai import AzureOpenAI

    client = AzureOpenAI(
        azure_endpoint=azure_oai_endpoint,
        api_key=azure_oai_key,
        api_version=azure_openai_api_version,
    )

    messages = [
        {
            "role": "user",
            "content": (
                "From the following document content, extract the **version number** and **publish date** "
                "(or effective date) if they exist. These are usually written together in a field like: \"Ver. 1.0, 05-25\"."
                "or the date maybe in format like 12/4/2024, the version maybe in format like 3.0. Please find the latest date and version.\n"
                "Return only in the following JSON format:\n"
                "{\"version\": <version>, \"publish_date\": <date>}\n"
                "If a field is not found, set it to null. Here is the document content: " + context[:4000]
            )
        }
    ]

    try:
        completion = client.chat.completions.create(
            model=azure_oai_deployment_model,
            messages=messages
        )
        result = completion.choices[0].message.content.strip()
        parsed = json.loads(result)
        return parsed.get("version"), parsed.get("publish_date")
    except Exception as e:
        return None, None


def save_metadata_to_blob(metadata_df: pd.DataFrame, connection_string: str, container_name: str, blob_name: str) -> None:
    """Persist a DataFrame as CSV to a blob path, overwriting prior content."""
    output_csv = metadata_df.to_csv(index=False)
    blob_client = BlobClient.from_connection_string(conn_str=connection_string, container_name=container_name, blob_name=blob_name)
    blob_client.upload_blob(output_csv, overwrite=True)


def is_english_filename(name: str) -> bool:
    """Return True if the filename is ASCII-only; False otherwise. Avoids regex to simplify delta pipeline."""
    try:
        name.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def make_doc_id(file_key: str, chunk_id: int) -> str:
    """Create a stable chunk identifier for Azure AI Search from a file key and chunk ordinal."""
    encoded = base64.urlsafe_b64encode(file_key.encode("utf-8")).decode("utf-8").rstrip("=")
    return f"{encoded}-chunk-{chunk_id}"


def compute_sha256(text: str) -> str:
    """Compute a SHA-256 hex digest of input text for change detection."""
    if text is None:
        text = ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_existing_chunks(search_client: SearchClient, filename: str) -> Dict[str, Dict]:
    """Fetch existing chunk docs for a filename from the index. Returns mapping id -> {chunk_id, content_sha256}."""
    results = search_client.search(
        search_text="*",
        filter=f"filename eq '{filename.lower()}'",
        select=["id", "chunk_id", "content_sha256"],
        top=1000,
    )
    out = {}
    for r in results:
        out[r["id"]] = {"chunk_id": r.get("chunk_id"), "content_sha256": r.get("content_sha256")}
    return out


def iter_all_index_docs(search_client: SearchClient, select_fields: List[str]) -> Iterable[dict]:
    """Stream all documents from the index selecting the given fields."""
    results = search_client.search(search_text="*", select=select_fields, top=1000, include_total_count=True)
    for r in results:
        yield r


def delete_docs_by_ids(search_client: SearchClient, ids: List[str]) -> None:
    """Delete documents from the index by their keys in chunks of 1000."""
    for i in range(0, len(ids), 1000):
        keys = [{"id": k} for k in ids[i : i + 1000]]
        search_client.delete_documents(documents=keys)


def chunk_and_embed_single_file(
    splitter: RecursiveCharacterTextSplitter,
    embedder,
    embedder_client,
    connection_string: str,
    container_name: str,
    metadata_df: pd.DataFrame,
    file_blob_name: str,
    azure_doc_intell_endpoint: str,
    azure_doc_intell_key: str,
    azure_oai_endpoint: str,
    azure_oai_key: str,
    azure_openai_api_version: str,
    azure_oai_deployment_model: str,
    search_client: SearchClient,
    using_embedder: bool = True,
) -> Tuple[List[dict], List[str]]:
    """Process a single blob with delta detection. Returns (docs_to_upsert, doc_ids_to_delete)."""
    file_name = file_blob_name
    extension = os.path.splitext(file_name)[-1].lower()
    sas_url = generate_blob_sas_url(connection_string, container_name, file_name)
    doc_content = document_read(sas_url, azure_doc_intell_endpoint, azure_doc_intell_key, extension)

    chunks = splitter.create_documents([doc_content])
    texts = [c.page_content for c in chunks]
    chunk_hashes = [compute_sha256(t) for t in texts]
    existing = get_existing_chunks(search_client, file_name.lower())
    # Skip processing if nothing has changed
    existing_hashes = {v['chunk_id']: v['content_sha256'] for v in existing.values()}
    current_hashes = {i: h for i, h in enumerate(chunk_hashes)}
    if existing_hashes == current_hashes:
        return [], []

    # Only now do the expensive work
    topics = obtain_topics(doc_content, azure_oai_endpoint, azure_oai_key, azure_oai_deployment_model, azure_openai_api_version)
    terms = obtain_key_terms(doc_content, azure_oai_endpoint, azure_oai_key, azure_oai_deployment_model, azure_openai_api_version)
    summary = truncate_summary(obtain_summary(doc_content, azure_oai_endpoint, azure_oai_key, azure_oai_deployment_model, azure_openai_api_version))
    ver, date = obtain_version_and_publish_date(doc_content, azure_oai_endpoint, azure_oai_key, azure_oai_deployment_model, azure_openai_api_version)

    meta_row = metadata_df[metadata_df["Name"].str.lower() == os.path.basename(file_name).lower()]
    if meta_row.empty:
        new_row = {
        "Name": os.path.basename(file_name),
        "Title": None,
        "url": None,
        "Document Owner(s)": None,
        "Doc Type": None,
        "Doc Category": None,
        "Function": None,
        "version": ver,
        "publish date": date,
        }
        metadata_df = pd.concat([metadata_df, pd.DataFrame([new_row])], ignore_index=True)
        meta = new_row
    else:
        idx = meta_row.index[0]
        metadata_df.at[idx, "version"] = ver
        metadata_df.at[idx, "publish date"] = date
        meta = meta_row.iloc[0].to_dict()

    if using_embedder:
        vectors = embedder.embed_documents(texts)
    else:
        resp = embedder_client.embeddings.create(model="text-embedding-3-small", input=texts)
        vectors = [item.embedding for item in resp.data]

    new_docs: List[dict] = []
    new_ids: Set[str] = set()

    for chunk_id, (vec, chunk_text, chash) in enumerate(zip(vectors, texts, chunk_hashes)):
        doc_id = make_doc_id(file_name.lower(), chunk_id)
        new_ids.add(doc_id)
        prev = existing.get(doc_id)
        if prev and prev.get("content_sha256") == chash:
            continue
        new_docs.append(
            {
            "id": doc_id,
            "filename": file_name.lower(),
            "title": (meta.get("Title") or "").strip().lower(),
            "url": (meta.get("url") or "").strip().lower(),
            "owner": (meta.get("Document Owner(s)") or "").strip().lower(),
            "doc_type": (meta.get("Doc Type") or "").strip().lower(),
            "doc_category": (meta.get("Doc Category") or "").strip().lower(),
            "doc_function": (meta.get("Function") or "").strip().lower(),
            "terms": terms,
            "topics": topics,
            "summary": summary,
            "content": chunk_text,
            "chunk_id": chunk_id,
            "content_sha256": chash,
            "last_modified": datetime.utcnow().isoformat() + "Z",
            "content_embedding": vec,
            }
        )

    to_delete = [doc_id for doc_id in existing.keys() if doc_id not in new_ids]
    return new_docs, to_delete


def upload_search_index(index_name: str, search_key: str, search_endpoint: str, indexed_docs: List[dict]) -> None:
    """Upload documents to Azure AI Search in blocks of 1000 using upload (no delta logic)."""
    credential = AzureKeyCredential(search_key)
    search_client = SearchClient(endpoint=search_endpoint, index_name=index_name, credential=credential)
    for i in range(0, len(indexed_docs), 1000):
        search_client.upload_documents(documents=indexed_docs[i : i + 1000])


def data_chunk_embed_upload_batch(
    splitter: RecursiveCharacterTextSplitter,
    embedder,
    embedder_client,
    connection_string: str,
    container_name: str,
    metadata_df: pd.DataFrame,
    metadata_container: str,
    metadata_blob_name: str,
    index_name: str,
    azure_doc_intell_endpoint: str,
    azure_doc_intell_key: str,
    azure_oai_endpoint: str,
    azure_oai_key: str,
    azure_openai_api_version: str,
    azure_oai_deployment_model: str,
    using_embedder: bool = True,
    batch_number: int = 0,
    batch_size: int = 30,
    total_batches: int = None,
    blob_subset=None
) -> pd.DataFrame:
    """Process a slice of blobs with delta detection and synchronize changes to Azure AI Search (upserts and deletes)."""

    if "version" not in metadata_df.columns:
        metadata_df["version"] = None
    if "publish date" not in metadata_df.columns:
        metadata_df["publish date"] = None

    container_client = ContainerClient.from_connection_string(connection_string, container_name)
    all_blobs = list(container_client.list_blobs())
    if blob_subset is not None:
        # Use passed subset directly
        blob_subset_set = set(blob_subset)  # For faster lookup
        current_batch = [b for b in all_blobs if b.name in blob_subset_set]
        total_files = len(current_batch)
        start = 0
        end = total_files
    else:
        # Use calculated slicing based on batch_number and batch_size
        blob_list = [b for b in all_blobs if b.name != "index_log.csv"]
        total_files = len(blob_list)
        start = batch_number * batch_size
        end = min(start + batch_size, total_files)
        current_batch = blob_list[start:end]

    if not current_batch:
        print(f"No files found in batch {batch_number}")
        return  {
                "metadata_df": metadata_df,
                "uploaded_chunks": 0,
                "deleted_chunks": 0,
                "skipped_files": 0,
                "added_files": 0,
                "deleted_files": 0,
                "modified_files": 0,
                "failed_files": 0
                }

    print(f"\n📦 [Index: {index_name}] Starting batch {batch_number + 1}: processing files {start + 1} to {end} of {total_files}")

    credential = AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY"))
    search_client = SearchClient(endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"), index_name=index_name, credential=credential)

    upserts: List[dict] = []
    deletes: List[str] = []
    upserts_by_file = {}
    failed_files = []
    current_filenames: Set[str] = set()

    added_files = set()
    deleted_files = set()
    modified_files = set()

    for i, blob in enumerate(current_batch):
        print(f"📄 [Index: {index_name} ({total_files} files)] [Batch {batch_number + 1}/{total_batches}] [{start + i + 1}/{total_files}] Processing: {blob.name}")
        if not is_english_filename(blob.name):
            continue
        current_filenames.add(blob.name.lower())
        try:
            new_docs, to_delete = chunk_and_embed_single_file(
                splitter=splitter,
                embedder=embedder,
                embedder_client=embedder_client,
                connection_string=connection_string,
                container_name=container_name,
                metadata_df=metadata_df,
                file_blob_name=blob.name,
                azure_doc_intell_endpoint=azure_doc_intell_endpoint,
                azure_doc_intell_key=azure_doc_intell_key,
                azure_oai_endpoint=azure_oai_endpoint,
                azure_oai_key=azure_oai_key,
                azure_openai_api_version=azure_openai_api_version,  
                azure_oai_deployment_model=azure_oai_deployment_model,
                search_client=search_client,
                using_embedder=using_embedder,
            )

            # Log changes per file
            if new_docs and not to_delete:
                added_files.add(blob.name)
            elif to_delete and not new_docs:
                deleted_files.add(blob.name)
            elif new_docs and to_delete:
                modified_files.add(blob.name)

            upserts_by_file[blob.name] = len(new_docs)
            upserts.extend(new_docs)
            deletes.extend(to_delete)
        except Exception as e:
            print(f"❌ Failed: {blob.name} — {str(e)}")
            failed_files.append(blob.name)
            continue

    save_metadata_to_blob(metadata_df, connection_string, metadata_container, metadata_blob_name)

    for i in range(0, len(upserts), 1000):
        batch = upserts[i : i + 1000]
        res = search_client.merge_or_upload_documents(documents=batch)
        for r in res:
            if not r.succeeded:
                print(f"❌ [Index: {index_name}] [Batch {batch_number + 1}/{total_batches or '?'}] Upload failed: {r.key} — {r.error_message}")
        print(f"✅ [Index: {index_name}] [Batch {batch_number + 1}/{total_batches or '?'}] Uploaded batch segment {i//1000 + 1} ({min(i+1000, len(upserts))}/{len(upserts)} chunks)")

    print(f"🎉 Finished uploading [Index: {index_name}] batch {batch_number + 1}/{total_batches or '?'} ({len(upserts)} chunks)")

    if deletes:
        delete_docs_by_ids(search_client, deletes)

    results = {
    "metadata_df": metadata_df,
    "uploaded_chunks": len(upserts),
    "deleted_chunks": len(deletes),
    "skipped_files": len(current_batch) - (len(upserts_by_file) + len(failed_files)),
    "added_files": len(added_files),
    "deleted_files": len(deleted_files),
    "modified_files": len(modified_files),
    "failed_files": len(failed_files),
    }

    return results


def clean_file_level_deletes_after_batch(
    connection_string: str,
    container_name: str,
    search_client: SearchClient
    ) -> None:
    """Deletes index documents for files no longer present in blob storage."""
    print("🔍 Starting post-batch file-level cleanup...")
    
    blob_filenames = {b.name.lower() for b in list_blobs(connection_string, container_name)}
    index_filenames = set()
    for doc in iter_all_index_docs(search_client, ["filename"]):
        index_filenames.add(doc.get("filename", "").lower())

    missing_in_blob = [fn for fn in index_filenames if fn and fn not in blob_filenames]
    print(f"🧹 Found {len(missing_in_blob)} file(s) in index that are no longer in blob storage.")

    stale_ids = []
    for fn in missing_in_blob:
        for r in search_client.search(search_text="*", filter=f"filename eq '{fn}'", select=["id"], top=1000):
            stale_ids.append(r["id"])

    if stale_ids:
        delete_docs_by_ids(search_client, stale_ids)
        print(f"✅ Deleted {len(stale_ids)} stale documents from index.")
    else:
        print("✅ No stale documents to delete.")
