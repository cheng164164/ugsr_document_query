import os
import hashlib
import base64
import time
from datetime import datetime
from lxml import etree
from pytz import timezone
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import ContainerClient
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SearchField, SimpleField, SearchableField,
    VectorSearch, VectorSearchProfile, HnswAlgorithmConfiguration,
    VectorSearchAlgorithmKind, SemanticPrioritizedFields,
    SemanticConfiguration, SemanticField, SemanticSearch,
    SearchFieldDataType
)
from azure.storage.blob import BlobServiceClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from openai import RateLimitError
import logging
from azure.core.exceptions import ResourceNotFoundError
from typing import List, Dict, Set
from io import StringIO
import pandas as pd



def compute_sha256(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def make_doc_id(file_key: str, chunk_id: int):
    encoded = base64.urlsafe_b64encode(file_key.encode('utf-8')).decode('utf-8').rstrip('=')
    return f"{encoded}-chunk-{chunk_id}"


def retry_embedding_with_backoff(embedder, texts, max_retries=5):
    delay = 10
    for attempt in range(max_retries):
        try:
            return embedder.embed_documents(texts)
        except RateLimitError:
            time.sleep(delay)
            delay *= 2
    raise Exception("Embedding failed after retries.")


def get_existing_chunks(search_client: SearchClient, filename: str):
    results = search_client.search(
        search_text="*",
        filter=f"filename eq '{filename}'",
        select=["id", "chunk_id", "content_sha256"],
        top=1000,
    )
    return {r["id"]: {"chunk_id": r["chunk_id"], "content_sha256": r["content_sha256"]} for r in results}



def parse_dita_xml_blob(blob_data: bytes, filename: str):
    parser = etree.XMLParser(recover=True)
    root = etree.fromstring(blob_data, parser=parser)
    chunks = []
    title_elem = root.find('.//{*}title')
    main_title = title_elem.text.strip() if title_elem is not None else ""

    for i, elem in enumerate(root.iter()):
        if not isinstance(elem.tag, str):
            continue
        tag = etree.QName(elem).localname
        text = (elem.text or '').strip()
        if tag == 'p' and text:
            chunks.append((filename, i, main_title, text))
        elif tag == 'note' and text:
            note_type = elem.attrib.get('type', 'note').upper()
            chunks.append((filename, i, main_title, f"[{note_type}] {text}"))
        elif tag == 'title' and text and elem.getparent() is not None and etree.QName(elem.getparent()).localname == 'fig':
            chunks.append((filename, i, main_title, f"[FIGURE] {text}"))
    return chunks


def append_parsed_chunks_to_csv_blob(connection_string: str, container_name: str, folder_path: str, filename: str, new_chunks: List[tuple]) -> None:
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    blob_path = f"{folder_path}/{filename}"
    full_df = pd.DataFrame(columns=["filename", "chunk_id", "main_title", "text"])
    try:
        blob_client = container_client.get_blob_client(blob_path)
        existing_data = blob_client.download_blob().readall().decode("utf-8")
        full_df = pd.read_csv(StringIO(existing_data))
    except ResourceNotFoundError:
        pass

    new_df = pd.DataFrame(new_chunks, columns=["filename", "chunk_id", "main_title", "text"])
    combined_df = pd.concat([full_df, new_df], ignore_index=True)

    csv_buffer = StringIO()
    combined_df.to_csv(csv_buffer, index=False)
    container_client.upload_blob(name=blob_path, data=csv_buffer.getvalue(), overwrite=True)


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
        SimpleField(name="url", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="filename", type=SearchFieldDataType.String, filterable=True, searchable=True, facetable=True, retrievable=True),
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


def delete_docs_by_ids(search_client: SearchClient, ids: List[str]) -> None:
    for i in range(0, len(ids), 1000):
        keys = [{"id": k} for k in ids[i:i+1000]]
        search_client.delete_documents(documents=keys)


def iter_all_index_docs(search_client: SearchClient, select_fields: List[str]):
    results = search_client.search(search_text="*", select=select_fields, top=1000)
    for r in results:
        yield r


def chunk_and_embed_single_file(blob_name, container_client, embedder, embedder_client, search_client, using_embedder=True):
    blob_client = container_client.get_blob_client(blob_name)
    blob_data = blob_client.download_blob().readall()
    filename = os.path.basename(blob_name).lower()

    chunks_raw = parse_dita_xml_blob(blob_data, filename)
    texts = [c[3] for c in chunks_raw]


    existing = get_existing_chunks(search_client, filename)
    current_hashes = {i: compute_sha256(text) for i, text in enumerate(texts)}
    existing_hashes = {v["chunk_id"]: v["content_sha256"] for v in existing.values()}

    if existing_hashes == current_hashes:
        return [], [], "skipped"

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.create_documents(texts)
    if using_embedder:
        vectors = retry_embedding_with_backoff(embedder, [d.page_content for d in split_docs])
    else:
        vectors = embedder_client.embeddings.create(
            model="text-embedding-3-small",
            input=[d.page_content for d in split_docs]
        ).data

    new_ids = set()
    upserts = []
    for i, ((filename, chunk_id, title, text), vec) in enumerate(zip(chunks_raw, vectors)):
        sha = compute_sha256(text)
        doc_id = make_doc_id(filename, chunk_id)
        new_ids.add(doc_id)
        if doc_id in existing and existing[doc_id]["content_sha256"] == sha:
            continue
        upserts.append({
            "id": doc_id,
            "filename": filename,
            "title": title,
            "chunk_id": chunk_id,
            "content": text,
            "content_sha256": sha,
            "last_modified": datetime.utcnow().isoformat() + "Z",
            "url": "", "owner": "", "doc_type": "", "doc_category": "",
            "doc_function": "", "terms": "", "topics": "", "summary": "",
            "content_embedding": vec if using_embedder else vec.embedding
        })

    to_delete = [doc_id for doc_id in existing if doc_id not in new_ids]
    return upserts, to_delete, "changed", chunks_raw


def data_chunk_embed_upload_batch(
    splitter,
    embedder,
    embedder_client,
    connection_string: str,
    container_name: str,
    metadata_df,
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
) -> Dict:
    central_time = datetime.now(timezone("US/Central")).strftime("%Y-%m-%d %H:%M:%S")

    container_client = ContainerClient.from_connection_string(connection_string, container_name)
    all_blobs = list(container_client.list_blobs())
    xml_blobs = [b for b in all_blobs if b.name.lower().endswith(".xml")]

    if blob_subset is not None:
        blob_subset_set = set(blob_subset)
        current_batch = [b for b in xml_blobs if b.name in blob_subset_set]
    else:
        start = batch_number * batch_size
        end = min(start + batch_size, len(xml_blobs))
        current_batch = xml_blobs[start:end]

    search_client = SearchClient(endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
                                 index_name=index_name,
                                 credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY")))

    upserts: List[dict] = []
    deletes: List[str] = []
    added_files = set()
    modified_files = set()
    skipped_files = []
    failed_files = []

    all_chunks_raw = []
    for blob in current_batch:
        try:
            existing_chunks = get_existing_chunks(search_client, blob.name.lower())

            new_docs, to_delete, status, chunks_raw = chunk_and_embed_single_file(
                splitter=splitter,
                embedder=embedder,
                embedder_client=embedder_client,
                container_client=container_client,
                blob_name=blob.name,
                search_client=search_client,
                using_embedder=using_embedder
            )

            all_chunks_raw.extend(chunks_raw)

            if not new_docs and not to_delete:
                skipped_files.append(blob.name)
            elif not existing_chunks and new_docs:
                added_files.add(blob.name)
            elif existing_chunks and (new_docs or to_delete):
                modified_files.add(blob.name)

            upserts.extend(new_docs)
            deletes.extend(to_delete)

        except Exception as e:
            failed_files.append(blob.name)
            continue

    for i in range(0, len(upserts), 1000):
        search_client.merge_or_upload_documents(upserts[i:i + 1000])
    if deletes:
        delete_docs_by_ids(search_client, deletes)

    deleted_files = clean_file_level_deletes_after_batch(connection_string, container_name, search_client)
    
    append_parsed_chunks_to_csv_blob(connection_string=connection_string,
                                    container_name="index-logs",
                                    folder_path="xml-parsing-logs",
                                    filename="parsed_chunks_all_batches.csv",
                                    new_chunks=all_chunks_raw
                                    )
    
    return {
        "timestamp_central": central_time,
        "metadata_df": metadata_df,
        "uploaded_chunks": len(upserts),
        "deleted_chunks": len(deletes),
        "skipped_files": len(skipped_files),
        "added_files": {"count": len(added_files), "files": sorted(added_files)},
        "deleted_files": {"count": len(deleted_files), "files": sorted(deleted_files)},
        "modified_files": {"count": len(modified_files), "files": sorted(modified_files)},
        "failed_files": {"count": len(failed_files), "files": sorted(failed_files)},
    }


def clean_file_level_deletes_after_batch(connection_string: str, container_name: str, search_client: SearchClient) -> List[str]:
    container_client = ContainerClient.from_connection_string(connection_string, container_name)
    blob_filenames = {b.name.lower() for b in container_client.list_blobs()}
    index_filenames = set()
    for doc in iter_all_index_docs(search_client, ["filename"]):
        index_filenames.add(doc.get("filename", "").lower())

    missing_in_blob = [fn for fn in index_filenames if fn and fn not in blob_filenames]
    stale_ids = []
    for fn in missing_in_blob:
        for r in search_client.search(search_text="*", filter=f"filename eq '{fn}'", select=["id"], top=1000):
            stale_ids.append(r["id"])

    if stale_ids:
        delete_docs_by_ids(search_client, stale_ids)
    return missing_in_blob