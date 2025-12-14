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
from langchain_openai import AzureOpenAIEmbeddings
from openai import RateLimitError
import logging
from azure.core.exceptions import ResourceNotFoundError
from typing import List, Dict, Set, Tuple
from io import StringIO
import pandas as pd
from index_creation.config import enable_json_flattening, ENV_VARS
from index_creation.util import set_env_vars


set_env_vars(ENV_VARS)
azure_oai_embedding_deployment = os.getenv('AZURE_EMBEDDING_DEPLOYMENT_NAME', 'text-embedding-3-small')

if azure_oai_embedding_deployment == "text-embedding-3-large":
    EMBEDDING_DIMENSIONS = 3072
else:
    EMBEDDING_DIMENSIONS = 1536

MAX_CHARS_PER_CHUNK = 1000  # adjust as needed


class ChunkCollector:
    def __init__(self, filename, topic_title, max_chars=MAX_CHARS_PER_CHUNK):
        self.filename = filename
        self.topic_title = topic_title
        self.max_chars = max_chars
        self.buffer = ""
        self.chunk_id = 0
        self.chunks = []

    def add_text(self, text):
        if len(self.buffer) + len(text) > self.max_chars:
            self.flush()
        self.buffer += text + "\n"

    def flush(self, force=False):
        if self.buffer.strip() and (force or len(self.buffer) >= self.max_chars):
            self.chunks.append((
                self.filename,
                self.chunk_id,
                self.topic_title,
                self.buffer.strip()
            ))
            self.chunk_id += 1
            self.buffer = ""

    def finalize(self):
        self.flush(force=True)
        return self.chunks



def compute_sha256(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def make_doc_id(file_key: str, chunk_id: int):
    encoded = base64.urlsafe_b64encode(file_key.encode('utf-8')).decode('utf-8').rstrip('=')
    return f"{encoded}-chunk-{chunk_id}"


def retry_embedding_with_backoff(embedder, texts: List[str], max_retries=5):
    delay = 10
    for attempt in range(max_retries):
        try:
            return embedder.embed_documents(texts)
        except RateLimitError as e:
            print(f"⚠️ Rate limit hit. Retry {attempt + 1}/{max_retries} in {delay}s...")
            time.sleep(delay)
            delay *= 2  # Exponential backoff
    raise RateLimitError("❌ Failed after max retries due to open AI rate limiting.")



def get_existing_chunks(search_client: SearchClient, filename: str):
    filename = filename.replace("'", "''")  # Escape single quotes for Azure Search
    results = search_client.search(
        search_text="*",
        filter=f"filename eq '{filename}'",
        select=["id", "chunk_id", "content_sha256"],
        top=1000,
    )
    return {r["id"]: {"chunk_id": r["chunk_id"], "content_sha256": r["content_sha256"]} for r in results}


def xml_element_to_json(element):
    if not isinstance(element.tag, str):
        return None

    node = {
        "tag": etree.QName(element).localname,
        "attributes": {k: v for k, v in element.attrib.items() if k != "class"},
        "text": (element.text or "").strip(),
        "children": []
    }

    for child in element:
        # Full recursive convert
        child_node = xml_element_to_json(child)
        if child_node:
            node["children"].append(child_node)

        # Tail becomes a pseudo-node (NO tag, no attributes)
        tail = (child.tail or "").strip()
        if tail:
            node["children"].append({
                "tag": None,          # IMPORTANT: no tag name!
                "attributes": {},
                "text": tail,
                "is_tail": True,      # marker for flattening logic
                "children": []
            })

    return node


def flatten_json_content(json_node, buffer=None):
    if buffer is None:
        buffer = []

    tag = json_node["tag"]
    text = json_node["text"]
    attrs = json_node["attributes"]
    is_tail = json_node.get("is_tail", False)

    # Inline tags (no labels)
    inline_tags = {"b", "ph", "i", "u", "span", "code", "em", "strong"}

    # 1. HANDLE TAIL NODES
    if is_tail:
        # Append tail text directly (no new line)
        if text:
            buffer.append(text)
        return buffer

    # 2. HANDLE INLINE TAGS
    is_inline = (tag in inline_tags)

    # Label only for block tags
    label = "" if is_inline else f"[{tag.upper()}]" if tag else ""

    # Attributes except class
    attr_info = " ".join([f'{k}="{v}"' for k, v in attrs.items()]) if attrs else ""

    # Build line
    line_parts = [label, text, attr_info]
    line = " ".join(part for part in line_parts if part).strip()

    if line:
        buffer.append(line)

    # 3. RECURSIVE CHILD PROCESSING
    for child in json_node["children"]:
        flatten_json_content(child, buffer)

    return buffer


def extract_xml_flattened_json(root, filename: str, topic_title: str) -> List[Tuple[str, int, str, str]]:
    json_tree = xml_element_to_json(root)
    flat_lines = flatten_json_content(json_tree)

    collector = ChunkCollector(filename, topic_title, max_chars=MAX_CHARS_PER_CHUNK)
    for line in flat_lines:
        collector.add_text(line)
    chunks = collector.finalize()

    return [
        (filename, chunk_id, topic_title, chunk_text)
        for filename, chunk_id, topic_title, chunk_text in chunks
    ]


def extract_xml_text_generalized(elem, filename="", topic_title=""):
    collector = ChunkCollector(filename, topic_title, max_chars=MAX_CHARS_PER_CHUNK)

    def recurse(element, level=0):
        if not isinstance(element.tag, str):
            return

        tag = etree.QName(element).localname
        attrs = " ".join([f'{k}="{v}"' for k, v in element.attrib.items()])
        prefix = "  " * level

        # Include the tag and attributes
        collector.add_text(f"{prefix}<{tag} {attrs}>".strip())

        # Include inner text
        text = (element.text or "").strip()
        if text:
            collector.add_text(f"{prefix}  {text}")

        # Traverse children
        for child in element:
            recurse(child, level + 1)

        # Close tag and tail
        collector.add_text(f"{prefix}</{tag}>")
        tail = (element.tail or "").strip()
        if tail:
            collector.add_text(f"{prefix}  {tail}")

    recurse(elem)
    return collector.finalize()


def parse_dita_xml_blob(blob_data: bytes, filename: str) -> list:
    parser = etree.XMLParser(recover=True)
    try:
        root = etree.fromstring(blob_data, parser=parser)
    except Exception as e:
        raise ValueError(f"XML parsing failed for {filename}: {e}")

    title_elem = root.find('.//{*}title')
    topic_title = title_elem.text.strip() if title_elem is not None else ""

    if enable_json_flattening:
        return extract_xml_flattened_json(root, filename, topic_title)
    else:
        return extract_xml_text_generalized(root, filename, topic_title)


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
        SearchField(name="content_embedding", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, hidden=False, vector_search_dimensions=EMBEDDING_DIMENSIONS, vector_search_profile_name='default'),
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


def chunk_and_embed_single_file(splitter, embedder, embedder_client, container_client, blob_name, search_client, using_embedder=True):
    blob_client = container_client.get_blob_client(blob_name)
    blob_data = blob_client.download_blob().readall()
    filename = os.path.basename(blob_name).lower()

    chunks_raw = parse_dita_xml_blob(blob_data, filename)
    logging.info(f"🔍 Processing file: {filename}, blob size: {len(blob_data)} bytes")
    logging.info(f"✅ Chunks extracted: {len(chunks_raw)}")

    texts = [c[3] for c in chunks_raw]


    existing = get_existing_chunks(search_client, filename)
    current_hashes = {i: compute_sha256(text) for i, text in enumerate(texts)}
    existing_hashes = {v["chunk_id"]: v["content_sha256"] for v in existing.values()}

    if existing_hashes == current_hashes:
        return [], [], "skipped", chunks_raw

    split_docs = splitter.create_documents(texts)
    if using_embedder:
        try:
            vectors = retry_embedding_with_backoff(embedder, [d.page_content for d in split_docs])
        except Exception as e:
            raise RuntimeError(f"Embedding failed for {filename} after retries: {str(e)}")
    else:
        vectors = embedder_client.embeddings.create(
            model="text-embedding-3-small",
            input=[d.page_content for d in split_docs]
        ).data

    if len(vectors) != len(split_docs):
        raise ValueError(f"Mismatch between vectors ({len(vectors)}) and chunks ({len(split_docs)}) for {filename}")
    
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
        total_files = len(current_batch)
        start = 0
        end = total_files
    else:
        start = batch_number * batch_size
        end = min(start + batch_size, len(xml_blobs))
        current_batch = xml_blobs[start:end]
        total_files = len(xml_blobs)

    print(f"📦 [Index: {index_name}] Starting batch {batch_number + 1}/{total_batches or '?'}: processing files {start + 1} to {end} of {total_files}")

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
    for i, blob in enumerate(current_batch):
        print(f"📄 [{index_name}][Batch {batch_number + 1}/{total_batches or '?'}] [{start + i + 1}/{total_files}] Processing: {blob.name}")
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
            logging.error(f"❌ Failed to process {blob.name}: {e}", exc_info=True)
            failed_files.append(blob.name)
            continue

    for i in range(0, len(upserts), 1000):
        search_client.merge_or_upload_documents(upserts[i:i + 1000])
        print(f"✅ Uploaded batch segment {i//1000 + 1} ({min(i+1000, len(upserts))}/{len(upserts)} chunks)")

    if deletes:
        delete_docs_by_ids(search_client, deletes)

    deleted_files = clean_file_level_deletes_after_batch(connection_string, container_name, search_client)
    
    append_parsed_chunks_to_csv_blob(connection_string=connection_string,
                                    container_name="index-logs",
                                    folder_path="xml-parsing-logs",
                                    filename=f"parsed_chunks_{central_time}.csv",
                                    new_chunks=all_chunks_raw
                                    )
    print(f"🎉 Finished [Index: {index_name}] Batch {batch_number + 1}/{total_batches or '?'} — Uploaded: {len(upserts)}, Deleted: {len(deletes)}, Skipped: {len(skipped_files)}, Failed: {len(failed_files)}")
    
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