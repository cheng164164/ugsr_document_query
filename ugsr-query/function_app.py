import azure.functions as func
import logging
import os
import json
from openai import AzureOpenAI
import requests
import re
from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient, BlobClient, generate_blob_sas, BlobSasPermissions
import json
import io
import openpyxl
import pandas as pd


# Load environment variables
AZURE_BLOB_CONN_STRING = os.getenv("AZURE_BLOB_CONN_STRING")
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = "text-embedding-3-small"


# Define mutli indexes names to search
index_names = ["business_index",
               "ugsr_index"
              ]

# Metadata files in blob storage for each index
metadata_files = {
    "business_index": {'container_name': "north-america-business-documents-metadata", 'file_name': "meta_data_business_doc.xlsx"},
    "ugsr_index": {'container_name': "undergroound-engineering-document-metadata", 'file_name': "meta_data_UGSR.xlsx"}
}

# SharePoint URLs for each index
share_point_urls = {
    "business_index": {'name': 'Business Documents North America', "url": "https://globalkomatsu.sharepoint.com/sites/NAGMUSGR00243/SitePages/PublishedDocuments.aspx"},
    "ugsr_index": {'name': 'UGSR Engineering Documents', "url": "https://globalkomatsu.sharepoint.com/sites/NAGMUSGR00221/engres_joy/PPED/JGUEngDocs?viewpath=%2Fsites%2FNAGMUSGR00221%2Fengres%5Fjoy%2FPPED%2FJGUEngDocs"}
}

# Feature On/Off flags
debug_mode = True   # Set to True to enable debug prints
custom_ranking = True   # Set to True to enable custom ranking (vector + keyword); False to use Azure default ranking
dynamic_filtering = False   # Set to True to enable dynamic metadata filtering based on query keywords    
metadata_search = True    # Set to True to enable metadata-only search for relevant queries
use_prev_context = True   # Set to True to enable the feature that uses previous queries as context

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)


def clean_query_for_llm(raw_query, route_keywords={"metadata", "content", "contents"}):
    """
    Removes standalone routing keywords from the query if they are not part of a sentence.
    """
    try:
        # Normalize and split query into clauses using punctuation
        parts = re.split(r'[,.!?;|\n]+', raw_query)
        cleaned = []

        for part in parts:
            stripped = part.strip()
            # If this part is exactly one of the routing keywords, skip it
            if stripped.lower() in route_keywords:
                continue
            # If it's empty or only a keyword, skip
            if not stripped or all(w.lower() in route_keywords for w in stripped.split()):
                continue
            cleaned.append(stripped)

        return '. '.join(cleaned).strip()
    
    except Exception as e:
        logging.error(f"Error cleaning query: {e}")
        return raw_query


def rewrite_query_with_history(current_query, previous_queries):
    """
    Use LLM to rewrite the current query with context from previous queries.
    Ensures vague or reference-based language is clarified.
    """
    if not previous_queries:
        return current_query
    
    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version="2024-12-01-preview"
    )

    prompt = (
        "You are a query rewriter. A user is asking a follow-up question, and you are given some of their previous questions.\n"
        "Your task is to rewrite the current question clearly and fully, incorporating relevant context from previous questions.\n"
        "If the current question includes vague references like 'it' or 'that', make them explicit.\n"
        "Do NOT answer any questions.\n"
        "Return only the rewritten version of the current question.\n\n"
        f"PREVIOUS QUESTIONS:\n{previous_queries}\n\n"
        f"CURRENT QUESTION:\n{current_query}"
    )

    response = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()


def llm_search_query_optimizer(query, previous_queries, use_previous_context):
    system_prompt = f"You are a search query optimizer. \
        Read the user question and extract the key words to generate search queries to improve semantic matching for vector search.\
        Make sure your rewritten query includes all important original terms (do not drop any).\
        but also add relevant synonyms and related descriptions.\
        Remove stopwords.\
        Return a single line of plain text that expands the original query, not a list. Do not return JSON, quotes, or explanations"


    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version="2024-12-01-preview"
    )

    if previous_queries and use_previous_context:
        query_input = f"Context: {previous_queries}\nOriginal query: {query}"
    else:
        query_input = f"Original query: {query}"
    

    response = client.chat.completions.create(
                        model="o3-mini",
                        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query_input}
        ]  
                                )

    optimized_query = response.choices[0].message.content.strip().lower()
    return optimized_query

def get_query_embedding(query):
    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version="2024-12-01-preview"
    )

    embedding_response = client.embeddings.create(
        model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        input=query
    )
    return embedding_response.data[0].embedding

def extract_keywords(query, optimized_query, debug=False):
    original_keywords = set(re.findall(r"\b\w+\b", query.lower()))
    optimized_keywords = set(re.findall(r"\b\w+\b", optimized_query.lower()))
    stopwords = {"how", "can", "the", "is", "what", "my", "i", "you", "your", "to", "a", "and", "in", "of", "for", 
                 "on", "with", "that", "this", "it", "as", "an", "by", "at", "from", "be", "are", "or", "if", "but", 
                 "not", "all", "any", "so", "do", "does", "did", "have", "has", "had", "documents", "document", "documentation", "documentations"}
    intersection_keywords = (original_keywords.intersection(optimized_keywords)) - stopwords
    if debug:
        print(f"Original query: {query.lower()}")
        print(f"Optimized query: {optimized_query.lower()}")
        print(f"Original query keywords: {original_keywords}")
        print(f"Optimized query keywords: {optimized_keywords}")
        print(f"intersection keywords: {intersection_keywords}")
    return intersection_keywords


def normalize_plural_keywords(keywords):
    """
    Normalize plural keywords to match singular variations like 'forms' -> 'form', 'processes' -> 'process'.
    """
    normalized = set(keywords)
    for kw in keywords:
        if kw.endswith("es"):
            normalized.add(kw[:-2])  # handles e.g., "processes" -> "process"
            normalized.add(kw[:-1])  # also handles e.g., "titles" -> "title"
        elif kw.endswith("s"):
            normalized.add(kw[:-1])  # handles e.g., "forms" -> "form"
    return list(normalized)


def fetch_metadata_from_index(headers, index_name, keywords, fields, sample_size=500):
    """
    Fetch a small number of documents from each index to collect known metadata field values.
    """
    metadata_docs = []
    normalized_keywords = normalize_plural_keywords(keywords)  # handle plural
    url = f"{AZURE_SEARCH_ENDPOINT}/indexes/{index_name}/docs/search?api-version=2024-07-01"
    payload = {
        "search": " ".join(normalized_keywords),
        "top": sample_size,
        "queryType": "simple",         # enables keyword search with implicit OR logic
        "searchFields": ",".join(fields),  # restricts scope
        "searchMode": "any"
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        metadata_docs.extend(response.json().get("value", []))
    return metadata_docs, normalized_keywords


def collect_field_values(docs, fields):
    """
    Collect unique values for each metadata field across all documents.
    Returns a dict like { "doc_type": {"form", "procedure"}, ... }
    """
    values_by_field = {field: set() for field in fields}
    for doc in docs:
        for field in fields:
            val = doc.get(field, "").lower()
            if val:
                values_by_field[field].add(val)
    return values_by_field


def generate_or_filter_from_keywords(keywords, field_value_dict):
    """
    Build an OData filter clause using OR logic where keywords match metadata field values.
    """
    clauses = []
    for field, values in field_value_dict.items():
        for val in values:
            for kw in keywords:
                if kw in val:
                    clauses.append(f"{field} eq '{val}'")
                    break
    return " or ".join(clauses)


def generate_field_based_filter(headers, payload, index_name, keywords, filter_fields):
    """
    Use keyword-based OR filter if any metadata fields match and apply to payload.
    """
    metadata_docs, normalized_keywords = fetch_metadata_from_index(headers, index_name, keywords, filter_fields, sample_size=500)
    field_values = collect_field_values(metadata_docs, filter_fields)
    or_data_filter = generate_or_filter_from_keywords(normalized_keywords, field_values)
    if or_data_filter:
        payload["filter"] = or_data_filter


def cosine_similarity(vec1, vec2):
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    return dot / (norm1 * norm2 + 1e-8)


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


def should_use_metadata_search(query):
    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version="2024-12-01-preview"
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a router. Your job is to decide whether a query should be answered by:\n"
                "- metadata: if it asks for listings, filenames, titles, document types, categories, release version, revision date, release data, owner etc.\n"
                "- semantic: if it needs detailed answers from document content.\n"
                "If the query is ambiguous or general, prefer semantic.\n"
                "If the query contains words like 'metadata', choose metadata. If the query contains words like 'content', choose semantic.\n"
                "Only reply with one word: 'metadata' or 'semantic'."
            )
        },
        {
            "role": "user",
            "content": f"Query: {query}"
        }
    ]

    response = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=messages
    )

    decision = response.choices[0].message.content.strip().lower()
    print(f"* Routing decision: {decision} *\n")
    return decision == "metadata"


def metadata_table_by_index(index_names):
    metadata_by_index = {}
    for index_name in index_names:
        if index_name in metadata_files:
            container_name = metadata_files[index_name]['container_name']
            file_name = metadata_files[index_name]['file_name']
            try:
                blob = BlobClient.from_connection_string(AZURE_BLOB_CONN_STRING, container_name, file_name)
                data = blob.download_blob().readall()
            except Exception as e:
                logging.error(f"[Azure] Blob fetch error: {type(e).__name__} - {str(e)}")
                metadata_by_index[index_name] = [{"error": f"Azure Blob error: {type(e).__name__} - {str(e)}"}]
                continue
            
            try:
                df = pd.read_excel(io.BytesIO(data), engine="openpyxl")
                df = df.fillna("").astype(str)
                metadata_by_index[index_name] = df.to_dict(orient="records")
            except Exception as e:
                logging.error(f"[Pandas] Excel parse error: {type(e).__name__} - {str(e)}")
                metadata_by_index[index_name] = [{"error": f"Pandas Excel error: {type(e).__name__} - {str(e)}"}]
        
    return metadata_by_index


'''
def metadata_table_by_index(index_names):
    metadata_by_index = {}
    BLOB_URLS_BY_INDEX = {}
    for index_name in index_names:
        if index_name in metadata_files:
            container_name = metadata_files[index_name]['container_name']
            file_name = metadata_files[index_name]['file_name']
            BLOB_URLS_BY_INDEX[index_name] = generate_blob_sas_url(AZURE_BLOB_CONN_STRING, container_name, file_name)

    for index_name, blob_url in BLOB_URLS_BY_INDEX.items():
        response = requests.get(blob_url)
        if b"<?xml" in response.content or b"AccessDenied" in response.content or b"<Error>" in response.content:
            metadata_by_index[index_name] = [{"error": "Failed to fetch metadata. Access is restricted or URL is invalid."}]
            print("error: ", "Failed to fetch metadata. Access is restricted or URL is invalid.")
            continue
        
        # df = pd.read_excel(io.BytesIO(response.content))
        # df = df.fillna("").astype(str)
        # metadata_by_index[index_name] = df.to_dict(orient="records")
        
        wb = openpyxl.load_workbook(io.BytesIO(response.content), data_only=True)
        sheet = wb.active
        headers = [cell.value for cell in next(sheet.iter_rows(min_row=1, max_row=1))]
        records = []
        for row in sheet.iter_rows(min_row=2, values_only=True):
            record = {headers[i]: str(cell).strip() if cell is not None else "" for i, cell in enumerate(row)}
            records.append(record)
        metadata_by_index[index_name] = records
        
    return metadata_by_index
'''

def summarize_full_metadata(query, previous_queries, metadata_by_index, use_previous_context=True):
    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version="2024-12-01-preview"
    )

    if use_previous_context:     # rewrite current query according to the query history
        rewrited_query = rewrite_query_with_history(query, previous_queries)
    else:
        rewrited_query = query

    index_contexts = []
    for index_name, docs in metadata_by_index.items():
        context_lines = [f"\nIndex: {index_name}"]
        for doc in docs:
            row_desc = "; ".join(f"{k}: {v}" for k, v in doc.items())
            context_lines.append(f"- {row_desc}")
        index_contexts.append("\n".join(context_lines))
    full_context = "\n\n".join(index_contexts)

    messages = []
    messages.append({
        "role": "system",
        "content": (
            "You are a helpful assistant that summarizes document metadata. "
            "The user may ask follow-up questions based on earlier queries."
        )
    })

    # Optional prior user queries (as memory)
    messages.append({
            "role": "user",
            "content": f"Previous queries: {previous_queries}"
        })
    
    messages.append({
            "role": "user",
            "content": (
                f"User current query: {rewrited_query}\n\n"
                f"Here is all available metadata grouped by index:\n\n{full_context}\n\n"
                f"Only answer the current query. Do not answer or repeat previous questions.\n"
                f"If the query mentions a specific index, only summarize that index. Otherwise, summarize all indexes.\n"
                f"list as many relevant documents as possible that match the user query.\n"
                f"If you are not sure about the answer or nothing relevant is found, say 'Sorry, I cannot help with it. Please try looking it up on above share point links'.\n"
                f"Use clear bullet points or sections."
            )
        })

    completion = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=messages
    )
    summary = completion.choices[0].message.content.strip()

    # Append SharePoint links as references
    reference_links = "\n\n\nYou can look up metadata information from the SharePoint.\n\n"+ "**SharePoint Links:**\n"

    for index, values in share_point_urls.items():
        url = values.get("url", "")
        if url:
            reference_links += f"- {values.get('name', 'Unknown')} : {url}\n\n"

    return  f"**Answer:**\n\n{reference_links} \n\n\n\n {summary}"         


def multi_index_search_documents(query, index_names, previous_queries, vector_weight=0.6, top_k=6, dynamic_filtering=True, custom_ranking=True, use_previous_context=True, debug=False):
    """
    Performs hybrid search across multiple indexes.
    Args:
        query (str): The search query.
        index_names (list): List of Azure Search index names.
        vector_weight (float): Weight for vector similarity in final ranking [0.0 - 1.0].
        top_k (int): Number of top documents to return per index.
        debug (bool): If True, print debug output
        custom_ranking (bool): If True, manually calculate final score using vector + keyword; otherwise use Azure's ranking.
    Returns:
        list of documents with relevance scores and index tags.
    """

    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_SEARCH_KEY
    }

    keyword_fields = [
        "title", "doc_type", "doc_category", "doc_function",
        "terms", "topics", "summary", "content"
    ]

    metadata_filter_fields = ["doc_type", "doc_category", "doc_function"]
    vector_field = "content_embedding"
    
    if use_previous_context:     # rewrite current query according to the query history
        rewrited_query = rewrite_query_with_history(query, previous_queries)
    else:
        rewrited_query = query

    optimized_query = llm_search_query_optimizer(query, previous_queries, use_previous_context)
    keywords = extract_keywords(query, optimized_query, debug=debug)
    query_em = get_query_embedding(optimized_query)

    all_results = []
    all_content = []  # Collect all document content for keyword checking
    index_debug_data = {} 

    for index_name in index_names:
        """Retrieves relevant documents from Azure AI Search"""
        select_fields = [
            "filename", "title", "doc_type", "doc_category", "doc_function",
            "terms", "topics", "summary", "content", "owner", "url"
        ]
        if custom_ranking:
            select_fields.append("content_embedding")

        url = f"{AZURE_SEARCH_ENDPOINT}/indexes/{index_name}/docs/search?api-version=2024-07-01"
        payload = {
            "search": rewrited_query.lower(),
            "count": True,
            "top": top_k,
            "select": ",".join(select_fields),
            "searchFields": ",".join(keyword_fields),    
            "queryType": "semantic",  
            "searchMode": "all",
            "semanticConfiguration": "default-semantic",     
            "vectorQueries": [
                {
                    "kind": "vector",
                    "vector": query_em,
                    "fields": vector_field,
                    "k": top_k*2,
                    "exhaustive": False,
                }
            ]
        }

        if dynamic_filtering: 
            # Apply dynamic metadata filtering
            generate_field_based_filter(headers, payload, index_name, keywords, filter_fields=metadata_filter_fields)
            if debug:
                print(f"⚙️ Applied filter for index {index_name}: {payload.get('filter', 'None')}")

        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            hits = response.json().get("value", [])
            for doc in hits:
                keyword_score = doc.get("@search.rerankerScore") or doc.get("@search.score") or 0.0  
                if custom_ranking:
                    embedding = doc.get("content_embedding")
                    vector_score = cosine_similarity(query_em, embedding) if embedding else 0.0
                    norm_keyword_score = keyword_score / 4.0  # Normalize keyword score to 0-1 range
                    final_score = (1 - vector_weight) * norm_keyword_score + vector_weight * vector_score   
                    doc["_vector_score"] = vector_score
                    doc["_final_score"] = final_score
                else:
                    final_score = keyword_score
                    doc["_final_score"] = final_score

                doc["_index"] = index_name
                doc["_keyword_score"] = keyword_score
                doc["_azure_score"] = doc.get("@search.score", 0.0)
                all_results.append(doc)
                all_content.append(doc.get("content", "").lower() + " " + doc.get("summary", "").lower())

            sorted_hits = sorted(hits, key=lambda x: x.get("_final_score", 0), reverse=True)
            if debug:
                index_debug_data[index_name] = sorted_hits[:3]  # Store top 3
        else:
            logging.warning(f"❌ Search failed on {index_name}: {response.status_code} — {response.text}")
    
    if not all_results:
        return []

    # Compute average top-3 final scores per index
    scores_by_index = {}
    for index in index_names:
        top_docs = sorted([d for d in all_results if d["_index"] == index], key=lambda d: d["_final_score"], reverse=True)[:3]
        if top_docs:
            avg_score = sum(d["_final_score"] for d in top_docs) / len(top_docs)
            scores_by_index[index] = avg_score

    sorted_indexes = sorted(scores_by_index.items(), key=lambda x: x[1], reverse=True)
    best_index, best_score = sorted_indexes[0]

    if debug:
        print("\n🔍 Ranked Indexes by Average Final Score:")
        for name, score in sorted_indexes:
            print(f"  - {name}: {score:.4f}")
        print(f"✅ Best index selected: {best_index} (score: {best_score:.4f})\n")
        
        print("\n🔎 Top 3 chunks from each index:")
        for index, docs in index_debug_data.items():
            print(f"\nIndex: {index}")
            for i, doc in enumerate(docs, 1):
                snippet = doc.get("content", "").replace("\n", " ").strip()
                print(
                    f"  {i}. Score: {doc.get('_final_score', 0):.4f}, "
                    f"Keyword: {doc.get('_keyword_score', 0):.4f}, " +
                    (f"Vector: {doc.get('_vector_score', 0):.4f}, " if custom_ranking else "") +
                    f"Azure: {doc.get('_azure_score', 0):.4f}, " +
                    f"Title: {doc.get('title', 'N/A')}\n     Snippet: {snippet}..."
                )   
    # Return top_k results from best index
    final_results = [doc for doc in all_results if doc["_index"] == best_index]
    final_results = sorted(final_results, key=lambda d: d["_final_score"], reverse=True)[:8]  # Return top 8 as final answer

    # --- Check for missing keywords ---
    combined_text = " ".join(all_content)
    missing_keywords = [kw for kw in keywords if kw not in combined_text]

    if missing_keywords:
        warning_msg = f"Keyword search indicates the following words are not found in any relevant documents: '{', '.join(missing_keywords)}'. You can ignore this message or consider rephrasing your question."
    else:
        warning_msg = ""

    # Final results with warning if needed
    final_answer = warning_msg, final_results
    return rewrited_query, final_answer


def multi_index_generate_response(query, context):
    """
    Generates an answer using OpenAI's o3-mini model with the top chunks, and includes a formatted reference section
    with relevance explanations based on each document's content, summary, topics, and terms.
    """
    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version="2024-12-01-preview"
    )

    warning_msg, top_chunks = context
    context_texts = []
    doc_groups  = {}

    for doc in top_chunks:
        filename = doc.get("filename", "N/A")
        content = doc.get("content", "")
        score = doc.get('_final_score', 0)
        context_texts.append(f"[{filename}] {content}")

        if filename not in doc_groups :
            doc_groups [filename] = {
                "document_name": filename,
                "url": doc.get("url", "N/A"),
                "key_contact": doc.get("owner", "N/A"),
                "key_topics": doc.get("topics", ""),
                "key_terms": doc.get("terms", ""),
                "summary": doc.get("summary", ""),
                "scores": [score],
            }
        else:
            doc_groups [filename]["scores"].append(score)
    
    # --- Build context string for LLM to generate main answer ---
    context_str = "\n\n".join(context_texts)
    messages = [
        {
            "role": "user",
            "content": (
                f"Answer the following question using the context from the top relevant documents:\n"
                f"QUESTION: {query}\n\n"
                f"DOCUMENT CHUNKS:\n{context_str}\n\n"
                f"INSTRUCTIONS:\n- Be concise\n- Use only information from the documents. Do not generate answers that don't use the source documents provided.\n"
                # f"- If insufficient information or not sure about the answer, ask clarifying questions instead of directly answering it. \n"
                f"- You may be provided with multiple sources. Read all sources and find the most relavant information to best answer user question.\n"
                f"- If your answer describes a process, include step-by-step instructions and seperate each step by bullet symbol.\n"
                f"- Use plain text with no HTML.\n"
                f"- Separate sections with line breaks."
            )
        }
    ]

    completion = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=messages
    )

    main_answer = completion.choices[0].message.content.strip()
    if warning_msg:
        main_answer = f"Reminder: {warning_msg.strip()}" + "\n\n" + main_answer

    # --- Build reference section ---
    reference_text = "\n\n**References:**\n"

    for doc in list(doc_groups.values())[:3]:
        # Build a focused relevance summary prompt
        relevance_context = (
            f"QUESTION: {query}\n\n"
            f"DOCUMENT SUMMARY: {doc['summary']}\n\n"
            f"KEY TOPICS: {doc['key_topics']}\n\n"
            f"KEY TERMS: {doc['key_terms']}\n\n"
        )

        relevance_prompt = [
            {
                "role": "user",
                "content": (
                    f"Be concise, Summarize in less than 150 words why this document is relevant to the question below, "
                    f"based only on the document's summary, key topics, and key terms. "
                    f"Use bullet points for clarity.\n"
                    f"Do not include HTML, markdown, or field labels.\n"
                    f"Each bullet point should start with a new line."
                    f"Do NOT list the raw fields; only give a clean, concise explanation.\n\n"
                    f"{relevance_context}"
                )
            }
        ]

        rel_response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=relevance_prompt
        )

        relevance_summary = rel_response.choices[0].message.content.strip()
        avg_score = round(sum(doc["scores"]) / len(doc["scores"]), 2)

        reference_text += (
                f"\n---\n"
                f"**Document Title**: {doc['document_name']}\n\n"
                f"**Key Contact**: {doc['key_contact']}\n\n"
                # f" **Similarity Score**: {avg_score}\n\n"
                f"**Relevance**: {relevance_summary}\n\n"
                f"**URL**: {doc['url']}\n\n"
        )

    return f"**Answer:**\n\n{main_answer}{reference_text}"


@app.route(route="multiindexquery")
def multiindexquery(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    try:
        req_body = req.get_json()
        query = req_body.get("query")
        query_history = req_body.get("context")

        if not query:
            return func.HttpResponse("Error: Missing 'query' parameter", status_code=400)

        cleaned_query = clean_query_for_llm(query)  # Clean query by removing routing keywords
        cleaned_query_history = clean_query_for_llm(query_history)
        if metadata_search:
            use_metadata_search_flag = should_use_metadata_search(query)    # use raw query for routing decision
            if use_metadata_search_flag:
                metadata_by_index = metadata_table_by_index(index_names)
                llm_summary = summarize_full_metadata(cleaned_query, cleaned_query_history, metadata_by_index, use_previous_context=use_prev_context)        
                return func.HttpResponse(json.dumps({"answer": llm_summary}, ensure_ascii=False, indent=2), mimetype="application/json", status_code=200)

        # Step 1: Search all indexes
        rewrited_query, docs = multi_index_search_documents(cleaned_query, index_names, cleaned_query_history, vector_weight=0.5, top_k=16, 
                                            dynamic_filtering=dynamic_filtering, custom_ranking=custom_ranking, 
                                            use_previous_context=use_prev_context, debug=debug_mode)
        if not docs:
            return func.HttpResponse("No relevant documents found.", status_code=404)
        
        # Step 2: Generate response from AI with retrieved context
        ai_response = multi_index_generate_response(rewrited_query, docs)

        return func.HttpResponse(json.dumps({"answer": ai_response}, ensure_ascii=False, indent=2), mimetype="application/json", status_code=200)

    except Exception as e:
        return func.HttpResponse(f"Error: {str(e)}", status_code=500)