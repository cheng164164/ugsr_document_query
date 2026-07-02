import azure.functions as func
import logging
import os
import concurrent.futures
import json
from openai import AzureOpenAI
import requests
import re
from azure.storage.blob import BlobClient
import json
import io
import openpyxl
import pandas as pd
from .config import ENV_VARS, index_names, metadata_files, share_point_urls, supplement_files, feature_flags, chatbot_name
from .util import (extract_structured_filenames, set_env_vars, strip_doc_references, title_case_filename, title_case_name, resolve_reference_url, 
                   resolve_reference_name, truncate_history,tokenizer, extract_structured_filenames, append_images_to_answer,
                   generate_blob_sas_url)
import numpy as np

set_env_vars(ENV_VARS)
# Load environment variables
AZURE_BLOB_CONN_STRING = os.getenv("AZURE_BLOB_CONN_STRING")
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

def clean_query_for_llm(raw_query, route_keywords={"metadata", "content", "contents"}):
    """
    Cleans a single query or a query history string separated by '|'.
    - If input contains '|', split into segments by '|', clean each, then rejoin.
    - If input is a single query, clean it directly.
    """
    try:
        if not raw_query or not isinstance(raw_query, str):
            return ""

        def clean_segment(segment: str) -> str:
            # Split on sentence punctuation only, not pipes
            parts = re.split(r'[,.!?;\n]+', segment)
            cleaned = []
            for part in parts:
                stripped = part.strip()
                if not stripped:
                    continue
                # Skip if it's exactly or only routing keywords
                if stripped.lower() in route_keywords:
                    continue
                if all(w.lower() in route_keywords for w in stripped.split()):
                    continue
                cleaned.append(stripped)
            return ".".join(cleaned).strip()

        if "|" in raw_query:
            # Process history: split by '|', clean each, rejoin
            segments = [s.strip() for s in raw_query.split("|") if s.strip()]
            cleaned_segments = [clean_segment(seg) for seg in segments if seg]
            return "|".join(cleaned_segments)
        else:
            # Process single query
            return clean_segment(raw_query)

    except Exception as e:
        logging.error(f"Error cleaning query: {e}")
        return raw_query if raw_query else ""


def decompose_query(query: str, debug: bool = False) -> list[str]:
    prompt = f"""
    You are an expert assistant that decomposes complex multi-part questions into smaller, independent sub-questions — **but only when it is clearly necessary**.

    Your behavior:
    - Most user inputs are *single* question or short phrases. For those, you must **not** change anything or add any extra text to the query.
    - If the input is already a single focused question or a short phrase, return a JSON array with exactly one element, which is the original query copied character-for-character.
    - Only decompose into multiple sub-questions if it is **obviously** composed of multiple distinct questions or steps.

    Decomposition rules (when it *is* clearly multi-part):
    - Break it into multiple sub-questions if the question clearly involves multiple steps, tasks, or clauses joined with words like “and”, “then”, “first... next...”, etc.
    - Multiple WH-words ("what, how, who, why, when, where, which") can indicate separate sub-questions.
    - A question mark (?) usually indicates the end of a question; multiple question marks suggest multiple questions.
    - If the concepts are tightly related and naturally part of one question (e.g. “risks and mitigation strategies for cloud migration”), do **not** split them.
    - When you do split, each sub-question must be fully self-contained.
    - Avoid pronouns like "these", "those", "they", or "it" that depend on earlier context; restate the referenced concept explicitly.
    - Do **not** expand, rephrase, summarize, or add explanations to any question text. Keep each sub-question minimal and as close as possible to the original wording.

    Formatting:
    - Respond **only** with a JSON array of strings, no markdown, no comments, no extra keys.
    - For single/atomic inputs, the array must be: ["<original user text>"] with the text copied exactly.

    Examples:

    Input: "What are the risks and mitigation strategies for cloud migration?"
    → ["What are the risks and mitigation strategies for cloud migration?"]

    Input: "Who is the design owner for GMNR Auth Group and what is the design owner responsible for global controlled design?"
    → ["Who is the design owner for GMNR Auth Group?",
        "What is the design owner responsible for in global controlled design?"]

    Input: "How do I perform a software release test?"
    → ["How do I perform a software release test?"]

    Input: "First I want to extract the data, then clean it, and finally store it."
    → ["How do I extract the data?",
        "How do I clean the data?",
        "How do I store the data?"]

    Now process this input. Remember:
    - If it is not clearly multi-part, return a **single-element array** with the original text exactly.
    - Otherwise, decompose as described above.

    User question: "{query}"
    """.strip()

    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version="2024-12-01-preview"
    )

    try:
        response = client.chat.completions.create(
            model="o4-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        raw_output = response.choices[0].message.content.strip()
        
        # --- CLEAN CODE FENCES ---
        cleaned = raw_output

        # Remove ```json ... ```
        cleaned = re.sub(r"^```json\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()

        # Remove generic ```
        cleaned = re.sub(r"^```\s*|\s*```$", "", cleaned, flags=re.DOTALL).strip()

        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logging.warning(f"Decomposed query raw result: {raw_output!r}")
            return [query]  # Fallback to original

        if debug:
            print(f"Decomposed query result: {result}")

        if isinstance(result, list) and all(isinstance(item, str) for item in result) and result:
            return result
        else:
            return [query]
    except Exception as e:
        logging.warning(f"❌ Failed to decompose query: {e}")
        return [query]
    

def filter_relevant_history(current_query, query_history, answer_history):
    """
    Filters relevant user-bot turns from chat history based on the current query.
    Returns a list of relevant exchanges (dicts with 'user' and 'bot').
    """
    if not current_query.strip():
        return ""

    try:
        queries = [q.strip() for q in query_history.split("|") if q.strip()]
        answers = [a.strip() for a in answer_history.split("|") if a.strip()]

        history_turns = []
        max_len = max(len(queries), len(answers))

        for i in range(max_len):
            user = queries[i] if i < len(queries) else None
            bot = answers[i] if i < len(answers) else None
            turn_number = i+1
            turn_lines = [f"Turn {turn_number}:"]
            if user and bot:
                turn_lines.append(f"User: {user}\nBot: {bot}")
            elif user:
                turn_lines.append(f"User: {user}")
            elif bot:
                turn_lines.append(f"Bot: {bot}")

            history_turns.append("\n".join(turn_lines))

        if not history_turns:
            return ""

        client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version="2024-12-01-preview"
        )

        history_turns = truncate_history(history_turns)
        transcript = "\n\n".join(history_turns)

        prompt = (
            "You are an assistant that filters past chat history to keep only what is useful for understanding the current question.\n"
            "Each turn may include a user question, a bot answer, or both.\n"
            "Your task is to return only the relevant items from the history that help clarify or add context to the current question.\n"
            "Ignore unrelated entries.\n\n"
            "If the CURRENT QUESTION appears vague (e.g., contains 'it' or 'that'), lacking of context and reference. In this case, "
            "prioritize the most recent turns (e.g., Turn 5 is newer than Turn 1) that might clarify those references, and ignore older, unrelated entries.\n"
            f"CURRENT QUESTION:\n{current_query}\n\n"
            f"PAST HISTORY:\n{transcript}"
        )

        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        logging.warning(f"⚠️ Failed to filter relevant history: {e}")
        return ""
    
def rewrite_query_with_history(current_query, relevant_history_text):
    """
    Rewrites the current query using relevant chat history for clarity.
    """
    if not relevant_history_text or not current_query.strip():
        return current_query

    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version="2024-12-01-preview"
    )

    system_prompt = (
        "You are a smart query rewriter that creates a clear and self-contained version of a user's intent.\n"
        "You should never assume the user is referring to video games, pop culture, or unrelated general knowledge unless it's clearly stated. Focus strictly on mechanical, industrial, or Komatsu-related topics."
        "The user may respond with clarifying statements, follow-up questions, or additional details.\n"
        "You are given:\n"
        "- The current user input (which may be a question or clarification)\n"
        "- Relevant prior conversation turns (user and assistant)\n"
        "The assistant may have previously asked the user to clarify their question,\n"
        "so the latest user message may be a direct clarification of an earlier vague or incomplete query.\n"
        "If you found the latest user message is a new query/topic which is not about clarification or irrelevant to prior conversation turns, then simply skip rewriting and return the original query exactly word by word."
        "Otherwise, your task is to synthesize all of this context into a single rewritten query that:\n"
        "- Clearly expresses the user's intended question\n"
        "- Resolves any vague references (e.g., 'this', 'it', 'that', 'these', 'those', 'they', 'the one')\n"
        "- Incorporates relevant details and clarifications from the current and previous turns\n"
        "- Is suitable for retrieval or search\n"
        "Do NOT answer the question or include chat history in the output.\n"
        "Only return the rewritten query or orignal query."
    )

    user_prompt = (
        f"RELEVANT HISTORY:\n{relevant_history_text}\n\n"
        f"CURRENT QUESTION:\n{current_query}"
    )

    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        return response.choices[0].message.content.strip()
    
    except Exception as e:
        logging.warning(f"Query rewrite failed: {e}")
        return current_query


def llm_search_query_optimizer(query):
    """
    Turn a natural-language question into a short, search-optimized query.
    General behavior:
    - Remove 'how do I', 'what is', 'can you', etc.
    - Keep key nouns/phrases (skills, competencies, topics).
    - Add a few related terms, but stay short.
    """
    system_prompt = (
        "You are a search query normalizer.\n"
        "Input is a natural language question.\n"
        "You must output a SINGLE, short search query that captures the main subject.\n"
        "Rules:\n"
        "- Remove helper phrases like 'how do I', 'what is', 'can you', 'please help me with'.\n"
        "- Keep important nouns and noun phrases (skills, competencies, topics, tools).\n"
        "- It's OK to add 1-3 related words (synonyms or context like 'course', 'training', 'resources') if helpful.\n"
        "- Do NOT return a full sentence.\n"
        "- Do NOT add explanations, JSON, or quotes. Return plain text only."
    )

    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version="2024-12-01-preview"
    )

    resp = client.chat.completions.create(
        model="o3-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {query}"}
        ]
    )
    canonical = resp.choices[0].message.content.strip()
    return canonical


def embed_query_for_routing(query: str):
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version="2024-12-01-preview"
    )
    resp = client.embeddings.create(
        model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        input=query
    )
    return np.array(resp.data[0].embedding, dtype=np.float32)


def select_indexes_by_cluster_embeddings(query, cluster_embeddings, top_n=1):
    try:
        qvec = embed_query_for_routing(query)

        sims = []
        for index, vec in cluster_embeddings.items():
            vec_np = np.array(vec, dtype=np.float32)
            cos_sim = np.dot(qvec, vec_np) / (
                np.linalg.norm(qvec) * np.linalg.norm(vec_np)
            )
            sims.append((index, cos_sim))

        sims.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in sims[:top_n]]

    except Exception as e:
        logging.error(f"Embedding-based index routing failed: {e}")
        return []


def select_relevant_indexes_via_llm(query, index_metadata_summaries: dict, top_n: int = 2) -> list[str]:
    """
    Use LLM to select the top N most relevant indexes for the given query based on metadata summaries.
    """
    if not index_metadata_summaries:
        return []

    client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version="2024-12-01-preview"
        )
    
    prompt = f"""
            You are a document routing assistant.
            Your task is to evaluate the user's question and determine which document indexes are **most likely** to contain information related to that question, based on metadata summaries of each index.
            The index metadata summaries provide a brief overview of the topics, terms. Keywords matching is good indicator of relevance, but also consider related concepts and synonyms. 
            Important:
            - You are **not** answering the user's question.
            - You are **not** checking if the answer is guaranteed to exist.
            - You are simply ranking which indexes are most likely to be useful based on their description and scope.
            - You must assign a score between 1 and 10 to each index (10 = highly relevant, 1 = not relevant).
            - Return the top 1 or 2 indexes based on these scores.

            User query:
            \"{query}\"

            Available indexes and their metadata summaries:
            """
    for index_name, summary in index_metadata_summaries.items():
        prompt += f"- {index_name}: {summary}\n"

    prompt += """
            Return a JSON object with the following structure:

            {
            "ranked": [["index_name", score], ...],
            "selected": ["top_index", "optional_second_index_if_close"]
            }

            "ranked" is a list of all indexes with their scores, sorted from highest to lowest score. Even the highest score is 10, all indexes should be included in the "ranked" list. 
            Only include the second index in \"selected\" if its score is at least 70% of the top one.
            Do not return an empty list. Use only the index names provided above.
            Make sure to vary the scores meaningfully. Do not assign all indexes the same score.
            """
    
    try:
        response = client.chat.completions.create(
        model="o4-mini",
        messages=[
        {"role": "system", "content": "You are a helpful assistant that selects document indexes based on metadata relevance."},
        {"role": "user", "content": prompt}
        ]
        )

        content = response.choices[0].message.content.strip()
        parsed = json.loads(content)

        selected = parsed.get("selected", [])
        ranked = parsed.get("ranked", [])

        logging.info("📊 LLM index ranking scores: " + "; ".join([f"{name}: {score}" for name, score in ranked]))
        valid = [idx for idx in selected if idx in index_metadata_summaries]
        if not valid:
            logging.warning("⚠️ LLM returned no valid selected indexes, falling back to top 2.")
            return list(index_metadata_summaries.keys())[:2]

        return valid[:2]

    except Exception as e:
        logging.warning(f"⚠️ LLM index selection failed: {e}")
        return []  # Fallback


def merge_routing_signals(embedding_results, sampling_results):
    """
    embedding_results: list of 2 indexes (always)
    sampling_results: list of up to 2 indexes
    """
    # Normalize
    E = embedding_results or []
    S = sampling_results or []

    # Not enough embedding data
    if len(E) == 0:
        return S
    if len(E) == 1:
        return list(dict.fromkeys([E[0]] + S))

    e1, e2 = E[0], (E[1] if len(E) > 1 else None)
    s1 = S[0] if len(S) > 0 else None

    # --------------------------------------------
    # RULE 1 — Double confirmation (strongest)
    # --------------------------------------------
    if s1 and e1 == s1:
        return [e1]

    # --------------------------------------------
    # RULE 2 — Overlap between embeddings and sampling
    # --------------------------------------------
    overlap = set(E) & set(S)
    if overlap:
        return [idx for idx in E[:2]]

    # --------------------------------------------
    # RULE 3 — No overlap → use E[0], E[1], S[0]
    # --------------------------------------------
    if s1:
        return [e1, e2, s1]

    # If sampling empty, fallback to embedding only
    return [e1, e2]



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
                f"You are a router. Your job is to decide whether a query should be answered by:\n"
                f"- contact: questions asking for point of contact, contact person, technical expert, representative, owner/backup of a BUSINESS AREA / PRODUCT / APPLICATION (from contact lists)\n"
                f"- metadata: Only if it asks for listing of metadata terms such as filenames, titles, document types, categories, release version, revision date, release data, document owner(s) of the documents in the resources etc.\n"
                f"- semantic: If the query asks for listing of terms outside of above metadata terms, or it needs detailed answers or summaries from document content.\n"
                f"- general: If the query asks about the general topics, scopes, functionalities, capabilities or logistic questions of the chatbot." 
                f"For example general questions like 'what can you help with?', 'what is {chatbot_name}', 'how can i use {chatbot_name}?', 'what are the questions that you can answer?'etc.\n"
                f"If the query contains words like 'metadata', choose metadata. If the query contains words like 'content', choose semantic.\n"
                f"Only reply with one word: 'contact' or 'metadata' or 'semantic' or 'general'."
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
    return decision


def llm_context_guard_check(query, context_text, client, deployment=AZURE_OPENAI_DEPLOYMENT, strict_mode=False):
    """
    Uses LLM to confirm whether the provided context actually answers the user's query.
    Returns:
        is_valid (bool)
        explanation (str)
        is_completely_irrelevant (bool)
        selected_doc_tags (list[str])   <-- (LLM-selected DocN tags)
    """

    if strict_mode: 
        system_msg = {
            "role": "system",
            "content": (
                "You are a validation agent. Your job is to decide if the provided CONTEXT truly answers the USER QUESTION.\n"
                "You should never assume the user is referring to video games, pop culture, or unrelated general knowledge unless it's clearly stated. Focus strictly on mechanical, industrial, or Komatsu-related topics.\n"
                "Be strict. If the context uses different terms, systems, or services than the question, reply 'no'.\n"
                "Only consider exact term matches. \n"
                "Do not infer or guess intent beyond what the context supports. Do NOT introduce unrelated interpretations of words based on common alternative meanings.\n"
                "Only consider what is explicitly stated in the context.\n"
                "Your reply must start with 'yes' or 'no'. Then give a brief reason why.\n"
                "If your answer is 'no', also include a short summary (1–2 sentences) of what the context is actually about — but DO NOT mention any unrelated definitions or meanings of the words.(eg:, cranes)\n"
                "For example, if the USER QUESTION mentions 'crane', and the context is about lifting equipment, do not mention bird species.\n"
                "The purpose of explanation is also to help guide the user toward a more appropriate query.\n"
                "If the query partially matches certain keywords in the CONTEXT, prompt the user for clarification, but ONLY based on the meaning used within the CONTEXT.\n"
                "If answer is 'no', end the short summary by asking: 'Would you like to clarify your question?'"
                "Do not reference any specific document in your explanation. Your explanation should be general and should not mention Doc1, Doc2, etc., even if they appear in the context."
            )
        }

    else:
        system_msg = {
            "role": "system",
            "content": (
                "You are a supportive assistant helping users determine if their query can be answered with the available CONTEXT.\n"
                "You should never assume the user is referring to video games, pop culture, or unrelated general knowledge unless it's clearly stated. Focus strictly on mechanical, industrial, or Komatsu-related topics.\n"
                "Be flexible — if the query is somehow related to the topic of context, or any keyword or phrase partially match, answer 'yes'.\n"
                "Only when the query is totally unrelated with the topic of context, answer 'no' with a brief explanation and suggest the closest match if possible.\n"
                "Always try to assist even with keywords or partial match.\n"
                "Reply must start with 'yes' or 'no'. If answer is 'no', end the explanation by asking: 'Would you like to clarify your question?'.\n"
                "Do not reference any specific document in your explanation. Your explanation should be general and should not mention Doc1, Doc2, etc., even if they appear in the context."
            )
        }

    user_msg = {
        "role": "user",
        "content": (
            f"USER QUESTION: {query}\n\n"
            f"CONTEXT: {context_text}\n\n"
        )
    }
    response = client.chat.completions.create(
        model=deployment,
        messages=[system_msg, user_msg]
    )
    answer = response.choices[0].message.content.strip().lower()
    is_valid = answer.startswith("yes")

    # Second: determine if the question is completely irrelevant
    is_completely_irrelevant = False
    selected_doc_tags = []

    if not is_valid:
        irrelevance_messages =[{
                "role": "user",
                "content": (
                    f"Determine whether the following question is completely unrelated to the provided context.\n"
                    f"Only perform this check if the question cannot be answered using the context.\n"
                    f"If the question is at least somewhat related to the context, respond with RELEVANT.\n"
                    f"If it is completely off-topic or unrelated, respond with IRRELEVANT.\n"
                    f"QUESTION: {query}\n"
                    f"CONTEXT:{context_text}\n"
                    f"INSTRUCTIONS: \n"
                    f" - If at least partial of keywords or terms match between the user query and provded context, then consider it as RELEVANT. \n" 
                    f" - Do not classify a question as IRRELEVANT just because it contains words that have other meanings.\n"
                    f" - Reply with one word only: RELEVANT or IRRELEVANT."
                )
            }
        ]        

        irrelevance_response = client.chat.completions.create(
            model=deployment,
            messages=irrelevance_messages
        )

        relevance_tag = irrelevance_response.choices[0].message.content.strip().upper()
        is_completely_irrelevant = relevance_tag == "IRRELEVANT"

    return is_valid, answer, is_completely_irrelevant


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
                if file_name.endswith(".csv"):
                    df = pd.read_csv(io.BytesIO(data))
                elif file_name.endswith(".xlsx") or file_name.endswith(".xls"):
                    df = pd.read_excel(io.BytesIO(data), engine="openpyxl")
                else:
                    raise ValueError("Unsupported file type")

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

def summarize_metadata_per_index(index_name, query, relevant_history_text, docs):
    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version="2024-12-01-preview"
    )

    metadata_lines = []
    for doc in docs:
        metadata_lines.append(
                            f"- Name: {doc.get('Name', '')}\n"
                            f" - Title: {doc.get('Title', '')}\n"
                            f" - Doc Type: {doc.get('Doc Type', '')}\n"
                            f" - Doc Category: {doc.get('Doc Category', '')}\n"
                            f" - Function: {doc.get('Function', '')}\n"
                            f" - Owner: {doc.get('Document Owner(s)', '')}\n"
                            f" - Version: {doc.get('version', '')}\n"
                            f" - Publish Date: {doc.get('publish date', '')}\n"
                            f" - URL: {doc.get('url', '')}\n"
                            )
    metadata_text = "\n".join(metadata_lines)
    history_prefix = (
        f"RELEVANT CHAT HISTORY:\n{relevant_history_text}\n\n"
        if relevant_history_text else ""
    )

    prompt = (
        f"You are an assistant that summarizes document metadata.\n"
        f"{history_prefix}"
        f"USER QUERY:\n{query}\n\n"
        f"DOCUMENT METADATA:\n{metadata_text}\n\n"
        f"Instructions:\n"
        f"- Use history only if it helps clarify the current query.\n"
        f"-Only answer the current query. Do not answer or repeat previous questions.\n"
        f"-If the query mentions a specific index name or library, only summarize for that index. if index name does not match, simply skip the search'.\n"
        f"-Only need to return 10 items at most in the response. If found 10 items, no more search is needed.\n"
        f"-If no relevant documents are found, respond with:'(No relevant metadata found in this index.)'\n"        
        f"-Use clear bullet points or sections."
    )
    
    completion = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}]
    )
    summary = completion.choices[0].message.content.strip()

    return   f"---\📚 Index: {share_point_urls[index_name]['name']}\n\n {summary}\n---"         


def summarize_full_metadata(query, relevant_history_text, metadata_by_index, parallel=True):
    """
    Summarizes metadata from multiple indexes. Supports parallel or sequential execution.
    Skips LLM combination if only one index is present.
    """
    # Add SharePoint reference section in the end of the response
    reference_links = "\n\n\nMore metadata information can be found from the SharePoint.\n\n**SharePoint Links:**\n"
    for index, values in share_point_urls.items():
        name = values.get("name", "Unknown")
        url = values.get("url", "")
        if url:
            reference_links += f"- [{name}]({url})\n\n"

    index_docs = [(index_name, docs) for index_name, docs in metadata_by_index.items() if docs]
    if not index_docs:
        return f"Sorry, no relevant metadata found.\n\n\n\n{reference_links}"

    if parallel and len(index_docs) > 1:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                        executor.submit(summarize_metadata_per_index, idx, query, relevant_history_text, docs)
                        for idx, docs in index_docs
                        ]
            all_responses = [f.result() for f in futures]
    else:
        all_responses = [
        summarize_metadata_per_index(idx, query, relevant_history_text, docs)
        for idx, docs in index_docs
        ]

    if len(all_responses) == 1:
        return f"**Answer:**\n\n{all_responses[0]}\n\n\n\n{reference_links}"  # No need to call LLM to summarize combined results

    combined_summary = summarize_combined_metadata_results(query, index_docs, all_responses)

    return f"**Answer:**\n\n{combined_summary} \n\n\n\n{reference_links}"


def summarize_combined_metadata_results(query, index_docs, raw_summaries: list):
    non_empty = [
        s for s in raw_summaries
        if "(No relevant metadata found" not in s and s.strip()
    ]

    if not non_empty:
        return "Sorry, I cannot help with it. Please try looking it up on the SharePoint links."

    displayed_names = []  # List of SharePoint display names for the user
    for (idx, _) in index_docs:
        displayed_names.append(share_point_urls[idx]["name"])

    combined_input = "\n\n".join(raw_summaries)

    header = (
        f"Results include documents from the following indexes: {', '.join(displayed_names)}\n\n"
        "Below is the combined list of results, grouped by index:\n\n"
    )

    system_prompt = (
        "You are a metadata summarization assistant.\n\n"
        "CRITICAL RULES:\n"
        "- You must NOT alter, rename, or remove any library headings.\n"
        "- The headings look like: '📚 Index: <SharePoint Name>'. Leave them exactly as they appear.\n"
        "- You MUST NOT introduce new library names.\n"
        "- You MUST NOT generate an introduction or explanation.\n"
        "- DO NOT reorder library sections.\n"
        "- DO NOT restate library names.\n"
        "- ONLY refine and clean the bullet lists under each existing heading.\n"
        "- DO NOT generate new bullets not based on raw results.\n"
        "- DO NOT add any lines outside the blocks.\n"
        "- Maintain section separators ('---') as provided.\n\n"
        "Do NOT add any other text."
    )

    user_prompt = (
        f"USER QUERY:\n{query}\n\n"
        f"RAW METADATA RESULTS:\n{combined_input}\n\n"
        "**Generate ONLY the grouped summary body. DO NOT write any introduction.**"
    )

    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version="2024-12-01-preview"
    )

    completion = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    llm_body = completion.choices[0].message.content.strip()
    return header + llm_body



def multi_index_search_documents(query, rewrited_query, index_names, vector_weight=0.6, top_k=6, 
                                 dynamic_filtering=True, keywords_matching=True, custom_ranking=True, 
                                 use_previous_context=True, parallel=True, debug=False):
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_SEARCH_KEY
    }

    keyword_fields = ["filename", "title", "doc_type", "doc_category", "doc_function",
                      "terms", "topics", "summary", "content"]
    
    metadata_filter_fields = ["doc_type", "doc_category", "doc_function"]
    vector_field = "content_embedding"

    filename_keywords = extract_structured_filenames(query)
    optimized_query = llm_search_query_optimizer(rewrited_query)
    keywords = extract_keywords(query, optimized_query, debug=feature_flags["debug_mode"]) if keywords_matching else None
    query_em = get_query_embedding(optimized_query)

    if debug:
        print(f"🔍 Optimized search query: {optimized_query}")

    all_results = []
    all_content = []
    index_debug_data = {}

    def search_index(index_name):
        if debug:
            print(f"🔎 Starting search on index: {index_name}")
        select_fields = [
            "filename", "title", "doc_type", "doc_category", "doc_function",
            "terms", "topics", "summary", "content", "owner", "url"
        ]
        if custom_ranking:
            select_fields.append("content_embedding")

        ## Use filename keywords for keyword search if available to enhance chunk retrieval, otherwise use the rewrited query
        if filename_keywords:
            search_terms = " ".join(filename_keywords)
        else:
            search_terms = optimized_query.lower()
        
        url = f"{AZURE_SEARCH_ENDPOINT}/indexes/{index_name}/docs/search?api-version=2024-07-01"
        payload = {
            "search": search_terms,
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
                    "k": top_k * 2,
                    "exhaustive": False,
                }
            ]
        }

        if dynamic_filtering and keywords_matching:
            generate_field_based_filter(headers, payload, index_name, keywords, filter_fields=metadata_filter_fields)
            if debug:
                print(f"⚙️ Applied filter for index {index_name}: {payload.get('filter', 'None')}")

        response = requests.post(url, headers=headers, json=payload)
        index_results = []
        index_content = []
        debug_data = []

        if response.status_code == 200:
            hits = response.json().get("value", [])
            if debug:
                print(f"✅ Index {index_name} returned {len(hits)} documents")
            for doc in hits:
                keyword_score = doc.get("@search.rerankerScore") or doc.get("@search.score") or 0.0
                if custom_ranking:
                    embedding = doc.get("content_embedding")
                    vector_score = cosine_similarity(query_em, embedding) if embedding else 0.0
                    norm_keyword_score = keyword_score / 4.0
                    final_score = (1 - vector_weight) * norm_keyword_score + vector_weight * vector_score
                    doc["_vector_score"] = vector_score
                    doc["_final_score"] = final_score
                else:
                    final_score = keyword_score
                    doc["_final_score"] = final_score

                doc["_index"] = index_name
                doc["_keyword_score"] = keyword_score
                doc["_azure_score"] = doc.get("@search.score", 0.0)
                index_results.append(doc)
                index_content.append(doc.get("content", "").lower() + " " + doc.get("summary", "").lower())

            sorted_hits = sorted(hits, key=lambda x: x.get("_final_score", 0), reverse=True)
            debug_data = sorted_hits[:]
        else:
            logging.warning(f"❌ Search failed on {index_name}: {response.status_code} — {response.text}")

        return index_results, index_content, index_name, debug_data

    def run_parallel_batch(index_list):
        results, contents, debug_data = [], [], {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(search_index, index): index for index in index_list}
            for future in concurrent.futures.as_completed(futures):
                index_name = futures[future]
                try:
                    index_results, index_content, _, debug = future.result()
                    results.extend(index_results)
                    contents.extend(index_content)
                    debug_data[index_name] = debug
                except Exception as e:
                    logging.error(f"❌ Exception in thread for index '{index_name}': {e}")
        return results, contents, debug_data

    def run_sequential(index_list):
        results, contents, debug_data = [], [], {}
        for index in index_list:
            index_results, index_content, _, debug = search_index(index)
            results.extend(index_results)
            contents.extend(index_content)
            debug_data[index] = debug
        return results, contents, debug_data

    # --- Run depending on toggle ---
    if parallel:
        if debug:
            print("🚀 Running in two-batch parallel mode")
        midpoint = len(index_names) // 2
        batch1 = index_names[:midpoint]
        batch2 = index_names[midpoint:]

        results1, content1, debug1 = run_parallel_batch(batch1)
        results2, content2, debug2 = run_parallel_batch(batch2)

        all_results = results1 + results2
        all_content = content1 + content2
        index_debug_data.update(debug1)
        index_debug_data.update(debug2)

    else:
        if debug:
            print("🐢 Running in sequential mode")
        all_results, all_content, index_debug_data = run_sequential(index_names)

    if not all_results:
        return []

    # Score & sort
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
                    f"Title: {doc.get('filename', 'N/A')}\n     Snippet: {snippet}..."
                )

    final_results = [doc for doc in all_results if doc["_index"] == best_index]
    final_results = sorted(final_results, key=lambda d: d["_final_score"], reverse=True)[:top_k]

    warning_msg = ""
    if keywords_matching:
        combined_text = " ".join(all_content)
        missing_keywords = [kw for kw in keywords if kw not in combined_text]
        if missing_keywords:
            warning_msg = f"Keyword search indicates the following words are not found in any relevant documents: '{'; '.join(missing_keywords)}'. You can ignore this message or consider rephrasing your question."

    return warning_msg, final_results
    

'''
def multi_index_search_documents(query, rewrited_query, index_names, vector_weight=0.6, top_k=6, 
                                 dynamic_filtering=True, keywords_matching=True, custom_ranking=True, 
                                 use_previous_context = True, debug=False):
    """
    Performs hybrid search across multiple indexes.
    Args:
        query (str): The search query.
        index_names (list): List of Azure Search index names.
        vector_weight (float): Weight for vector similarity in final ranking [0.0 - 1.0].
        top_k (int): Number of top documents to return per index.
        dynamic_filtering (bool): If True, apply dynamic metadata filtering based on query keywords.
        keywords_matching (bool): If True, check if keywords are present in retrieved documents and warn if missing.
        custom_ranking (bool): If True, manually calculate final score using vector + keyword; otherwise use Azure's ranking.
        use_previous_context (bool): If True, rewrite current query using previous queries as context.
        debug (bool): If True, print debug output
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
    
    optimized_query = llm_search_query_optimizer(query, rewrited_query, use_previous_context)
    keywords = extract_keywords(query, optimized_query, debug=feature_flags["debug_mode"]) if keywords_matching else None
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

        if dynamic_filtering and keywords_matching: 
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

    warning_msg = ""
    if keywords_matching:
        # --- Check for missing keywords ---
        combined_text = " ".join(all_content)
        missing_keywords = [kw for kw in keywords if kw not in combined_text]

        if missing_keywords:
            warning_msg = f"Keyword search indicates the following words are not found in any relevant documents: '{'; '.join(missing_keywords)}'. You can ignore this message or consider rephrasing your question."
        
    # Final results with warning if needed
    final_answer = warning_msg, final_results
    return final_answer
'''

def multi_index_generate_response(query, context, 
                                  hide_ref_relevance, 
                                  hide_ref_contact, 
                                  show_image, 
                                  show_title_in_ref,
                                  strict_mode=False):
    from collections import OrderedDict
    import re

    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version="2024-12-01-preview"
    )

    warning_msg, top_chunks = context
    context_texts = []
    doc_groups = OrderedDict()
    tag_lookup = {}  # Maps doc tag to filename

    for idx, doc in enumerate(top_chunks):
        tag = f"Doc{idx + 1}"
        filename = doc.get("filename", "N/A")
        title = doc.get("title")
        content = doc.get("content", "")
        score = doc.get('_final_score', 0)
        context_texts.append(f"[{tag}] {content}")
        tag_lookup[tag] = filename

        url = resolve_reference_url(filename, doc.get("url", "N/A"), supplement_files)

        if filename not in doc_groups:
            doc_groups[filename] = {
                "document_name": filename,
                "title": title,   
                "url": url,
                "key_contact": doc.get("owner", "N/A"),
                "key_topics": doc.get("topics", ""),
                "key_terms": doc.get("terms", ""),
                "summary": doc.get("summary", ""),
                "scores": [score],
                "tags": [tag],
            }
        else:
            doc_groups[filename]["scores"].append(score)
            doc_groups[filename]["tags"].append(tag)

    context_str = "\n\n".join(context_texts)

    is_valid, explanation, is_completely_irrelevant = llm_context_guard_check(
                                                                                query, context_str, client, deployment=AZURE_OPENAI_DEPLOYMENT, strict_mode=strict_mode
                                                                            )
    explanation = ' '.join(explanation.strip().split()[1:])
    print('LLM Context Validity Check:', f'is_valid={is_valid};\n is_completely_irrelevant={is_completely_irrelevant};\n explanation={explanation};\n')
    
    # === CASE: Context not valid ===
    if not is_valid:
        # --------------------------------------------------------
        # PARTIALLY RELEVANT
        # --------------------------------------------------------
        if not is_completely_irrelevant and top_chunks:

            # --- Generate best-effort answer ---
            fallback_prompt = (
                "You are answering a question where the documents are related but do not fully match.\n\n"
                "Your task:\n"
                "- First, provide the most relevant information from the documents that could help the user.\n"
                "- This may include general policies, standard procedures, or broadly applicable rules.\n"
                "- Do NOT refuse to answer simply because an exact match is missing.\n"
                "- Do NOT speculate or invent details.\n"
                "- You must reference document sources only using the format (Doc1), (Doc2), etc. Always use parentheses.\n"
                "- You must place the reference immediately after the sentence or claim it supports. For example:\n"
                "    • The operator must press the emergency stop button. (Doc2)\n"
                "    • The hydraulic fluid should be replaced every 1000 hours. (Doc5)\n"
                "- Never use \"see DocX\", \"as shown in DocX\", or any other format.\n"
                "- Never group multiple document references together. If you are referencing more than one document, use separate tags like (Doc1), (Doc2), (Doc3) — not (Doc1, Doc2) or (Doc1 and Doc2).\n"
                "- Do NOT use any other format like 'See Doc1', 'as shown in Doc2', 'in Doc1' or simply reference without parentheses.\n"
                "After providing the relevant information, briefly note what specific detail is missing.\n\n"
                f"QUESTION: {query}\n\n"
                f"DOCUMENT CHUNKS:\n{context_str}"
            )

            completion = client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=[{"role": "user", "content": fallback_prompt}]
            )

            partial_answer = completion.choices[0].message.content.strip()

            # --- Extract referenced DocN tags ---
            referenced_tags = set(re.findall(r"Doc\d+", partial_answer))
            referenced_files = {tag_lookup[tag] for tag in referenced_tags if tag in tag_lookup}

            # Fallback if LLM did not cite
            if not referenced_files and top_chunks:
                referenced_files = {doc.get("filename", "N/A") for doc in top_chunks[:2]}

            # --- Clean references from answer ---
            partial_answer = strip_doc_references(partial_answer)
            clean_explanation = strip_doc_references(explanation)

            main_answer = (
                f"{partial_answer}\n\n"
                "Note: the provided documents may not clearly explain the requested information.\n"
                f"Reason: {clean_explanation}"
            )

            # --- Build references (same as normal path) ---
            reference_text = "\n\n**References:**\n"
            ref_count = 0

            for filename, doc in doc_groups.items():
                if filename not in referenced_files or ref_count >= 3:
                    continue

                ref_count += 1

                if show_title_in_ref and doc.get("title"):
                    document_display = doc["title"]
                else:
                    document_display = title_case_filename(
                        resolve_reference_name(doc["document_name"], supplement_files)
                    )

                url_value = resolve_reference_url(
                    doc["document_name"],
                    (doc.get("url") or "").split(";")[0].strip(),
                    supplement_files
                )

                reference_text += f"\n---\n**Resource**: [{document_display}]({url_value})\n\n"

                if not hide_ref_contact:
                    reference_text += (
                        f"**Key Contact**: {title_case_name(doc['key_contact'])}\n\n"
                    )
                    
            return f"**Answer:**\n\n{main_answer}{reference_text}"

        # --------------------------------------------------------
        # ORIGINAL — COMPLETELY IRRELEVANT
        # --------------------------------------------------------
        main_answer = (
            "Sorry, I cannot help with that. "
            "The provided documents do not clearly explain the requested information.\n"
            f"(Reason: {explanation})"
        )

        main_answer = strip_doc_references(main_answer)
        return f"**Answer:**\n\n{main_answer}"

    # === CASE: Valid context — build answer using tags ===
    instructions = [
        "You should never assume the user is referring to video games, pop culture, or unrelated general knowledge unless it's clearly stated. Focus strictly on mechanical, industrial, or Komatsu-related topics.",
        "- Be concise",
        "- You must use only the information explicitly stated in the document chunks.",
        "- Do not make assumptions, guesses, or inferences beyond the content.",
        "- If the answer is not clearly present in the provided documents, do not attempt to answer.",
    ]

    if strict_mode:
        instructions.append("- If insufficient information or not sure about the answer, respond with: 'Sorry, I cannot help with that.' Then briefly explain reasoning.")
    else:
        instructions.append("- If the answer is not fully supported by the context, ask the user to clarify their question instead of guessing.")

    instructions.extend([
        "- You must reference document sources only using the format (Doc1), (Doc2), etc. Always use parentheses.",
        "- You must place the reference immediately after the sentence or claim it supports. For example:",
        "    • The operator must press the emergency stop button. (Doc2)",
        "    • The hydraulic fluid should be replaced every 1000 hours. (Doc5)",
        "- Never use \"see DocX\", \"as shown in DocX\", or any other format.",
        "- Never group multiple document references together. If you are referencing more than one document, use separate tags like (Doc1), (Doc2), (Doc3) — not (Doc1, Doc2) or (Doc1 and Doc2).",
        "- Do NOT use any other format like 'See Doc1', 'as shown in Doc2', 'in Doc1' or simply reference without parentheses.",
        "- If your answer describes a process, include step-by-step instructions using bullet symbols.",
        "- When referring to any resource, document, or tool that includes URLs or links in the document chunks, format the urls or links as Markdown-style hyperlink. If fail to generate hyperlink, then just use the raw urls. Make sure to show all the links of the tools as reference",
        "- Use plain text with no HTML.",
        "- Separate sections and lists with line breaks."
    ])

    full_prompt = (
        f"Answer the following question using the context from the top relevant documents.\n"
        f"QUESTION: {query}\n\n"
        f"DOCUMENT CHUNKS:\n{context_str}\n\n"
        f"INSTRUCTIONS:\n" + "\n".join(instructions)
    )

    messages = [{"role": "user", "content": full_prompt}]

    completion = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=messages
    )

    main_answer = completion.choices[0].message.content.strip()

    if warning_msg:
        main_answer = f"Notice: {warning_msg.strip()}" + "\n\n" + main_answer

    print("* Generated raw Main Answer *:", main_answer)
    # Identify which tags were referenced
    referenced_tags = set(re.findall(r"Doc\d+", main_answer))
    referenced_files = {tag_lookup[tag] for tag in referenced_tags if tag in tag_lookup}

    # Fallback: if no (DocN) tags were referenced, fallback to top chunk
    if not referenced_files and top_chunks:     
        referenced_files = {doc.get("filename", "N/A") for doc in top_chunks[:2]}

    # Clean tag references from main_answer
    main_answer = strip_doc_references(main_answer)


    # === OPTIONAL IMAGE LINKS (NOW A SEPARATE FUNCTION) ===
    if show_image:
        main_answer = append_images_to_answer(main_answer, show_image)

    # Build reference section, max 4
    reference_text = "\n\n**References:**\n"
    ref_count = 0
    for filename, doc in doc_groups.items():
        if filename not in referenced_files:
            continue
        if ref_count >= 4:
            break
        ref_count += 1
        # Use title or filename
        if show_title_in_ref and doc.get("title"):
            document_display = doc["title"]
        else:
            document_display = title_case_filename(
                resolve_reference_name(doc["document_name"], supplement_files)
            )
        url_value = resolve_reference_url(doc['document_name'], (doc.get("url") or "").split(";")[0].strip(), supplement_files)
        reference_text += f"\n---\n**Resource**: [{document_display}]({url_value})\n\n"

        if not hide_ref_contact:
            key_contact = title_case_name(doc['key_contact'])
            reference_text += f"**Key Contact**: {key_contact}\n\n"

        if hide_ref_relevance:
            continue

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
                    f"Be concise. Summarize in less than 100 words why this document is relevant to the question below, "
                    f"based only on the document's summary, key topics, and key terms.\n"
                    f"Use bullet points for clarity. Do NOT use field labels or markdown.\n\n{relevance_context}"
                )
            }
        ]

        rel_response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=relevance_prompt
        )

        relevance_summary = rel_response.choices[0].message.content.strip()
        reference_text += f"**Relevance**: {relevance_summary}\n\n"

    return f"**Answer:**\n\n{main_answer}{reference_text}"



def combine_subquery_answers(subquery_answer_map: dict, original_query: str) -> str:
    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version="2024-12-01-preview"
    )

    if len(subquery_answer_map) == 1:
        return list(subquery_answer_map.values())[0]

    formatted_answers = "\n\n".join(
                                    [f"**Q:** {q}\n**A:** {a}" for q, a in subquery_answer_map.items()]
                                    )

    llm_prompt = (
    f"The user asked: \"{original_query}\"\n\n"
    f"The following are responses to sub-questions related to the query:\n\n"
    f"{formatted_answers}\n\n"
    "Please combine these answers into a single, coherent response for the user:"
    "Please ensure the final answer is concise, clear, and directly addresses the original question."
    "Use clear bullet points or sections. Each bullet point should start with a new line."
    "Do not remove or reformat any markdown hyperlinks (e.g., [text](url)).\n"
    "Preserve all links exactly as they appear.\n"
    )

    response = client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=[{"role": "user", "content": llm_prompt}],
                )
    return response.choices[0].message.content.strip()


def answer_general_question(query: str, index_keyterms_summary: dict):
    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version="2024-12-01-preview"
    )
    
    general_context = (
    f"You are a helpful assistant that answers general questions about the chatbot whose name is {chatbot_name}.\n"
    f"You can explain its capabilities, scope, features, supported indexes, and how it works. You answer questions like 'what can you help with?', 'what can you do?', 'what is your purpose?', 'What is {chatbot_name}','which document libraries do you cover?', etc.\n"
    f"Use the provided index keyterms and topics summary to inform your answers.\n"
    f"Purpose: This chatbot is designed to assist users by answering questions based on the provided source documents. It serves as a digital assistant for quick reference, clarification, and navigation across various document types."
    f"Capabilities: Answer questions based on indexed documents. Extract relevant information from documentation.Summarize content from documents. Locate document titles, and revision dates. Identify responsible groups or design owners. Guide users step-by-step through processes outlined in the documentation."
    )

    index_summary_context = "\n\nIndex Keyterms and topics Summary:\n" + json.dumps(index_keyterms_summary, indent=2)

    prompt = (
    f"Context:\n{general_context}\n\n"
    f"{index_summary_context}\n\n"
    f"Question: {query}\n\n"
    "Answer the question clearly and concisely, mentioning relevant topics or indexes if needed. if <index_summary_context> has nothing relevant to the question, just say 'Sorry, I don't have information about that. Could you please provide more details or clarify your question?'"
    )

    completion = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}]
    )
    summary = completion.choices[0].message.content.strip()
    return summary