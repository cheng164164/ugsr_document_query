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
from .config import ENV_VARS, index_names, metadata_files, share_point_urls, index_aliases, feature_flags
from .search_query import *
from .util import detect_specific_index, get_or_build_metadata_summaries


debug_mode = feature_flags["debug_mode"]
index_suggestion = feature_flags.get("index_suggestion", True)
parallel_queries = feature_flags["parallel_queries"]
keywords_matching = feature_flags["keywords_matching"]
custom_ranking = feature_flags["custom_ranking"]
dynamic_filtering = feature_flags["dynamic_filtering"]
metadata_search = feature_flags["metadata_search"]
use_prev_context = feature_flags["use_prev_context"]
hide_ref_relevance = feature_flags["hide_ref_relevance"]
strict_mode = feature_flags.get("strict_mode", False)
mock_db = feature_flags["mock_db"]

BLOB_CONN_STR = os.getenv("AZURE_BLOB_CONN_STRING")
INDEX_METADATA_SUMMARIES = None  # global cache

if mock_db:
    from .db_utils import save_chat, fetch_recent_history
else:
    from .util import save_chat, fetch_recent_history


def main(req: func.HttpRequest) -> func.HttpResponse:
    global INDEX_METADATA_SUMMARIES
    if index_suggestion:
        if INDEX_METADATA_SUMMARIES is None:
            INDEX_METADATA_SUMMARIES = get_or_build_metadata_summaries(index_names, 
                                                                   BLOB_CONN_STR, 
                                                                   container_name='index-metadata-summary', 
                                                                   blob_name='metadata_summaries.json')

    logging.info('Python HTTP trigger function processed a request.')
    try:
        req_body = req.get_json()
        query = req_body.get("query")
        user_id = req_body.get("user_id", "unknown")
        user_name = req_body.get("user_name", "anonymous")
        metadata = req_body.get("metadata", {})        
        query_history = req_body.get("queryhistory", "")
        answer_history = req_body.get("answerhistory", "")

        if not query or query.strip() == "":
            return func.HttpResponse(json.dumps({
                        "answer": "Hi! It looks like you've returned after a break. Please re-enter your question so I can assist you."
                        }), mimetype="application/json", status_code=200)

        cleaned_query = clean_query_for_llm(query)  # Clean query by removing routing keywords if there are any

        ## saving user query to DB
        save_chat(user_id, user_name, "user", cleaned_query, metadata)

        ## fetching last N turns
        query_history, answer_history = fetch_recent_history(query_history, answer_history, user_id, 5)
        
        if use_prev_context and (query_history or answer_history):
            history_context = filter_relevant_history(cleaned_query, query_history, answer_history)
            rewrited_query = rewrite_query_with_history(cleaned_query, history_context)
        else:
            history_context = ""
            rewrited_query = cleaned_query

        sub_queries = decompose_query(rewrited_query)[:4]  # Limit to top 4 sub-queries
        logging.info(f"🔍 Decomposed into {len(sub_queries)} sub-queries.")
        
        ## Step 1: Index filtering
        if len(index_names) == 1:
            target_indexes = index_names
        else:
            target_index_by_keyword = detect_specific_index(query, index_aliases)
            if target_index_by_keyword:
                target_indexes = [target_index_by_keyword]
                logging.info(f"🔍 LLM-suggested indexes by keyword matching: {target_indexes}")
            elif index_suggestion:
                target_indexes = select_relevant_indexes_via_llm(query, INDEX_METADATA_SUMMARIES, top_n=2)
                logging.info(f"📚 LLM-suggested indexes by metadata summaries: {target_indexes}")
            else:
                target_indexes = index_names
                logging.info(f"📚 No index suggestion, searching all indexes: {target_indexes}")
                
        ## Step 2: Check if metadata search is needed to gnenerate answer
        use_metadata_search_flag = should_use_metadata_search(query)    # use raw query for routing decision
        if metadata_search and use_metadata_search_flag == "metadata":
            if target_indexes:
                doc_metadata_by_index = metadata_table_by_index(target_indexes)
                logging.info(f"🗂️ Using metadata search for indexes: {target_indexes}")
            else:
                doc_metadata_by_index = metadata_table_by_index(index_names)

            def handle_subquery_metadata(subq):
                sub_ans = summarize_full_metadata(subq, history_context, doc_metadata_by_index, parallel=parallel_queries)
                return sub_ans
                
            if len(sub_queries) == 1:
                llm_summary = handle_subquery_metadata(sub_queries[0])
            else:
                if parallel_queries:
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        subquery_results = list(executor.map(handle_subquery_metadata, sub_queries))
                else:
                    subquery_results = [handle_subquery_metadata(sq) for sq in sub_queries]

                subquery_answer_map = dict(zip(sub_queries, subquery_results))
                llm_summary = combine_subquery_answers(subquery_answer_map, cleaned_query)

            try:
                save_chat(user_id, user_name, "bot", llm_summary, metadata)
            except Exception as e:
                logging.warning(f"⚠️ Failed to save bot response to DB: {e}")        
        
            return func.HttpResponse(json.dumps({"answer": llm_summary}, ensure_ascii=False, indent=2), mimetype="application/json", status_code=200)

        if metadata_search and use_metadata_search_flag == "general":
            llm_summary = answer_general_question(rewrited_query, index_keyterms_summary=INDEX_METADATA_SUMMARIES)
            save_chat(user_id, user_name, "bot", llm_summary, metadata)       
            return func.HttpResponse(json.dumps({"answer": llm_summary}, ensure_ascii=False, indent=2), mimetype="application/json", status_code=200)

        ## Step 3: Otherwise, Search all indexes and generate answer
        search_scope = target_indexes if target_indexes else index_names
        parallel_flag = parallel_queries if len(target_indexes)>1 else False
        logging.info(f"🔍 Searching indexes: {search_scope} | Parallel: {parallel_flag}")

        def process_content_subquery(subq):
            docs = multi_index_search_documents(cleaned_query, rewrited_query, search_scope, vector_weight=0.6, top_k=8, 
                                                                dynamic_filtering=dynamic_filtering, 
                                                                keywords_matching = keywords_matching,
                                                                custom_ranking=custom_ranking,
                                                                use_previous_context = use_prev_context,    
                                                                parallel=parallel_flag, 
                                                                debug=debug_mode)
            
            if not docs:
                return func.HttpResponse("No relevant documents found.", status_code=404)
            return multi_index_generate_response(subq, docs, hide_ref_relevance=hide_ref_relevance, strict_mode=strict_mode)
        
        if len(sub_queries) == 1:
            ai_response = process_content_subquery(sub_queries[0])
        else:
            if parallel_queries:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    subquery_results = list(executor.map(process_content_subquery, sub_queries))
            else:
                subquery_results = [process_content_subquery(sq) for sq in sub_queries]

            subquery_answer_map = dict(zip(sub_queries, subquery_results))
            ai_response = combine_subquery_answers(subquery_answer_map, cleaned_query)
            
        ## Step 4: Save bot response with metadata
        save_chat(user_id, user_name, "bot", ai_response, metadata)       
        
        return func.HttpResponse(json.dumps({"answer": ai_response}, ensure_ascii=False, indent=2), mimetype="application/json", status_code=200)

    except Exception as e:
        return func.HttpResponse(f"Error: {str(e)}", status_code=500)