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
from .util import detect_specific_index

debug_mode = feature_flags["debug_mode"]
parallel_queries = feature_flags["parallel_queries"]
keywords_matching = feature_flags["keywords_matching"]
custom_ranking = feature_flags["custom_ranking"]
dynamic_filtering = feature_flags["dynamic_filtering"]
metadata_search = feature_flags["metadata_search"]
use_prev_context = feature_flags["use_prev_context"]
hide_ref_relevance = feature_flags["hide_ref_relevance"]
mock_db = feature_flags["mock_db"]


if mock_db:
    from .db_utils import save_chat, fetch_recent_history
else:
    from .util import save_chat, fetch_recent_history


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    try:
        req_body = req.get_json()
        query = req_body.get("query")
        user_id = req_body.get("user_id", "unknown")
        user_name = req_body.get("user_name", "anonymous")
        metadata = req_body.get("metadata", {})        
        # query_history = req_body.get("queryhistory", "")
        # answer_history = req_body.get("answerhistory", "")

        if not query or query.strip() == "":
            return func.HttpResponse(json.dumps({
                        "answer": "Hi! It looks like you've returned after a break. Please re-enter your question so I can assist you."
                        }), mimetype="application/json", status_code=200)

        cleaned_query = clean_query_for_llm(query)  # Clean query by removing routing keywords if there are any

        # saving user query to DB
        try:
            save_chat(user_id, user_name, "user", cleaned_query, metadata)
        except Exception as e:
            logging.warning(f"⚠️ Failed to save user query to DB: {e}")

        # fetching last 10 turns
        try:
            query_history, answer_history = fetch_recent_history(user_id, 5)
        except Exception as e:
            logging.warning(f"⚠️ Failed to fetch chat history: {e}")
            query_history, answer_history = "", ""
        
        if use_prev_context and (query_history or answer_history):
            history_context = filter_relevant_history(cleaned_query, query_history, answer_history)
            rewrited_query = rewrite_query_with_history(cleaned_query, history_context)
        else:
            history_context = ""
            rewrited_query = cleaned_query

        ## Step 1 : Check if query mentions a specific index
        target_index = detect_specific_index(query, index_aliases)

        ## Step 2: Check if metadata search is needed
        if metadata_search:
            use_metadata_search_flag = should_use_metadata_search(query)    # use raw query for routing decision
            if use_metadata_search_flag:
                if target_index:
                    logging.info(f"🎯 Metadata search restricted to index: {target_index}")
                    metadata_by_index = metadata_table_by_index([target_index])
                else:
                    metadata_by_index = metadata_table_by_index(index_names)

                llm_summary = summarize_full_metadata(rewrited_query, history_context, metadata_by_index)  

                try:
                    save_chat(user_id, user_name, "bot", llm_summary, metadata)
                except Exception as e:
                    logging.warning(f"⚠️ Failed to save bot response to DB: {e}")        
        
                return func.HttpResponse(json.dumps({"answer": llm_summary}, ensure_ascii=False, indent=2), mimetype="application/json", status_code=200)


        ## Step 3: Search all indexes
        if target_index:
            logging.info(f"🎯 Content search restricted to index: {target_index}")
            search_scope = [target_index]
            parallel_flag = False
        else:
            search_scope = index_names
            parallel_flag = parallel_queries
        docs = multi_index_search_documents(cleaned_query, rewrited_query, search_scope, vector_weight=0.6, top_k=8, 
                                                            dynamic_filtering=dynamic_filtering, 
                                                            keywords_matching = keywords_matching,
                                                            custom_ranking=custom_ranking,
                                                            use_previous_context = use_prev_context,    
                                                            parallel=parallel_flag, 
                                                            debug=debug_mode)
        if not docs:
            return func.HttpResponse("No relevant documents found.", status_code=404)
        
        ## Step 4: Generate response from AI with retrieved context
        ai_response = multi_index_generate_response(rewrited_query, docs, hide_ref_relevance=hide_ref_relevance)

        ## Step 5: Save bot response with metadata
        try:
            save_chat(user_id, user_name, "bot", ai_response, metadata)
        except Exception as e:
            logging.warning(f"⚠️ Failed to save bot response to DB: {e}")        
        
        return func.HttpResponse(json.dumps({"answer": ai_response}, ensure_ascii=False, indent=2), mimetype="application/json", status_code=200)

    except Exception as e:
        return func.HttpResponse(f"Error: {str(e)}", status_code=500)