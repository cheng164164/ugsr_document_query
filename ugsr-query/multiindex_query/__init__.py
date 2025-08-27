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
from .config import ENV_VARS, index_names, metadata_files, share_point_urls, feature_flags
from .search_query import *


debug_mode = feature_flags["debug_mode"]
keywords_matching = feature_flags["keywords_matching"]
custom_ranking = feature_flags["custom_ranking"]
dynamic_filtering = feature_flags["dynamic_filtering"]
metadata_search = feature_flags["metadata_search"]
use_prev_context = feature_flags["use_prev_context"]
hide_ref_relevance = feature_flags["hide_ref_relevance"]


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    try:
        req_body = req.get_json()
        query = req_body.get("query")
        query_history = req_body.get("queryhistory", "")
        answer_history = req_body.get("answerhistory", "")
        if not query:
            return func.HttpResponse("Error: Missing 'query' parameter", status_code=400)

        cleaned_query = clean_query_for_llm(query)  # Clean query by removing routing keywords if there are any
        cleaned_query_history = clean_query_for_llm(query_history)

        history_context = filter_relevant_history(cleaned_query, cleaned_query_history, answer_history)
        if use_prev_context:
            rewrited_query = rewrite_query_with_history(cleaned_query, history_context)
        else:
            rewrited_query = cleaned_query

        if metadata_search:
            use_metadata_search_flag = should_use_metadata_search(query)    # use raw query for routing decision
            if use_metadata_search_flag:
                metadata_by_index = metadata_table_by_index(index_names)
                llm_summary = summarize_full_metadata(rewrited_query, history_context, metadata_by_index)        
                return func.HttpResponse(json.dumps({"answer": llm_summary}, ensure_ascii=False, indent=2), mimetype="application/json", status_code=200)

        # Step 1: Search all indexes
        docs = multi_index_search_documents(cleaned_query, rewrited_query, index_names, vector_weight=0.6, top_k=16, 
                                                            dynamic_filtering=dynamic_filtering, 
                                                            keywords_matching = keywords_matching,
                                                            use_previous_context = use_prev_context,    
                                                            custom_ranking=custom_ranking, 
                                                            debug=debug_mode)
        if not docs:
            return func.HttpResponse("No relevant documents found.", status_code=404)
        
        # Step 2: Generate response from AI with retrieved context
        ai_response = multi_index_generate_response(rewrited_query, docs, hide_ref_relevance=hide_ref_relevance)

        return func.HttpResponse(json.dumps({"answer": ai_response}, ensure_ascii=False, indent=2), mimetype="application/json", status_code=200)

    except Exception as e:
        return func.HttpResponse(f"Error: {str(e)}", status_code=500)