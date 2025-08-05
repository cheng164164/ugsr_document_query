# Configuration for index creation
# Add more index configs as needed
import os

ENV_VARS = {
    "AZURE_BLOB_CONN_STRING": "<your_blob_connection_string>",
    "AZURE_DOC_INTELL_ENDPOINT": "<your_doc_intell_endpoint>",
    "AZURE_DOC_INTELL_KEY": "<your_doc_intell_key>",
    "AZURE_OPENAI_DEPLOYMENT_ENDPOINT": "<your_openai_deployment_endpoint>",
    "AZURE_OPENAI_KEY": "<your_openai_key>",
    "AZURE_OPENAI_DEPLOYMENT_NAME": "<your_openai_deployment_name>",
    "AZURE_OPENAI_API_VERSION": "<your_openai_api_version>",
    "AZURE_SEARCH_ENDPOINT": "<your_search_endpoint>",
    "AZURE_SEARCH_KEY": "<your_search_key>",
    "AZURE_OPENAI_ENDPOINT": "<your_openai_endpoint>",
    "AZURE_EMBEDDING_DEPLOYMENT_NAME": "<your_embedding_deployment_name>"
}


INDEX_CONFIGS = [
    {
        "index_name": "ugsr_index_v2",
        "metadata_container": "undergroound-engineering-document-metadata",
        "metadata_blob": "auto_extraction/ugsr_metadata_new.csv",
        "document_container": "underground-engineering-documents"
    },
    {
        "index_name": "business_index_v2",
        "metadata_container": "north-america-business-documents-metadata",
        "metadata_blob": "auto_extraction/business_metadata_new.csv",
        "document_container": "north-america-business-documents"
    }
]

SCHEMA_MAPPING_DICT = {
    # Mapping of standard schema -> metadata columns
    "ugsr_index_v2": {
        "Name": "Document Name",
        "Title": "Document Title",
        "Doc Type": "Type",
        "Document Owner(s)": "Key Contact",
        "Doc Category": "Document Category",
        "Function": None, # No direct mapping
        "url": "URL"
    },
    "business_index_v2": {
        "Name": "Name",
        "Title": "Title",
        "Doc Type": "Doc Type",
        "Document Owner(s)": "Document Owner(s)",
        "Doc Category": "Doc Category",
        "Function": "Function",
        "url": "url"
    }
}