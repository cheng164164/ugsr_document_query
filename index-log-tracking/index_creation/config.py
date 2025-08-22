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
    "AZURE_EMBEDDING_DEPLOYMENT_NAME": "<your_embedding_deployment_name>",
    "QUEUE_NAME": "<your_queue_name>"
}



#### Chatbot "Kimmi"
INDEX_CONFIGS = [
    {
        "index_name": "ugsr_index",
        "metadata_container": "undergroound-engineering-document-metadata",
        "metadata_blob": "auto_extraction/ugsr_metadata_new.csv",
        "document_container": "underground-engineering-documents",
        "group": 1
    },
    {
        "index_name": "business_index",
        "metadata_container": "north-america-business-documents-metadata",
        "metadata_blob": "auto_extraction/business_metadata_new.csv",
        "document_container": "north-america-business-documents",
        "group": 1
    },
    {
        "index_name": "all_regions_index",
        "metadata_container": "all-regions-documents-metadata",
        "metadata_blob": "auto_extraction/all_regions_metadata_new.csv",
        "document_container": "all-regions-documents",
        "group": 1
    },
    {
        "index_name": "ehs_index",
        "metadata_container": "global-ehs-documents-metadata",
        "metadata_blob": "auto_extraction/ehs_metadata_new.csv",
        "document_container": "global-ehs-documents",
        "group": 2
    },
    {
        "index_name": "policy_center_index",
        "metadata_container": "policy-center-documents-metadata",
        "metadata_blob": "auto_extraction/policy_center_metadata_new.csv",
        "document_container": "policy-center-documents",
        "group": 2
    },
]

SCHEMA_MAPPING_DICT = {
    ### Mapping of standard schema -> metadata columns
    "ugsr_index": {
        "Name": "Name",
        "Title": "Title",
        "Doc Type": "Doc Type",
        "Document Owner(s)": "Document Owner(s)",
        "Doc Category": "Doc Category",
        "Function": None, # No direct mapping
        "url": "url"
    },
    "business_index": {
        "Name": "Name",
        "Title": "Title",
        "Doc Type": "Doc Type",
        "Document Owner(s)": "Document Owner(s)",
        "Doc Category": "Doc Category",
        "Function": "Function",
        "url": "url"
    },
    "all_regions_index": {
        "Name": "Name",
        "Title": "Title",
        "Doc Type": "Doc Type",
        "Document Owner(s)": "Document Owner(s)",
        "Doc Category": "Doc Category",
        "Function": "Function",
        "url": "url"
    },
    "ehs_index": {
        "Name": "Name",
        "Title": None,
        "Doc Type": None,
        "Document Owner(s)": "Document Owner(s)",
        "Doc Category": None,
        "Function": None,
        "url": "url"
    }, 
    "policy_center_index": {
        "Name": "Name",
        "Title": None,
        "Doc Type": None,
        "Document Owner(s)": "Document Owner(s)",
        "Doc Category": None,
        "Function": "Function",
        "url": "url"
    },    
}


'''
#### Chatbot "Andy"
INDEX_CONFIGS = [
    {
        "index_name": "dev_hub_index",
        "metadata_container": "development-hub-document-metadata",
        "metadata_blob": "auto_extraction/development_hub_metadata_new.csv",
        "document_container": "development-hub-documents",
        "group": 1
    },
]

SCHEMA_MAPPING_DICT = {
    ### Mapping of standard schema -> metadata columns
    "dev_hub_index": {
        "Name": "Name",
        "Title": None,
        "Doc Type": None,
        "Document Owner(s)": "Document Owner(s)",
        "Doc Category": "Doc Category",
        "Function": None, # No direct mapping
        "url": "url"
    } 
}
'''