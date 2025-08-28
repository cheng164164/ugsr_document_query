##################### General Settings ######################################
ENV_VARS = {
    "AZURE_BLOB_CONN_STRING": "<your_blob_connection_string>",
    "AZURE_OPENAI_KEY": "<your_openai_key>",
    "AZURE_OPENAI_DEPLOYMENT_NAME": "<your_openai_deployment_name>",
    "AZURE_OPENAI_ENDPOINT": "<your_openai_endpoint>",
    "AZURE_SEARCH_ENDPOINT": "<your_search_endpoint>",
    "AZURE_SEARCH_KEY": "<your_search_key>",
    "AZURE_SEARCH_INDEX": "<your_search_index_name>",
}


# Feature On/Off flags
feature_flags = {
    "debug_mode": False,   # Set to True to enable debug prints
    "custom_ranking": True,   # Set to True to enable custom ranking (vector + keyword); False to use Azure default ranking
    "dynamic_filtering": False,   # Set to True to enable dynamic metadata filtering based on query keywords
    "keywords_matching": False,   # Set to True to enable keyword matching check and warning 
    "metadata_search": True,     # Set to True to enable metadata-only search for relevant queries
    "use_prev_context": True,    # Set to True to enable the feature that uses previous queries as context
    "hide_ref_relevance": True    # Set to True to hide relevance explanation in the reference section
}



######################### Chatbot Config ######################################
### Chatbot Kimmi
'''
# Define mutli indexes names to search
index_names = ["business_index",
               "ugsr_index",
               "ehs_index",
               "all_regions_index",
               "policy_center_index"
              ]

# Metadata files in blob storage for each index
metadata_files = {
    "business_index": {'container_name': "north-america-business-documents-metadata", 'file_name': "auto_extraction/business_metadata_new.csv"},
    "ugsr_index": {'container_name': "undergroound-engineering-document-metadata", 'file_name': "auto_extraction/ugsr_metadata_new.csv"},
    "ehs_index": {'container_name': "global-ehs-documents-metadata", 'file_name': "auto_extraction/ehs_metadata_new.csv"},
    "all_regions_index": {'container_name': "all-regions-documents-metadata", 'file_name': "auto_extraction/all_regions_metadata_new.csv"},
    "policy_center_index": {'container_name': "policy-center-documents-metadata", 'file_name': "auto_extraction/policy_center_metadata_new.csv"}
}

# SharePoint URLs for each index
share_point_urls = {
    "business_index": {'name': 'Business Documents North America', "url": "https://globalkomatsu.sharepoint.com/sites/NAGMUSGR00243/SitePages/PublishedDocuments.aspx"},
    "ugsr_index": {'name': 'UGSR Engineering Documents', "url": "https://globalkomatsu.sharepoint.com/sites/NAGMUSGR00221/engres_joy/PPED/JGUEngDocs?viewpath=%2Fsites%2FNAGMUSGR00221%2Fengres%5Fjoy%2FPPED%2FJGUEngDocs"},
    "ehs_index": {'name': 'Global EHS Documents', "url": "https://globalkomatsu.sharepoint.com/sites/Velocity-GlobalPoliciesandProcedures/Shared%20Documents/Forms/AllItems.aspx?id=%2Fsites%2FVelocity%2DGlobalPoliciesandProcedures%2FShared%20Documents%2FGeneral%2FGlobal%20EHS%20Policies&viewid=3d027989%2Ddf2e%2D434d%2Da643%2D3e28353d8fbb&csf=1&web=1&e=pcFPZF&CID=940883b9%2D627f%2D4a42%2D8bad%2D5120ca6b6223&FolderCTID=0x0120003F332C7233C5DB4A94D41DD5FBC21C23"},
    "all_regions_index": {'name': 'Business Docuemnts All Regions', "url": "https://globalkomatsu.sharepoint.com/sites/komunity/policycenter/SitePages/Policy-Center-Landing-Page.aspx"},
    "policy_center_index": {'name': 'Policy Center Documents', "url": "https://globalkomatsu.sharepoint.com/sites/komunity/policycenter/SitePages/Policy-Center-Landing-Page.aspx"}
}
'''



### Chatbot Andy
index_names = ["dev_hub_index"
              ]

# Metadata files in blob storage for each index
metadata_files = {
    "dev_hub_index": {'container_name': "development-hub-documents-metadata", 'file_name': "auto_extraction/development_hub_metadata_new.csv"}
}

# SharePoint URLs for each index
share_point_urls = {
    "dev_hub_index": {'name': 'Development Hub Documents', "url": "https://globalkomatsu.sharepoint.com/sites/NAGMUSGR00270/All%20Documents/Forms/AllItems.aspx"}
}
