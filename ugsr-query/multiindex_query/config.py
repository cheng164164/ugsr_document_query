##################### General Settings ######################################
ENV_VARS = {
    "AZURE_BLOB_CONN_STRING": "<your_blob_connection_string>",
    "AZURE_OPENAI_API_KEY": "<your_openai_api_key>",
    "AZURE_OPENAI_DEPLOYMENT": "<your_openai_deployment_name>",
    "AZURE_OPENAI_ENDPOINT": "<your_openai_endpoint>",
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "<your_openai_embedding_deployment_name>",
    "AZURE_SEARCH_ENDPOINT": "<your_search_endpoint>",
    "AZURE_SEARCH_KEY": "<your_search_key>",
    "AZURE_SEARCH_INDEX": "<your_search_index_name>",
    "AZURE_SQL_CONN_STR": "<your_sql_connection_string>"
}


# Global Feature On/Off flags
feature_flags = {
    "debug_mode": False,   # Set to True to enable debug prints
    "index_suggestion": False,  # Set to True to enable index suggestion based on query
    "metadata_search": False,     # Set to True to enable metadata-only search for relevant queries
    "parallel_queries": True,  # Set to True to enable parallel queries to multiple indexes
    "custom_ranking": True,   # Set to True to enable custom ranking (vector + keyword); False to use Azure default ranking
    "use_prev_context": True,    # Set to True to enable the feature that uses previous queries as context
    "hide_ref_relevance": True,    # Set to True to hide relevance explanation in the reference section
    "dynamic_filtering": False,   # Set to True to enable dynamic metadata filtering based on query keywords
    "keywords_matching": False,   # Set to True to enable keyword matching check and warning 
    "strict_mode": False,  # Set to True to enable strict model for ensuring query is explicitly answered by the retrieved context
    "show_image": False,  # Set to True to enable images in the reference section
    "show_title_in_ref": True,    # Set to True to show title in reference section
    "hide_ref_contact": True,    # Set to True to hide reference contact
    "mock_db": True    # Set to True to use mock DB functions for testing without actual DB connection
}


# Chatbot Feature Override Configurations
chatbot_feature_overrides = {
    "Kimmi": {
        "index_suggestion": True,
        "metadata_search": True,
        "hide_ref_contact": False,
        "mock_db": False
    },

    "Andi": {
        "index_suggestion": False,
        "metadata_search": False,
        "hide_ref_contact": True,
        "mock_db": True
    },
    
    "LDGN": {
        "index_suggestion": False,
        "metadata_search": False,
        "hide_ref_contact": True,
        "show_image": True,
        "show_title_in_ref": True,
        "mock_db": True
    }
}
######################### Chatbot Config ######################################
### Chatbot Kimmi
'''
# Define mutli indexes names to search
chatbot_name = "Kimmi"

index_names = ["ugsr_index",
               "business_index",
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
    "all_regions_index": {'name': 'Business Docuemnts All Regions', "url": "https://globalkomatsu.sharepoint.com/sites/NAGMUSGR00243/SitePages/PublishedDocuments.aspx"},
    "policy_center_index": {'name': 'Policy Center Documents', "url": "https://globalkomatsu.sharepoint.com/sites/komunity/policycenter/SitePages/Policy-Center-Landing-Page.aspx"}
}

# Supplementary files in blob storage for each index
supplement_files = {}

# Aliases for each index to catch user queries that mention library/index by name
index_aliases = {
    "business_index": ["north america", "north americas", "ameirca north", "americas north"],
    "ugsr_index": ["ugsr", "underground", "engineering"],
    "ehs_index": ["ehs", "global ehs"],
    "all_regions_index": ["all regions", "all region", "regions all", "region all"],
    "policy_center_index": ["policy center", "policy-center"]
}
'''
######################### Chatbot Config ######################################

'''
### Chatbot Andi
chatbot_name = "Andi"

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

# Supplementary files in blob storage for each index
supplement_files = {"dev_hub_index": [{'file_name': "Change_Management_HomePage.pdf",'reference_link': "https://globalkomatsu.sharepoint.com/sites/NAGMUSGR00270/SitePages/Change-Management.aspx"},
                                    {'file_name': "Career Planning Homepage.pdf",'reference_link': "https://globalkomatsu.sharepoint.com/sites/NAGMUSGR00270/SitePages/Your-Career.aspx"},
                                    {'file_name': "Development Hub Homepage.pdf",'reference_link': "https://globalkomatsu.sharepoint.com/sites/NAGMUSGR00270"},
                                    {'file_name': "Komatsu internal job candidate collection.pdf",'reference_link': "https://www.linkedin.com/learning/collections/enterprise/1~AAAAAAAjyzo=1979017?trk=share_ent_collection_url&shareId=qhfLNaPLQT2v0i0piOxY9A%3D%3D&accountId=2345786&u=2345786&success=true&authUUID=n6sdtoCXREC9D7rWZ3z9yA%3D%3D"},
                                    {'file_name': "Leadership Development Homepage.pdf",'reference_link': "https://globalkomatsu.sharepoint.com/sites/NAGMUSGR00270/SitePages/Leadership-Development.aspx"},
                                    {'file_name': "Performance and Goals Homepage.pdf",'reference_link': "https://globalkomatsu.sharepoint.com/sites/NAGMUSGR00270/SitePages/Performance-and-Goals.aspx"},
                                    {'file_name': "Welcome to the Development Hub_video.pdf",'reference_link': "https://globalkomatsu.sharepoint.com/sites/NAGMUSGR00270/_layouts/15/stream.aspx?sw=bypass&bypassReason=abandoned&id=%2Fsites%2FNAGMUSGR00270%2FAll+Documents%2FWelcome+to+the+Development+Hub.mp4&startedResponseCatch=true"}
                                    ]
}

# Aliases for each index to catch user queries that mention library/index by name
index_aliases = {}
'''


######################### Chatbot Config ######################################

### Chatbot LDGN
chatbot_name = "LDGN"

index_names = ["ldgn_index"]

# Metadata files in blob storage for each index
metadata_files = {}

# SharePoint URLs for each index
share_point_urls = {}

# Supplementary files in blob storage for each index
supplement_files = {}

# Aliases for each index to catch user queries that mention library/index by name
index_aliases = {}
