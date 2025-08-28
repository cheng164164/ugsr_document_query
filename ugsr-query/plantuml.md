
@startuml
title Query Search and Response Flow

start

:Request Received;
:Azure Function triggered by HTTP request (query + history);

partition "Query Pre-Processing" {
    :Clean/normalize query and history;
    :Filter relevant history (LLM);
    :Rewrite query with history (LLM);
}

:LLM decides routing (metadata or contents search);

if (Route: Metadata Search?) then (Yes)
    partition "Metadata Search" {
        :Read metadata table as dictionary per index;
        :LLM generates response/summary from metadata;
        :Return metadata response;
        stop
    }
else (No)
    partition "Multi-Indexes Search" {
        :Retrieve top K chunks from each index;
        :Rerank chunks (hybrid semantic + vector scoring);
        :Select best matching index (average top chunk score);
        :Return top chunks from best index;
    }
    partition "Multi-Indexes Response" {
        if (Query and context fully match?) then (Yes)
            :Return answer;
            stop
        else (No)
            if (Query and context partially match?) then (Yes)
                :Ask for clarification and provide assistance;
                stop
            else (No)
                :Ask for clarification only;
                stop
            endif
        endif
    }
endif

@enduml




########################################################################

@startuml
title multi_index_search_documents Flowchart

start

:Receive query, rewrited_query, index_names, and parameters;
:Prepare headers and field lists;
:Call llm_search_query_optimizer;
:Call get_query_embedding;
:Initialize all_results, all_content, index_debug_data;

:For each index_name in index_names;
repeat
    :Prepare select_fields and payload;
    :POST request to Azure Search;
    if (response.status_code == 200) then (yes)
        :For each doc in hits;
        :Calculate scores (cosine_similarity, keyword_score, final_score);
        :Add doc to all_results and all_content;
        :Sort hits by _final_score;
    else (no)
        :Log warning;
    endif
repeat while (more index_names?)

if (all_results is empty?) then (yes)
    :Return [];
    stop
endif

:Compute average top-3 final scores per index;
:Sort indexes by average score;
:Select best_index;
:Select top_k results from best_index;
:Sort by _final_score;
:Check for missing keywords in all_content (if enabled);
:Return warning_msg and final_results;

stop
@enduml



######################################################################


@startuml
title multi_index_generate_response Flowchart

start

:Receive query, context;
:Initialize AzureOpenAI client;
:Unpack warning_msg, top_chunks from context;
:Build context_texts and doc_groups from top_chunks;
:Build context_str for LLM;

:Call llm_context_guard_check to validate context;
if (Is context valid?) then (No)
    :Build main_answer with explanation;
    if (Not completely irrelevant and top_chunks exist) then (Yes)
        :Suggest most common doc and contact;
    endif
    :Return answer;
    stop
else (Yes)
    :Build messages for LLM answer;
    :Call AzureOpenAI to generate main_answer;
    :Build reference section;
    :For each doc in top 3 doc_groups;
        :Add doc and contact to references;
    :Return answer with references;
    stop
endif
@enduml





