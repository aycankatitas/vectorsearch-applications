from tiktoken import get_encoding, encoding_for_model
from weaviate_interface import WeaviateClient
from weaviate_interface import WhereFilter
from prompt_templates import question_answering_prompt_series, question_answering_system
from openai_interface import GPT_Turbo
from app_features import (convert_seconds, generate_prompt_series, search_result,
                          validate_token_threshold, load_content_cache, load_data)
from reranker import ReRanker
from loguru import logger 
import streamlit as st
import sys
import json
import os
from classmappings import class_model_mapping
import csv

from sentence_transformers import SentenceTransformer
from callbacks import *

import wikipedia

# load environment variables
from dotenv import load_dotenv
load_dotenv('.env', override=True)
 
## PAGE CONFIGURATION
st.set_page_config(page_title="Impact Theory", 
                   page_icon=None, 
                   layout="wide", 
                   initial_sidebar_state="auto", 
                   menu_items=None)

## KEYS 
api_key = os.environ["WEAVIATE_API_KEY"]
url = os.environ["WEAVIATE_ENDPOINT"]
openai_key = os.environ["OPENAI_API_KEY"]

## DATA
data_path = './data/impact_theory_data.json'
data = load_data(data_path)

## RETRIEVER

client = WeaviateClient(api_key,url)
client.display_properties.append('summary')
#available_classes=client.show_classes()

## RERANKER
reranker = ReRanker(model_name="cross-encoder/ms-marco-MiniLM-L-12-v2")

## LLM 
llm = GPT_Turbo(model="gpt-3.5-turbo-0613", api_key=openai_key)

## ENCODING
encoding = encoding_for_model("gpt-3.5-turbo-0613")

## INDEX NAME
#class_name = "Impact_theory_minilm_256"

#creates list of guests for sidebar
guest_list = sorted(list(set([d['guest'] for d in data])))

def main():
        
    with st.sidebar:
        guest = st.selectbox('Select Guest', 
                             options=guest_list, 
                             index=None, 
                             placeholder='Select Guest')
        guest_filter = None
        if guest:
            guest_filter = WhereFilter(['guest'], operator="Equal",valueText=guest).todict()

        alpha_input = st.slider('Alpha for Hybrid Search',0.00,1.00,0.30)
        retrieval_limit = st.slider('Hybrid Search Retrieval Results',1,100,10)
        reranker_topk = st.slider('Reranker Top K',1,50,3)
        temperature_input = st.slider('Temperature of LLM',0.0,2.0,1.0)

        model_name=st.selectbox('Choose a model:',
                                list(class_model_mapping.keys()),
                                placeholder='Select a Model to Run')
        
        class_name = class_model_mapping[model_name]["class_name"]
        model_name_or_path = class_model_mapping[model_name]["model_name"]
        
        client = WeaviateClient(api_key,
                                    url,
                                    model_name_or_path=model_name_or_path)
        
    st.image('./assets/impact-theory-logo.png', width=400)

    # Podcast Summary 

    show_summary = """
    If you’re looking to thrive in uncertain times, achieve unprecedented goals, and 
        improve the most meaningful aspects of your life, then Impact Theory is the show for you. 
             Hosted by Tom Bilyeu, a voracious learner and hyper-successful entrepreneur, the show 
             investigates and analyzes the most useful topics with the world’s most sought-after guests. 
             Bilyeu attacks each episode with a clear desire to further evolve the holistic skillset 
             that allowed him to co-found the billion dollar company Quest Nutrition, generate over 
             half a billion organic views on his content, build a thriving marriage of over 20 years, 
             and quantifiably improve the lives of over 10,000 people through his school, Impact Theory 
             University. Bilyeu’s insatiable hunger for knowledge gives the show urgency, relevance, 
             and depth while leaving listeners with the knowledge, tools, and empowerment to take 
             control of their lives and develop true personal power.
             """
    col1, _ = st.columns([3,3])
    col1.write(show_summary)
    
    st.subheader(f"Chat with the Impact Theory podcast: ")
    st.write('\n')
    st.write("Select one of the questions below or enter your own question to start searching.")
    col1, _ = st.columns([7,3])
    with col1:

        # Obtain list of suggested questions from CSV
        csv_path = "cuedquestions.csv"
        file_obj = open(csv_path, 'r')
        reader = csv.reader(file_obj)
        suggested_questions = []
        for item in reader:
            suggested_questions.append(item[0])

        if 'selected_question' not in st.session_state:
            st.session_state.selected_question = False

        st.radio(label="Suggested Questions", options=suggested_questions, key="selected_question")
        st.button(label="Ask Impact Theory Podcast", on_click=fill_suggested_question)

        st.write('\n\n\n\n\n')
                    
        st.text_input('Enter your question: ',key="q")
        st.write('\n\n\n\n\n')

        if st.session_state.q:
            query = st.session_state.q
        
            #st.write('Hmmm...this app does not seem to be working yet.  Please check back later.')
            #if guest:
             #   st.write(f'However, it looks like you selected {guest} as a filter.')

            # make hybrid call to weaviate
            display_properties = ["title", "guest","summary","content","thumbnail_url",
                                  "episode_url", "length","doc_id","views"]
            hybrid_response = client.hybrid_search(query,
                                                   class_name,
                                                   alpha=alpha_input,
                                                   display_properties=display_properties,
                                                   where_filter=guest_filter,
                                                   limit=retrieval_limit)
            # rerank results
            ranked_response = reranker.rerank(hybrid_response,
                                              query,
                                              apply_sigmoid=True,
                                              top_k=reranker_topk)
            # validate token count is below threshold
            valid_response = validate_token_threshold(ranked_response,
                                                      question_answering_prompt_series,
                                                      query=query,
                                                      tokenizer= encoding,
                                                      token_threshold=4000, 
                                                      verbose=True)
        
            # generate LLM prompt
            prompt = generate_prompt_series(query=query, results=valid_response)
            
            # prep for streaming response
            st.subheader("Response from Impact Theory (context)")
            with st.spinner('Generating Response...'):
                st.markdown("----")
                #creates container for LLM response
                chat_container, response_box = [], st.empty()
                
                # execute chat call to LLM

                for resp in llm.get_chat_completion(prompt=prompt,
                                                    temperature=temperature_input,
                                                    max_tokens=500,
                                                    show_response=True,
                                                    stream=True):
                    try:
                          #inserts chat stream from LLM
                        with response_box:
                            content = resp.choices[0].delta.content
                            if content:
                                chat_container.append(content)
                                result = "".join(chat_container).strip()
                                st.write(f'{result}')
                                
                    except Exception as e:
                        print(e)
            if result == "I cannot answer the question given the context.":
                st.write("Please try writing another question or select a question from the list.")
            else:


                st.subheader("Search Results")
                for i, hit in enumerate(valid_response):
                    col1, col2 = st.columns([7, 3], gap='large')
                    image = hit["thumbnail_url"]
                    episode_url = hit["episode_url"]
                    title = hit["title"]
                    show_length = hit["length"]
                    time_string = convert_seconds(show_length)

                    with col1:
                        st.write( search_result(  i=i, 
                                                    url=episode_url,
                                                    guest=hit['guest'],
                                                    title=title,
                                                    content=hit['content'], 
                                                    length=time_string),
                                                    unsafe_allow_html=True)
                        st.write('\n\n')

                        with st.expander("Click Here for Guest Info:"):
                            try:
                                input = wikipedia.page(hit['guest'], auto_suggest=False)
                                podcast_guest_info = input.summary
                                st.write(podcast_guest_info)
                            except Exception as e:
                                print(e)

                    with col2:
                        # st.write(f"<a href={episode_url} <img src={image} width='200'></a>", 
                        #             unsafe_allow_html=True)
                        st.image(image, caption=title.split('|')[0], width=200, use_column_width=False)

if __name__ == '__main__':
    main()