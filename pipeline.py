# Dataset Creation Functions
from torch import cuda
from typing import List, Tuple, Dict, Any
from sentence_transformers import SentenceTransformer
from llama_index.text_splitter import SentenceSplitter
from torch import cuda
import time
from preprocessing import FileIO
from tqdm.notebook import tqdm
import numpy as np
import unittest
from unittest_utils import TestSplitContents, TestEncodeContentSplits, TestEncodeContentSplitsDF

from llama_index.finetuning import EmbeddingQAFinetuneDataset
from retrieval_evaluation import calc_hit_rate_scores, calc_mrr_scores, record_results, add_params
from weaviate_interface import WeaviateClient



def create_dataset(corpus: List[dict],
                   embedding_model: SentenceTransformer,
                   text_splitter: SentenceSplitter,
                   file_outpath_prefix: str='./impact-theory-minilmL6',
                   content_field: str='content',
                   embedding_field: str='content_embedding',
                   device: str='cuda:0' if cuda.is_available() else 'cpu'
                   ) -> None:
    '''
    Given a raw corpus of data, this function creates a new dataset where each dataset
    doc contains episode metadata and it's associated text chunk and vector representation.
    Output is directly saved to disk.
    '''

    io = FileIO()

    # Chunk Settings
    chunk_size = text_splitter.chunk_size
    print(f'Creating dataset using chunk_size: {chunk_size}')
    start = time.perf_counter()


    # Split Corpus
    content_splits = split_contents(corpus,text_splitter)

    # Vector Embeddings
    text_vector_tuples = encode_content_splits(content_splits, embedding_model,device)

    # Join Metadata
    joined_docs = join_metadata(corpus, text_vector_tuples)
  
    file_path = f'{file_outpath_prefix}-{chunk_size}.parquet'
    io.save_as_parquet(file_path=file_path, data=joined_docs, overwrite=False)
    end = time.perf_counter() - start
    print(f'Total Time to process dataset of chunk_size ({chunk_size}): {round(end/60, 2)} minutes')



def split_contents(corpus: List[dict],
                   text_splitter: SentenceSplitter,
                   content_field: str='content'
                   ) -> List[List[str]]:
    '''
    Given a corpus of "documents" with text content, this function splits the
    content field into chunks sizes as specified by the text_splitter.

    Example
    -------
    corpus = [
            {'title': 'This is a cool show', 'content': 'There is so much good content on this show. \
              This would normally be a really long block of content. ... But for this example it will not be.'},
            {'title': 'Another Great Show', 'content': 'The content here is really good as well.  If you are \
              reading this you have too much time on your hands. ... More content, blah, blah.'}
           ]

    output = split_contents(data, text_splitter, content_field="content")

    output >>> [['There is so much good content on this show.', 'This would normally be a really long block of content.', \
                 'But for this example it will not be'],
                ['The content here is really good as well.', 'If you are reading this you have too much time on your hands.', \
                 'More content, blah, blah.']
                ]
    '''

    chunks = [text_splitter.split_text(episode[content_field]) for episode in tqdm(corpus)]

    return chunks

def encode_content_splits(content_splits: List[List[str]],
                          embedding_model: SentenceTransformer,
                          device: str='cuda:0'
                          ) -> List[List[Tuple[str, np.array]]]:
    '''
    Encode content splits as vector embeddings from a list of content splits
    where each list of splits is a single podcast episode.

    Example
    -------
    content_splits =  [['There is so much good content on this show.', 'This would normally be a really long block of content.'],
                       ['The content here is really good as well.', 'More content, blah, blah.']
                      ]

    output = encode_content_splits(content_splits, model)

    output >>> [
          EPISODE 1 -> [('There is so much good content on this show.', array[ 1.78036056e-02, -1.93265956e-02,  3.61164124e-03, -5.89650944e-02,
                                                                         1.91510320e-02,  1.60808843e-02,  1.13610983e-01,  3.59948091e-02,
                                                                        -1.73066761e-02, -3.30348089e-02, -1.00898169e-01,  2.34847311e-02]
                                                                        )
                         tuple(text, np.array), tuple(text, np.array), tuple(text, vector)....],
          EPISODE 2 ->  [tuple(text, np.array), tuple(text, np.array), tuple(text, vector)....],
          EPISODE 3 ->  [tuple(text, np.array), tuple(text, np.array), tuple(text, vector)....],
          EPISODE n ... [tuple(text, np.array), tuple(text, np.array), tuple(text, vector)....]
    '''

    text_vector_tuples = []

    vector = [embedding_model.encode(splits) for splits in tqdm(content_splits)]
    text_vector_tuples = [list(zip(content_splits[i], vector[i])) for i in range(len(content_splits))]


    return text_vector_tuples

def join_metadata(corpus: List[dict],
                  text_vector_list: List[List[Tuple[str, np.array]]],
                  content_field: str='content',
                  embedding_field: str='content_embedding'
                 ) -> List[dict]:
    '''
    Combine episode metadata from original corpus with text/vectors tuples.
    Creates a new dictionary for each text/vector combination.
    '''

    joined_documents = []

    for i,n in enumerate(corpus):
      for j,k in enumerate(text_vector_list[i]):
        new_dict= {key:value for key, value in n.items() if key !="content"}
        doc_id = f"{n['video_id']}_{j}"
        new_dict["doc_id"] = doc_id
        new_dict[content_field]=k[0]
        new_dict[embedding_field]=k[1].tolist()
        joined_documents.append(new_dict)

    return joined_documents

def retrieval_evaluation(dataset: EmbeddingQAFinetuneDataset, 
                         class_name: str, 
                         retriever: WeaviateClient,
                         retrieve_limit: int=5,
                         chunk_size: int=256,
                         hnsw_config_keys: List[str]=['maxConnections', 'efConstruction', 'ef'],
                         display_properties: List[str]=['doc_id', 'guest', 'content'],
                         dir_outpath: str='./eval_results',
                         include_miss_info: bool=False,
                         user_def_params: Dict[str,Any]=None
                         ) -> Dict[str, str|int|float]:
    '''
    Given a dataset and a retriever evaluate the performance of the retriever. Returns a dict of kw and vector
    hit rates and mrr scores. If inlude_miss_info is True, will also return a list of kw and vector responses 
    and their associated queries that did not return a hit, for deeper analysis. Text file with results output
    is automatically saved in the dir_outpath directory.

    Args:
    -----
    dataset: EmbeddingQAFinetuneDataset
        Dataset to be used for evaluation
    class_name: str
        Name of Class on Weaviate host to be used for retrieval
    retriever: WeaviateClient
        WeaviateClient object to be used for retrieval 
    retrieve_limit: int=5
        Number of documents to retrieve from Weaviate host
    chunk_size: int=256
        Number of tokens used to chunk text. This value is purely for results 
        recording purposes and does not affect results. 
    display_properties: List[str]=['doc_id', 'content']
        List of properties to be returned from Weaviate host for display in response
    dir_outpath: str='./eval_results'
        Directory path for saving results.  Directory will be created if it does not
        already exist. 
    include_miss_info: bool=False
        Option to include queries and their associated kw and vector response values
        for queries that are "total misses"
    user_def_params : dict=None
        Option for user to pass in a dictionary of user-defined parameters and their values.
    '''

    results_dict = {'n':retrieve_limit, 
                    'Retriever': retriever.model_name_or_path, 
                    'chunk_size': chunk_size,
                    'kw_hit_rate': 0,
                    'kw_mrr': 0,
                    'vector_hit_rate': 0,
                    'vector_mrr': 0,
                    'total_misses': 0,
                    'total_questions':0
                    }
    #add hnsw configs and user defined params (if any)
    results_dict = add_params(retriever, class_name, results_dict, user_def_params, hnsw_config_keys)
    
    start = time.perf_counter()
    miss_info = []
    for query_id, q in tqdm(dataset.queries.items(), 'Queries'):
        results_dict['total_questions'] += 1
        hit = False
        #make Keyword, Vector, and Hybrid calls to Weaviate host
        try:
            kw_response = retriever.keyword_search(request=q, class_name=class_name, limit=retrieve_limit, display_properties=display_properties)
            vector_response = retriever.vector_search(request=q, class_name=class_name, limit=retrieve_limit, display_properties=display_properties)
            
            #collect doc_ids and position of doc_ids to check for document matches
            kw_doc_ids = {result['doc_id']:i for i, result in enumerate(kw_response, 1)}
            vector_doc_ids = {result['doc_id']:i for i, result in enumerate(vector_response, 1)}
            
            #extract doc_id for scoring purposes
            doc_id = dataset.relevant_docs[query_id][0]
     
            #increment hit_rate counters and mrr scores
            if doc_id in kw_doc_ids:
                results_dict['kw_hit_rate'] += 1
                results_dict['kw_mrr'] += 1/kw_doc_ids[doc_id]
                hit = True
            if doc_id in vector_doc_ids:
                results_dict['vector_hit_rate'] += 1
                results_dict['vector_mrr'] += 1/vector_doc_ids[doc_id]
                hit = True
                
            # if no hits, let's capture that
            if not hit:
                results_dict['total_misses'] += 1
                miss_info.append({'query': q, 'kw_response': kw_response, 'vector_response': vector_response})
        except Exception as e:
            print(e)
            continue
    

    #use raw counts to calculate final scores
    calc_hit_rate_scores(results_dict)
    calc_mrr_scores(results_dict)
    
    end = time.perf_counter() - start
    print(f'Total Processing Time: {round(end/60, 2)} minutes')
    record_results(results_dict, chunk_size, dir_outpath=dir_outpath, as_text=True)
    
    if include_miss_info:
        return results_dict, miss_info
    return results_dict