# Dataset Creation Functions
def create_dataset(corpus: List[dict],
                   embedding_model: SentenceTransformer,
                   text_splitter: SentenceSplitter,
                   file_outpath_prefix: str='./impact-theory-minilmL6',
                   content_field: str='content',
                   embedding_field: str='content_embedding',
                   device: str='cuda:0' if torch.cuda.is_available() else 'cpu'
                   ) -> None:
    '''
    Given a raw corpus of data, this function creates a new dataset where each dataset
    doc contains episode metadata and it's associated text chunk and vector representation.
    Output is directly saved to disk.
    '''

    io = FileIO()

    chunk_size = text_splitter.chunk_size
    print(f'Creating dataset using chunk_size: {chunk_size}')
    start = time.perf_counter()

    content_splits = split_contents(corpus,text_splitter)
    text_vector_tuples = encode_content_splits(content_splits, embedding_model,device)
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