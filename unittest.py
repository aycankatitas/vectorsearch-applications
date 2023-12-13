# Unittests

class TestSplitContents(unittest.TestCase):
    '''
    Unit test to ensure proper functionality of split_contents function
    '''

    def test_split_contents(self):
        import tiktoken
        from llama_index.text_splitter import SentenceSplitter

        data = load_impact_theory_data()

        subset = data[:3]
        chunk_size = 256
        chunk_overlap = 0
        encoding = tiktoken.encoding_for_model('gpt-3.5-turbo-0613')
        gpt35_txt_splitter = SentenceSplitter(chunk_size=chunk_size, tokenizer=encoding.encode, chunk_overlap=chunk_overlap)
        results = split_contents(subset, gpt35_txt_splitter)
        self.assertEqual(len(results), 3)
        self.assertEqual(len(results[0]), 83)
        self.assertEqual(len(results[1]), 178)
        self.assertEqual(len(results[2]), 144)
        self.assertTrue(isinstance(results, list))
        self.assertTrue(isinstance(results[0], list))
        self.assertTrue(isinstance(results[0][0], str))

class TestEncodeContentSplits(unittest.TestCase):
    '''
    Unit test to ensure proper functionality of split_contents function
    '''

    def test_encode_content_splits(self):
        import tiktoken
        from numpy import ndarray
        from llama_index.text_splitter import SentenceSplitter
        from sentence_transformers import SentenceTransformer

        data = load_impact_theory_data()

        #get splits first
        subset = data[:3]
        chunk_size = 256
        chunk_overlap = 0
        encoding = tiktoken.encoding_for_model('gpt-3.5-turbo-0613')
        gpt35_txt_splitter = SentenceSplitter(chunk_size=chunk_size, tokenizer=encoding.encode, chunk_overlap=chunk_overlap)
        splits = split_contents(subset, gpt35_txt_splitter)

        #encode splits
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        results = encode_content_splits(splits, model)

        #run assertion tests
        self.assertEqual(len(results), 3)
        self.assertEqual(len(results[0]), 83)
        self.assertEqual(len(results[1]), 178)
        self.assertEqual(len(results[2]), 144)
        self.assertTrue(isinstance(results, list))
        self.assertTrue(isinstance(results[0], list))
        self.assertTrue(isinstance(results[0][0], tuple))
        self.assertTrue(isinstance(results[0][0][0], str))
        self.assertTrue(isinstance(results[0][0][1], ndarray))

        import unittest

class TestEncodeContentSplits(unittest.TestCase):
    '''
    Unit test to ensure proper functionality of split_contents function
    '''

    def test_encode_content_splits(self):
        import tiktoken
        from numpy import ndarray
        from llama_index.text_splitter import SentenceSplitter
        from sentence_transformers import SentenceTransformer

        data = load_impact_theory_data()

        #get splits first
        subset = data[:3]
        chunk_size = 256
        chunk_overlap = 0
        encoding = tiktoken.encoding_for_model('gpt-3.5-turbo-0613')
        gpt35_txt_splitter = SentenceSplitter(chunk_size=chunk_size, tokenizer=encoding.encode, chunk_overlap=chunk_overlap)
        splits = split_contents(subset, gpt35_txt_splitter)

        #encode splits
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        text_vector_tuples = encode_content_splits(splits, model)

        #joint metadata
        results = join_metadata(subset, text_vector_tuples)
        #run assertion tests
        self.assertEqual(len(results), 405)
        self.assertEqual(len(results[0]), 12)
        self.assertTrue(isinstance(results, list))
        self.assertTrue(isinstance(results[0], dict))
        self.assertTrue(isinstance(results[0]['content'], str))
        self.assertEqual(results[0]['doc_id'], 'nXJBccSwtB8_0')
        self.assertEqual(len(results[0]['content_embedding']), 384)
        self.assertTrue(isinstance(results[0]['content_embedding'], list))