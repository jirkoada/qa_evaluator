from typing import List
import fasttext

class FTembeddings:

    def __init__(self, model_path):
        self.model = fasttext.load_model(model_path)
        
    def embed_documents(self, texts: List[str], chunk_size=0) -> List[List[float]]:
        return [self.model.get_sentence_vector(t.lower().replace('\n', ' ')) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self.model.get_sentence_vector(text.lower().replace('\n', ' ')).tolist()