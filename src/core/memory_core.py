import time
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class MemoryCore:
    def __init__(self, capacity: int = 10, embedding_model: str = "all-MiniLM-L6-v2"):
        self.capacity = capacity
        self.embedding_model = SentenceTransformer(embedding_model)
        self.long_term_memory: List[Dict[str, Any]] = []
        self.short_term_memory: List[Dict[str, Any]] = []
        
    def _get_embedding(self, text: str) -> np.ndarray:
        return self.embedding_model.encode([text])[0]
    
    def _calculate_relevance(self, query_embedding: np.ndarray, memory_item: Dict[str, Any]) -> float:
        return cosine_similarity([query_embedding], [memory_item["embedding"]])[0][0]
    
    def add_to_memory(self, text: str, importance: float = 0.5):
        embedding = self._get_embedding(text)
        memory_item = {
            "text": text,
            "embedding": embedding,
            "importance": importance,
            "timestamp": time.time()
        }
        
        self.short_term_memory.append(memory_item)
        
        if len(self.short_term_memory) > self.capacity:
            self._compress_and_merge()
    
    def _compress_and_merge(self):
        self.short_term_memory.sort(key=lambda x: x["importance"], reverse=True)
        
        items_to_keep = self.short_term_memory[:self.capacity]
        
        self.long_term_memory.extend(items_to_keep)
        
        self.short_term_memory = []
    
    def retrieve_relevant_context(self, query: str, top_k: int = 3) -> List[str]:
        query_embedding = self._get_embedding(query)
        
        all_memory = self.long_term_memory + self.short_term_memory
        
        relevance_scores = [
            (item, self._calculate_relevance(query_embedding, item))
            for item in all_memory
        ]
        
        relevant_items = sorted(relevance_scores, key=lambda x: x[1], reverse=True)[:top_k]
        
        return [item[0]["text"] for item in relevant_items]
    
    def get_memory_summary(self) -> str:
        return f"Long-term memory: {len(self.long_term_memory)} items\nShort-term memory: {len(self.short_term_memory)} items" 