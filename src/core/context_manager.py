from typing import List, Optional
from .memory_core import MemoryCore
from src.models.llm import GroqLLM

class ContextManager:
    def __init__(self, memory_capacity: int = 10, model_name: str = "llama-3.3-70b-versatile"):
        self.memory_core = MemoryCore(capacity=memory_capacity)
        self.llm = GroqLLM(model_name=model_name)
        
    def process_query(self, query: str, max_context_items: int = 3) -> str:
        relevant_context = self.memory_core.retrieve_relevant_context(query, top_k=max_context_items)
        
        context_str = "\n".join(relevant_context)
        prompt = f"Context:\n{context_str}\n\nQuestion: {query}"
        
        response = self.llm.generate(prompt)
        
        self.memory_core.add_to_memory(f"Q: {query}\nA: {response}", importance=0.7)
        
        return response
    
    def add_to_memory(self, text: str, importance: float = 0.5):
        self.memory_core.add_to_memory(text, importance)
    
    def get_memory_summary(self) -> str:
        return self.memory_core.get_memory_summary()
    
    def clear_memory(self):
        self.memory_core = MemoryCore(capacity=self.memory_core.capacity) 