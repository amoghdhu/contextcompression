from src.core.context_manager import ContextManager

def main():
    manager = ContextManager(memory_capacity=5)
    
    initial_knowledge = [
        "The Earth orbits around the Sun.",
        "Water boils at 100 degrees Celsius at sea level.",
        "Python is a popular programming language.",
        "The human body has 206 bones.",
        "Photosynthesis is the process by which plants convert light energy into chemical energy."
    ]
    
    for knowledge in initial_knowledge:
        manager.add_to_memory(knowledge, importance=0.8)
    
    queries = [
        "What is photosynthesis?",
        "What temperature does water boil at?",
        "Tell me about Python programming.",
        "How many bones are in the human body?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        response = manager.process_query(query)
        print(f"Response: {response}")
        print(f"Memory Summary: {manager.get_memory_summary()}")

if __name__ == "__main__":
    main() 