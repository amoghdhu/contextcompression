# Dynamic Context Compression

A system that compresses and evolves a language model's context over time, allowing it to maintain coherence across long conversations or tasks.


## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your Groq API key:
   - Create a `.env` file in the project root
   - Add your API key: `GROQ_API_KEY="your-api-key-here"`


## Example

Run the example script to see the system in action:
```bash
uvicorn src.api.main:app --reload
```


## How it Works

1. Information is stored with embeddings for semantic search
2. Short-term memory is maintained for recent interactions
3. When capacity is reached, important information is compressed and moved to long-term memory
4. Queries are processed by retrieving relevant context and generating responses
5. All interactions are stored in memory for future reference 