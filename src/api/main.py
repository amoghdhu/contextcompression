from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np
from src.core.context_manager import ContextManager
import json
import os

app = FastAPI(title="Context Compression Visualization")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = ContextManager(memory_capacity=5)

app.mount("/static", StaticFiles(directory="src/api/static"), name="static")

class Query(BaseModel):
    text: str

class MemoryItem(BaseModel):
    text: str
    importance: float
    timestamp: float
    type: str  

@app.get("/")
async def read_root():
    return {"message": "Context Compression Visualization API"}

@app.post("/query")
async def process_query(query: Query):
    try:
        response = manager.process_query(query.text)
        return {
            "response": response,
            "memory_state": manager.get_memory_summary()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory")
async def get_memory_state():
    try:
        memory_items = []
        
        for item in manager.memory_core.long_term_memory:
            memory_items.append({
                "text": item["text"],
                "importance": item["importance"],
                "timestamp": item["timestamp"],
                "type": "long_term",
                "embedding": item["embedding"].tolist()  
            })
        
        for item in manager.memory_core.short_term_memory:
            memory_items.append({
                "text": item["text"],
                "importance": item["importance"],
                "timestamp": item["timestamp"],
                "type": "short_term",
                "embedding": item["embedding"].tolist() 
            })
        
        return {
            "items": memory_items,
            "summary": manager.get_memory_summary()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add-to-memory")
async def add_to_memory(item: MemoryItem):
    try:
        manager.add_to_memory(item.text, item.importance)
        return {"message": "Item added to memory successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear-memory")
async def clear_memory():
    try:
        manager.clear_memory()
        return {"message": "Memory cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 