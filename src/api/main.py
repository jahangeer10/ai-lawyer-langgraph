"""
FastAPI backend for AI Lawyer Agent
"""

import os
import tempfile
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
import uvicorn
from dotenv import load_dotenv

from ..agents.legal_graph import LegalAgentGraph, LegalAgentState
from ..utils.document_processor import DocumentProcessor
from ..utils.context_manager import format_legal_response

# Load environment variables
load_dotenv()

app = FastAPI(
    title="AI Lawyer Agent",
    description="AI-powered legal assistant for Indian law using LangGraph",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components with error handling
try:
    legal_graph = LegalAgentGraph()
    print("✓ Legal graph initialized successfully")
except Exception as e:
    print(f"⚠ Warning: Legal graph initialization failed: {e}")
    legal_graph = None

try:
    document_processor = DocumentProcessor()
    print("✓ Document processor initialized successfully")
except Exception as e:
    print(f"⚠ Warning: Document processor initialization failed: {e}")
    document_processor = None

# Pydantic models
class ChatMessage(BaseModel):
    content: str
    role: str = "user"

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    legal_domain: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    legal_domain: str
    confidence: float
    citations: List[str]
    conversation_id: str

class DocumentAnalysisRequest(BaseModel):
    document_ids: List[str]
    query: Optional[str] = None

class DocumentUploadResponse(BaseModel):
    document_id: str
    file_name: str
    file_type: str
    status: str
    message: str

# In-memory conversation storage (use Redis/DB in production)
conversations: Dict[str, List[Any]] = {}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Lawyer Agent API",
        "version": "1.0.0",
        "status": "active",
        "components": {
            "legal_graph": legal_graph is not None,
            "document_processor": document_processor is not None
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "ai-lawyer-agent",
        "components": {
            "legal_graph": "active" if legal_graph else "inactive",
            "document_processor": "active" if document_processor else "inactive"
        }
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint for legal queries"""
    if not legal_graph:
        raise HTTPException(status_code=503, detail="Legal graph service unavailable")
    
    try:
        conversation_id = request.conversation_id or "default"
        
        # Get or create conversation
        if conversation_id not in conversations:
            conversations[conversation_id] = []
        
        # Add user message
        user_message = HumanMessage(content=request.message)
        conversations[conversation_id].append(user_message)
        
        # Prepare state
        state: LegalAgentState = {
            "messages": conversations[conversation_id],
            "current_agent": None,
            "legal_domain": request.legal_domain,
            "case_context": {},
            "documents": [],
            "research_results": [],
            "legal_advice": None,
            "confidence_score": None,
            "citations": []
        }
        
        # Process through LangGraph
        result = await legal_graph.ainvoke(state)
        
        # Update conversation
        conversations[conversation_id] = result["messages"]
        
        # Extract response
        ai_response = result["messages"][-1].content
        legal_domain = result.get("legal_domain", "general")
        confidence = result.get("confidence_score", 0.0)
        citations = result.get("citations", [])
        
        # Format response
        formatted_response = format_legal_response(ai_response, citations, confidence)
        
        return ChatResponse(
            response=formatted_response,
            legal_domain=legal_domain,
            confidence=confidence,
            citations=citations,
            conversation_id=conversation_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

# Simple fallback chat endpoint for when LangGraph is unavailable
@app.post("/simple-chat")
async def simple_chat(request: ChatRequest):
    """Simple chat endpoint without LangGraph"""
    return {
        "response": f"Thank you for your query: '{request.message}'. The AI Lawyer Agent is currently initializing. Please ensure you have set up your OpenAI API key in the .env file and try again.",
        "legal_domain": "general",
        "confidence": 0.0,
        "citations": [],
        "conversation_id": request.conversation_id or "default"
    }

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )