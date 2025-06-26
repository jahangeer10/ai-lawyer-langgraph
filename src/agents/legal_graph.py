
"""
Main LangGraph workflow for AI Lawyer Agent
Implements multi-agent system for Indian legal assistance
"""

import os
from typing import TypedDict, List, Dict, Any, Optional, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from .legal_agents import (
    ConstitutionalLawAgent,
    CriminalLawAgent,
    CivilLawAgent,
    CorporateLawAgent,
    FamilyLawAgent,
    TaxLawAgent,
    LegalResearchAgent,
    DocumentAnalysisAgent
)
from ..utils.context_manager import trim_messages, count_tokens
from ..utils.legal_knowledge import LegalKnowledgeBase

class LegalAgentState(TypedDict):
    """State for the legal agent workflow"""
    messages: List[Any]
    current_agent: Optional[str]
    legal_domain: Optional[str]
    case_context: Dict[str, Any]
    documents: List[Dict[str, Any]]
    research_results: List[Dict[str, Any]]
    legal_advice: Optional[str]
    confidence_score: Optional[float]
    citations: List[str]

class LegalAgentGraph:
    """Main LangGraph workflow for AI Lawyer Agent"""
    
    def __init__(self, openai_api_key: str = None):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.1,
            api_key=self.openai_api_key
        )
        
        # Initialize specialized agents
        self.agents = {
            "constitutional": ConstitutionalLawAgent(self.llm),
            "criminal": CriminalLawAgent(self.llm),
            "civil": CivilLawAgent(self.llm),
            "corporate": CorporateLawAgent(self.llm),
            "family": FamilyLawAgent(self.llm),
            "tax": TaxLawAgent(self.llm),
            "research": LegalResearchAgent(self.llm),
            "document_analysis": DocumentAnalysisAgent(self.llm)
        }
        
        self.knowledge_base = LegalKnowledgeBase()
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        graph = StateGraph(LegalAgentState)
        
        # Add nodes
        graph.add_node("router", self._route_query)
        graph.add_node("constitutional_agent", self._constitutional_handler)
        graph.add_node("criminal_agent", self._criminal_handler)
        graph.add_node("civil_agent", self._civil_handler)
        graph.add_node("corporate_agent", self._corporate_handler)
        graph.add_node("family_agent", self._family_handler)
        graph.add_node("tax_agent", self._tax_handler)
        graph.add_node("synthesizer", self._synthesize_response)
        
        # Add edges
        graph.add_edge(START, "router")
        
        # Conditional edges from router
        graph.add_conditional_edges(
            "router",
            self._route_decision,
            {
                "constitutional": "constitutional_agent",
                "criminal": "criminal_agent",
                "civil": "civil_agent",
                "corporate": "corporate_agent",
                "family": "family_agent",
                "tax": "tax_agent",
                "synthesizer": "synthesizer",
            }
        )

        # All agents go to synthesizer
        graph.add_edge("constitutional_agent", "synthesizer")
        graph.add_edge("criminal_agent", "synthesizer")
        graph.add_edge("civil_agent", "synthesizer")
        graph.add_edge("corporate_agent", "synthesizer")
        graph.add_edge("family_agent", "synthesizer")
        graph.add_edge("tax_agent", "synthesizer")
        graph.add_edge("synthesizer", END)
        
        return graph.compile()
    
    def _route_query(self, state: LegalAgentState) -> LegalAgentState:
        """Route query to appropriate legal domain"""
        messages = state["messages"]
        latest_message = messages[-1].content if messages else ""
        
        # Simple keyword-based routing (can be enhanced with ML classification)
        domain = "general"

        if any(word in latest_message.lower() for word in ["constitution", "fundamental rights", "article"]):
            domain = "constitutional"
        elif any(word in latest_message.lower() for word in ["crime", "criminal", "ipc", "murder", "theft"]):
            domain = "criminal"
        elif any(word in latest_message.lower() for word in ["contract", "property", "tort", "civil"]):
            domain = "civil"
        elif any(word in latest_message.lower() for word in ["company", "shareholder", "corporate"]):
            domain = "corporate"
        elif any(word in latest_message.lower() for word in ["marriage", "divorce", "custody"]):
            domain = "family"
        elif any(word in latest_message.lower() for word in ["tax", "income tax", "gst"]):
            domain = "tax"
        
        state["legal_domain"] = domain
        state["current_agent"] = domain
        
        return state
    
    def _route_decision(self, state: LegalAgentState) -> str:
        """Decide which agent to route to"""
        domain = state.get("legal_domain", "general")
        
        if domain in [
            "constitutional",
            "criminal",
            "civil",
            "corporate",
            "family",
            "tax",
        ]:
            return domain
        return "synthesizer"
    
    def _constitutional_handler(self, state: LegalAgentState) -> LegalAgentState:
        """Handle constitutional law queries"""
        agent = self.agents["constitutional"]
        result = agent.process_query(state)
        
        state["legal_advice"] = result["advice"]
        state["confidence_score"] = result["confidence"]
        state["citations"] = result["citations"]
        
        # Add AI response to messages
        ai_message = AIMessage(content=result["advice"])
        state["messages"].append(ai_message)
        
        return state
    
    def _criminal_handler(self, state: LegalAgentState) -> LegalAgentState:
        """Handle criminal law queries"""
        agent = self.agents["criminal"]
        result = agent.process_query(state)
        
        state["legal_advice"] = result["advice"]
        state["confidence_score"] = result["confidence"]
        state["citations"] = result["citations"]
        
        # Add AI response to messages
        ai_message = AIMessage(content=result["advice"])
        state["messages"].append(ai_message)

        return state

    def _civil_handler(self, state: LegalAgentState) -> LegalAgentState:
        """Handle civil law queries"""
        agent = self.agents["civil"]
        result = agent.process_query(state)

        state["legal_advice"] = result["advice"]
        state["confidence_score"] = result["confidence"]
        state["citations"] = result["citations"]

        ai_message = AIMessage(content=result["advice"])
        state["messages"].append(ai_message)

        return state

    def _corporate_handler(self, state: LegalAgentState) -> LegalAgentState:
        agent = self.agents["corporate"]
        result = agent.process_query(state)

        state["legal_advice"] = result["advice"]
        state["confidence_score"] = result["confidence"]
        state["citations"] = result["citations"]

        ai_message = AIMessage(content=result["advice"])
        state["messages"].append(ai_message)

        return state

    def _family_handler(self, state: LegalAgentState) -> LegalAgentState:
        agent = self.agents["family"]
        result = agent.process_query(state)

        state["legal_advice"] = result["advice"]
        state["confidence_score"] = result["confidence"]
        state["citations"] = result["citations"]

        ai_message = AIMessage(content=result["advice"])
        state["messages"].append(ai_message)

        return state

    def _tax_handler(self, state: LegalAgentState) -> LegalAgentState:
        agent = self.agents["tax"]
        result = agent.process_query(state)

        state["legal_advice"] = result["advice"]
        state["confidence_score"] = result["confidence"]
        state["citations"] = result["citations"]

        ai_message = AIMessage(content=result["advice"])
        state["messages"].append(ai_message)

        return state
    
    def _synthesize_response(self, state: LegalAgentState) -> LegalAgentState:
        """Synthesize final response"""
        if not state.get("legal_advice"):
            # Fallback for general queries
            messages = state["messages"]
            latest_query = messages[-1].content if messages else ""
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an AI legal assistant for Indian law. 
                Provide helpful legal information while clearly stating that this is not legal advice 
                and users should consult qualified lawyers for specific legal matters."""),
                ("human", "{query}")
            ])
            
            response = self.llm.invoke(prompt.format_messages(query=latest_query))
            
            state["legal_advice"] = response.content
            state["confidence_score"] = 0.7
            state["citations"] = ["General legal information"]
            
            # Add AI response to messages
            ai_message = AIMessage(content=response.content)
            state["messages"].append(ai_message)
        
        return state
    
    async def ainvoke(self, state: LegalAgentState) -> LegalAgentState:
        """Async invoke the graph"""
        return await self.graph.ainvoke(state)
    
    def invoke(self, state: LegalAgentState) -> LegalAgentState:
        """Sync invoke the graph"""
        return self.graph.invoke(state)
    
    async def astream(self, state: LegalAgentState, stream_mode: str = "messages"):
        """Async stream the graph execution"""
        async for event in self.graph.astream(state, stream_mode=stream_mode):
            yield event
