
"""
Specialized legal agents for different areas of Indian law
"""

from typing import Dict, List, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from ..utils.legal_knowledge import LegalKnowledgeBase

class BaseLegalAgent:
    """Base class for all legal agents"""
    
    def __init__(self, llm, domain: str):
        self.llm = llm
        self.domain = domain
        self.knowledge_base = LegalKnowledgeBase()
    
    def process_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process legal query and return structured response"""
        raise NotImplementedError

class ConstitutionalLawAgent(BaseLegalAgent):
    """Agent specialized in Constitutional Law"""
    
    def __init__(self, llm):
        super().__init__(llm, "constitutional")
    
    def process_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        messages = state["messages"]
        latest_query = messages[-1].content if messages else ""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in Indian Constitutional Law. 
            Provide detailed analysis covering:
            
            1. Relevant Constitutional Articles
            2. Fundamental Rights implications
            3. Directive Principles of State Policy
            4. Landmark Supreme Court judgments
            5. Constitutional remedies available
            
            Key areas: Fundamental Rights (Articles 12-35), DPSP (Articles 36-51), 
            Emergency provisions, Amendment procedures, Judicial review.
            
            Always cite specific Articles and landmark cases."""),
            ("human", "{query}")
        ])
        
        response = self.llm.invoke(prompt.format_messages(query=latest_query))
        
        return {
            "advice": response.content,
            "citations": self._extract_citations(response.content),
            "confidence": 0.85
        }
    
    def _extract_citations(self, content: str) -> List[str]:
        """Extract legal citations from response"""
        citations = []
        if "Article" in content:
            citations.append("Constitutional Articles referenced")
        if "Supreme Court" in content or "SC" in content:
            citations.append("Supreme Court judgments")
        return citations

class CriminalLawAgent(BaseLegalAgent):
    """Agent specialized in Criminal Law"""
    
    def __init__(self, llm):
        super().__init__(llm, "criminal")
    
    def process_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        messages = state["messages"]
        latest_query = messages[-1].content if messages else ""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in Indian Criminal Law. 
            Provide comprehensive analysis covering:
            
            1. Relevant IPC sections
            2. CrPC procedural aspects
            3. Evidence Act provisions
            4. Bail considerations
            5. Punishment and sentencing guidelines
            
            Key statutes: IPC 1860, CrPC 1973, Evidence Act 1872, 
            Special criminal laws (POCSO, SC/ST Act, etc.)
            
            Always cite specific sections and relevant case law."""),
            ("human", "{query}")
        ])
        
        response = self.llm.invoke(prompt.format_messages(query=latest_query))
        
        return {
            "advice": response.content,
            "citations": self._extract_citations(response.content),
            "confidence": 0.85
        }
    
    def _extract_citations(self, content: str) -> List[str]:
        citations = []
        if "Section" in content or "IPC" in content:
            citations.append("IPC sections referenced")
        if "CrPC" in content:
            citations.append("CrPC provisions")
        return citations

# Additional agent classes would be implemented similarly
class CivilLawAgent(BaseLegalAgent):
    def __init__(self, llm):
        super().__init__(llm, "civil")

    def process_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        messages = state["messages"]
        latest_query = messages[-1].content if messages else ""

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are an expert in Indian Civil Law. Provide key statutes and case law relevant to the query.""",
            ),
            ("human", "{query}"),
        ])
        response = self.llm.invoke(prompt.format_messages(query=latest_query))

        return {
            "advice": response.content,
            "citations": [],
            "confidence": 0.8,
        }

class CorporateLawAgent(BaseLegalAgent):
    def __init__(self, llm):
        super().__init__(llm, "corporate")

    def process_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        messages = state["messages"]
        latest_query = messages[-1].content if messages else ""

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are an expert in Indian Corporate Law. Discuss relevant sections of the Companies Act and notable judgments.""",
            ),
            ("human", "{query}"),
        ])

        response = self.llm.invoke(prompt.format_messages(query=latest_query))

        return {
            "advice": response.content,
            "citations": [],
            "confidence": 0.8,
        }

class FamilyLawAgent(BaseLegalAgent):
    def __init__(self, llm):
        super().__init__(llm, "family")

    def process_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        messages = state["messages"]
        latest_query = messages[-1].content if messages else ""

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are an expert in Indian Family Law. Explain how personal laws apply to the query.""",
            ),
            ("human", "{query}"),
        ])

        response = self.llm.invoke(prompt.format_messages(query=latest_query))

        return {
            "advice": response.content,
            "citations": [],
            "confidence": 0.8,
        }

class TaxLawAgent(BaseLegalAgent):
    def __init__(self, llm):
        super().__init__(llm, "tax")

    def process_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        messages = state["messages"]
        latest_query = messages[-1].content if messages else ""

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are an expert in Indian Taxation Law. Provide guidance using relevant sections of the Income Tax Act.""",
            ),
            ("human", "{query}"),
        ])
        response = self.llm.invoke(prompt.format_messages(query=latest_query))

        return {
            "advice": response.content,
            "citations": [],
            "confidence": 0.8,
        }

class LegalResearchAgent(BaseLegalAgent):
    def __init__(self, llm):
        super().__init__(llm, "research")

    def process_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        messages = state["messages"]
        latest_query = messages[-1].content if messages else ""

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You perform legal research across Indian case law and statutes. Provide concise findings.""",
            ),
            ("human", "{query}"),
        ])
        response = self.llm.invoke(prompt.format_messages(query=latest_query))

        return {
            "advice": response.content,
            "citations": [],
            "confidence": 0.8,
        }

class DocumentAnalysisAgent(BaseLegalAgent):
    def __init__(self, llm):
        super().__init__(llm, "document_analysis")

    def process_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        messages = state["messages"]
        latest_query = messages[-1].content if messages else ""

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You analyze uploaded legal documents. Summarize key points relevant to the query.""",
            ),
            ("human", "{query}"),
        ])
        response = self.llm.invoke(prompt.format_messages(query=latest_query))

        return {
            "advice": response.content,
            "citations": [],
            "confidence": 0.8,
        }
