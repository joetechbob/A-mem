import keyword
from typing import List, Dict, Optional, Any, Tuple
import uuid
from datetime import datetime
from .llm_controller import LLMController
from .retrievers import ChromaRetriever, PersistentChromaRetriever
import json
import logging
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from abc import ABC, abstractmethod
from transformers import AutoModel, AutoTokenizer
from nltk.tokenize import word_tokenize
import pickle
from pathlib import Path
from litellm import completion
import time

logger = logging.getLogger(__name__)

class MemoryNote:
    """A memory note that represents a single unit of information in the memory system.
    
    This class encapsulates all metadata associated with a memory, including:
    - Core content and identifiers
    - Temporal information (creation and access times)
    - Semantic metadata (keywords, context, tags)
    - Relationship data (links to other memories)
    - Usage statistics (retrieval count)
    - Evolution tracking (history of changes)
    """
    
    def __init__(self, 
                 content: str,
                 id: Optional[str] = None,
                 keywords: Optional[List[str]] = None,
                 links: Optional[Dict] = None,
                 retrieval_count: Optional[int] = None,
                 timestamp: Optional[str] = None,
                 last_accessed: Optional[str] = None,
                 context: Optional[str] = None,
                 evolution_history: Optional[List] = None,
                 category: Optional[str] = None,
                 tags: Optional[List[str]] = None,
                 entity_type: Optional[str] = None,
                 entity_id: Optional[str] = None,
                 static_links: Optional[Dict[str, List[str]]] = None,
                 graph_nodes: Optional[List[str]] = None,
                 graph_relationships: Optional[List[str]] = None,
                 memory_role: Optional[str] = None,
                 describes_entity_id: Optional[str] = None,
                 employee_name: Optional[str] = None,
                 employee_id: Optional[str] = None):
        """Initialize a new memory note with its associated metadata.
        
        Args:
            content (str): The main text content of the memory
            id (Optional[str]): Unique identifier for the memory. If None, a UUID will be generated
            keywords (Optional[List[str]]): Key terms extracted from the content
            links (Optional[Dict]): References to related memories (dynamic links from A-mem evolution)
            retrieval_count (Optional[int]): Number of times this memory has been accessed
            timestamp (Optional[str]): Creation time in format YYYYMMDDHHMM
            last_accessed (Optional[str]): Last access time in format YYYYMMDDHHMM
            context (Optional[str]): The broader context or domain of the memory
            evolution_history (Optional[List]): Record of how the memory has evolved
            category (Optional[str]): Classification category
            tags (Optional[List[str]]): Additional classification tags
            entity_type (Optional[str]): Type of entity this note describes (e.g., "Employee", "ExpenseReport")
            entity_id (Optional[str]): Business ID of the entity (e.g., "EMP-00142", "C93FD504...")
            static_links (Optional[Dict[str, List[str]]]): Schema-defined relationships using document IDs
            graph_nodes (Optional[List[str]]): Graph nodes created from this note (for graph DB sync)
            graph_relationships (Optional[List[str]]): Graph relationships created from this note
            memory_role (Optional[str]): Role of this memory ("root_entity" or "attribute_fact")
            describes_entity_id (Optional[str]): For attribute facts, the entity ID they describe
        """
        # Core content and ID
        self.content = content
        self.id = id or str(uuid.uuid4())
        
        # Semantic metadata
        self.keywords = keywords or []
        self.links = links or []  # Dynamic links from A-mem evolution
        self.context = context or "General"
        self.category = category or "Uncategorized"
        self.tags = tags or []
        
        # Entity metadata (NEW)
        self.entity_type = entity_type
        self.entity_id = entity_id
        
        # Memory role metadata (NEW)
        self.memory_role = memory_role  # "root_entity" or "attribute_fact"
        self.describes_entity_id = describes_entity_id  # For attribute facts
        
        # Static structured links (NEW) - format: {'relationship_name': ['doc_id1', 'doc_id2']}
        self.static_links = static_links or {}
        
        # Graph database references (NEW)
        self.graph_nodes = graph_nodes or []
        self.graph_relationships = graph_relationships or []
        
        # Employee context (NEW)
        self.employee_name = employee_name
        self.employee_id = employee_id
        
        # Temporal information
        current_time = datetime.now().strftime("%Y%m%d%H%M")
        self.timestamp = timestamp or current_time
        self.last_accessed = last_accessed or current_time
        
        # Usage and evolution data
        self.retrieval_count = retrieval_count or 0
        self.evolution_history = evolution_history or []

class AgenticMemorySystem:
    """Core memory system that manages memory notes and their evolution.
    
    This system provides:
    - Memory creation, retrieval, update, and deletion
    - Content analysis and metadata extraction
    - Memory evolution and relationship management
    - Hybrid search capabilities
    """
    
    def __init__(self, 
                 model_name: str = 'all-MiniLM-L6-v2',
                 llm_backend: str = "openai",
                 llm_model: str = "gpt-4o-mini",
                 evo_threshold: int = 100,
                 api_key: Optional[str] = None):  
        """Initialize the memory system.
        
        Args:
            model_name: Name of the sentence transformer model
            llm_backend: LLM backend to use (openai/ollama)
            llm_model: Name of the LLM model
            evo_threshold: Number of memories before triggering evolution
            api_key: API key for the LLM service
        """
        self.memories = {}
        self.model_name = model_name
        
        # Initialize persistent ChromaDB retriever to preserve data across sessions
        self.retriever = PersistentChromaRetriever(
            directory="./chromadb_data",
            collection_name="memories",
            model_name=self.model_name,
            extend=True  # Allow loading existing collection
        )
        
        # Load existing memories from ChromaDB if any exist
        try:
            count = self.retriever.collection.count()
            if count > 0:
                # Get all existing documents
                results = self.retriever.collection.get()
                if results and 'ids' in results:
                    for i, doc_id in enumerate(results['ids']):
                        metadata = results['metadatas'][i] if i < len(results['metadatas']) else {}
                        
                        # Deserialize JSON strings back to lists
                        def safe_deserialize(value, default):
                            if isinstance(value, str):
                                try:
                                    return json.loads(value)
                                except:
                                    return default
                            return value if value else default
                        
                        # Reconstruct MemoryNote from metadata
                        note = MemoryNote(
                            content=metadata.get('content', ''),
                            id=doc_id,
                            keywords=safe_deserialize(metadata.get('keywords'), []),
                            links=safe_deserialize(metadata.get('links'), []),
                            retrieval_count=metadata.get('retrieval_count', 0),
                            timestamp=metadata.get('timestamp', ''),
                            last_accessed=metadata.get('last_accessed', ''),
                            context=metadata.get('context', 'General'),
                            evolution_history=safe_deserialize(metadata.get('evolution_history'), []),
                            category=metadata.get('category', 'Uncategorized'),
                            tags=safe_deserialize(metadata.get('tags'), []),
                            entity_type=metadata.get('entity_type'),
                            entity_id=metadata.get('entity_id'),
                            static_links=safe_deserialize(metadata.get('static_links'), {}),
                            graph_nodes=safe_deserialize(metadata.get('graph_nodes'), []),
                            graph_relationships=safe_deserialize(metadata.get('graph_relationships'), []),
                            memory_role=metadata.get('memory_role'),
                            describes_entity_id=metadata.get('describes_entity_id'),
                            employee_name=metadata.get('employee_name'),
                            employee_id=metadata.get('employee_id')
                        )
                        self.memories[doc_id] = note
                logger.info(f"Loaded {len(self.memories)} existing memories from ChromaDB")
        except Exception as e:
            logger.warning(f"Could not load existing memories: {e}")
        
        # Initialize LLM controller
        self.llm_controller = LLMController(llm_backend, llm_model, api_key)
        self.evo_cnt = 0
        self.evo_threshold = evo_threshold

        # Evolution system prompt
        self._evolution_system_prompt = '''
You are an AI memory evolution agent responsible for managing and evolving a knowledge base.
Analyze the new memory note according to keywords and context, also with their several nearest neighbors memory.
Make decisions about its evolution.

The new memory context:
{context}
content: {content}
keywords: {keywords}

The nearest neighbors memories:
{nearest_neighbors_memories}

Based on this information, determine:
1. Should this memory be evolved? Consider its relationships with other memories.
2. What specific actions should be taken (strengthen, update_neighbor)?
   2.1 If choose to strengthen the connection, which memory should it be connected to? Can you give the updated tags of this memory?
   2.2 If choose to update_neighbor, you can update the context and tags of these memories based on the understanding of these memories. If the context and the tags are not updated, the new context and tags should be the same as the original ones. Generate the new context and tags in the sequential order of the input neighbors.

Tags should be determined by the content of these characteristic of these memories, which can be used to retrieve them later and categorize them.

CRITICAL REQUIREMENTS FOR new_context_neighborhood:
- Each context label must be SHORT (5-10 words MAXIMUM)
- DO NOT copy full document text or long passages
- Provide brief categorical labels that describe the memory's domain or topic
- Examples of GOOD context labels:
  * "General"
  * "Project Staffing - Denver Team"
  * "Expense Report - Austin TX"
  * "Client Relationship - TexMed"
  * "Trip Logistics - January 2025"
- Examples of BAD context labels (TOO LONG):
  * "Report ID: C93FD504... [3000 characters of markdown]"
  * "# Expense Update Plan - Line 6... [full document]"

Note that the length of new_tags_neighborhood must equal the number of input neighbors, and the length of new_context_neighborhood must equal the number of input neighbors.
The number of neighbors is {neighbor_number}.

CRITICAL: Return ONLY valid JSON with NO explanatory text before or after.
Do NOT add explanations, comments, or any text outside the JSON object.
Your entire response must be parseable by json.loads().
Use lowercase true/false (not True/False).

Return your decision in JSON format with the following structure:
{{
    "should_evolve": true,
    "actions": ["strengthen", "update_neighbor"],
    "suggested_connections": ["neighbor_memory_ids"],
    "tags_to_update": ["tag_1","tag_n"], 
    "new_context_neighborhood": ["Short Label 1","Short Label 2"],
    "new_tags_neighborhood": [["tag_1","tag_n"],["tag_1","tag_n"]]
}}
'''
        
    def analyze_content(self, content: str) -> Dict:            
        """Analyze content using LLM to extract semantic metadata.
        
        Uses a language model to understand the content and extract:
        - Keywords: Important terms and concepts
        - Context: Overall domain or theme
        - Tags: Classification categories
        
        Args:
            content (str): The text content to analyze
            
        Returns:
            Dict: Contains extracted metadata with keys:
                - keywords: List[str]
                - context: str
                - tags: List[str]
        """
        prompt = """Generate a structured analysis of the following content by:
            1. Identifying the most salient keywords (focus on nouns, verbs, and key concepts)
            2. Extracting core themes and contextual elements
            3. Creating relevant categorical tags

            Format the response as a JSON object:
            {
                "keywords": [
                    // several specific, distinct keywords that capture key concepts and terminology
                    // Order from most to least important
                    // Don't include keywords that are the name of the speaker or time
                    // At least three keywords, but don't be too redundant.
                ],
                "context": 
                    // one sentence summarizing:
                    // - Main topic/domain
                    // - Key arguments/points
                    // - Intended audience/purpose
                ,
                "tags": [
                    // several broad categories/themes for classification
                    // Include domain, format, and type tags
                    // At least three tags, but don't be too redundant.
                ]
            }

            Content for analysis:
            """ + content
        try:
            response = self.llm_controller.llm.get_completion(prompt, response_format={"type": "json_schema", "json_schema": {
                        "name": "response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "keywords": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "context": {
                                    "type": "string",
                                },
                                "tags": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                }
                            }
                        }
                    }})
            return json.loads(response)
        except Exception as e:
            print(f"Error analyzing content: {e}")
            return {"keywords": [], "context": "General", "tags": []}

    def add_note(self, content: str, time: str = None, **kwargs) -> str:
        """Add a new memory note"""
        # Create MemoryNote without llm_controller
        if time is not None:
            kwargs['timestamp'] = time
        note = MemoryNote(content=content, **kwargs)
        
        # Update retriever with all documents
        evo_label, note = self.process_memory(note)
        self.memories[note.id] = note
        
        # Add to ChromaDB with complete metadata (including employee fields)
        metadata = {
            "id": note.id,
            "content": note.content,
            "keywords": note.keywords,
            "links": note.links,
            "retrieval_count": note.retrieval_count,
            "timestamp": note.timestamp,
            "last_accessed": note.last_accessed,
            "context": note.context,
            "evolution_history": note.evolution_history,
            "category": note.category,
            "tags": note.tags,
            "entity_type": note.entity_type,
            "entity_id": note.entity_id,
            "static_links": note.static_links,
            "graph_nodes": note.graph_nodes,
            "graph_relationships": note.graph_relationships,
            "employee_name": note.employee_name,
            "employee_id": note.employee_id
        }
        self.retriever.add_document(note.content, metadata, note.id)
        
        if evo_label == True:
            self.evo_cnt += 1
            if self.evo_cnt % self.evo_threshold == 0:
                self.consolidate_memories()
        return note.id
    
    def consolidate_memories(self):
        """Consolidate memories: update retriever with new documents
        
        Note: This method is called periodically (every evo_threshold memories)
        to rebuild the ChromaDB collection. Since we're using persistent storage,
        we need to be careful not to lose data.
        """
        # Delete the existing collection and recreate it
        try:
            self.retriever.client.delete_collection("memories")
        except:
            pass
            
        # Recreate persistent retriever
        self.retriever = PersistentChromaRetriever(
            directory="./chromadb_data",
            collection_name="memories",
            model_name=self.model_name,
            extend=False  # Create new collection
        )
        
        # Re-add all memory documents with their complete metadata (including employee fields)
        for memory in self.memories.values():
            metadata = {
                "id": memory.id,
                "content": memory.content,
                "keywords": memory.keywords,
                "links": memory.links,
                "retrieval_count": memory.retrieval_count,
                "timestamp": memory.timestamp,
                "last_accessed": memory.last_accessed,
                "context": memory.context,
                "evolution_history": memory.evolution_history,
                "category": memory.category,
                "tags": memory.tags,
                "entity_type": memory.entity_type,
                "entity_id": memory.entity_id,
                "static_links": memory.static_links,
                "graph_nodes": memory.graph_nodes,
                "graph_relationships": memory.graph_relationships,
                "employee_name": memory.employee_name,
                "employee_id": memory.employee_id
            }
            self.retriever.add_document(memory.content, metadata, memory.id)
    
    def find_related_memories(self, query: str, k: int = 5) -> Tuple[str, List[str]]:
        """Find related memories using ChromaDB retrieval"""
        if not self.memories:
            return "", []
            
        try:
            # Get results from ChromaDB
            results = self.retriever.search(query, k)
            
            # Convert to list of memories
            memory_str = ""
            doc_ids = []
            
            if 'ids' in results and results['ids'] and len(results['ids']) > 0 and len(results['ids'][0]) > 0:
                for i, doc_id in enumerate(results['ids'][0]):
                    # Get metadata from ChromaDB results
                    if i < len(results['metadatas'][0]):
                        metadata = results['metadatas'][0][i]
                        # Format memory string with document ID
                        memory_str += f"memory id:{doc_id}\ttalk start time:{metadata.get('timestamp', '')}\tmemory content: {metadata.get('content', '')}\tmemory context: {metadata.get('context', '')}\tmemory keywords: {str(metadata.get('keywords', []))}\tmemory tags: {str(metadata.get('tags', []))}\n"
                        doc_ids.append(doc_id)
                    
            return memory_str, doc_ids
        except Exception as e:
            logger.error(f"Error in find_related_memories: {str(e)}")
            return "", []

    def find_related_memories_raw(self, query: str, k: int = 5) -> str:
        """Find related memories using ChromaDB retrieval in raw format"""
        if not self.memories:
            return ""
            
        # Get results from ChromaDB
        results = self.retriever.search(query, k)
        
        # Convert to list of memories
        memory_str = ""
        
        if 'ids' in results and results['ids'] and len(results['ids']) > 0:
            for i, doc_id in enumerate(results['ids'][0][:k]):
                if i < len(results['metadatas'][0]):
                    # Get metadata from ChromaDB results
                    metadata = results['metadatas'][0][i]
                    
                    # Add main memory info
                    memory_str += f"talk start time:{metadata.get('timestamp', '')}\tmemory content: {metadata.get('content', '')}\tmemory context: {metadata.get('context', '')}\tmemory keywords: {str(metadata.get('keywords', []))}\tmemory tags: {str(metadata.get('tags', []))}\n"
                    
                    # Add linked memories if available
                    links = metadata.get('links', [])
                    j = 0
                    for link_id in links:
                        if link_id in self.memories and j < k:
                            neighbor = self.memories[link_id]
                            memory_str += f"talk start time:{neighbor.timestamp}\tmemory content: {neighbor.content}\tmemory context: {neighbor.context}\tmemory keywords: {str(neighbor.keywords)}\tmemory tags: {str(neighbor.tags)}\n"
                            j += 1
                            
        return memory_str

    def read(self, memory_id: str) -> Optional[MemoryNote]:
        """Retrieve a memory note by its ID.
        
        Args:
            memory_id (str): ID of the memory to retrieve
            
        Returns:
            MemoryNote if found, None otherwise
        """
        return self.memories.get(memory_id)
    
    def update(self, memory_id: str, **kwargs) -> bool:
        """Update a memory note.
        
        Args:
            memory_id: ID of memory to update
            **kwargs: Fields to update
            
        Returns:
            bool: True if update successful
        """
        if memory_id not in self.memories:
            return False
            
        note = self.memories[memory_id]
        
        # Update fields
        for key, value in kwargs.items():
            if hasattr(note, key):
                setattr(note, key, value)
                
        # Update in ChromaDB (including employee fields)
        metadata = {
            "id": note.id,
            "content": note.content,
            "keywords": note.keywords,
            "links": note.links,
            "retrieval_count": note.retrieval_count,
            "timestamp": note.timestamp,
            "last_accessed": note.last_accessed,
            "context": note.context,
            "evolution_history": note.evolution_history,
            "category": note.category,
            "tags": note.tags,
            "entity_type": note.entity_type,
            "entity_id": note.entity_id,
            "static_links": note.static_links,
            "graph_nodes": note.graph_nodes,
            "graph_relationships": note.graph_relationships,
            "employee_name": note.employee_name,
            "employee_id": note.employee_id
        }
        
        # Delete and re-add to update
        self.retriever.delete_document(memory_id)
        self.retriever.add_document(document=note.content, metadata=metadata, doc_id=memory_id)
        
        return True
    
    def delete(self, memory_id: str) -> bool:
        """Delete a memory note by its ID.
        
        Args:
            memory_id (str): ID of the memory to delete
            
        Returns:
            bool: True if memory was deleted, False if not found
        """
        if memory_id in self.memories:
            # Delete from ChromaDB
            self.retriever.delete_document(memory_id)
            # Delete from local storage
            del self.memories[memory_id]
            return True
        return False
    
    def _search_raw(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Internal search method that returns raw results from ChromaDB.
        
        This is used internally by the memory evolution system to find
        related memories for potential evolution.
        
        Args:
            query (str): The search query text
            k (int): Maximum number of results to return
            
        Returns:
            List[Dict[str, Any]]: Raw search results from ChromaDB
        """
        results = self.retriever.search(query, k)
        return [{'id': doc_id, 'score': score} 
                for doc_id, score in zip(results['ids'][0], results['distances'][0])]
                
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for memories using a hybrid retrieval approach."""
        # Get results from ChromaDB (only do this once)
        search_results = self.retriever.search(query, k)
        memories = []
        
        # Process ChromaDB results
        for i, doc_id in enumerate(search_results['ids'][0]):
            memory = self.memories.get(doc_id)
            if memory:
                memories.append({
                    'id': doc_id,
                    'content': memory.content,
                    'context': memory.context,
                    'keywords': memory.keywords,
                    'score': search_results['distances'][0][i]
                })
        
        return memories[:k]
    
    def _search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for memories using a hybrid retrieval approach.
        
        This method combines results from both:
        1. ChromaDB vector store (semantic similarity)
        2. Embedding-based retrieval (dense vectors)
        
        The results are deduplicated and ranked by relevance.
        
        Args:
            query (str): The search query text
            k (int): Maximum number of results to return
            
        Returns:
            List[Dict[str, Any]]: List of search results, each containing:
                - id: Memory ID
                - content: Memory content
                - score: Similarity score
                - metadata: Additional memory metadata
        """
        # Get results from ChromaDB
        chroma_results = self.retriever.search(query, k)
        memories = []
        
        # Process ChromaDB results
        for i, doc_id in enumerate(chroma_results['ids'][0]):
            memory = self.memories.get(doc_id)
            if memory:
                memories.append({
                    'id': doc_id,
                    'content': memory.content,
                    'context': memory.context,
                    'keywords': memory.keywords,
                    'score': chroma_results['distances'][0][i]
                })
                
        # Get results from embedding retriever
        embedding_results = self.retriever.search(query, k)
        
        # Combine results with deduplication
        seen_ids = set(m['id'] for m in memories)
        for result in embedding_results:
            memory_id = result.get('id')
            if memory_id and memory_id not in seen_ids:
                memory = self.memories.get(memory_id)
                if memory:
                    memories.append({
                        'id': memory_id,
                        'content': memory.content,
                        'context': memory.context,
                        'keywords': memory.keywords,
                        'score': result.get('score', 0.0)
                    })
                    seen_ids.add(memory_id)
                    
        return memories[:k]

    def search_agentic(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for memories using ChromaDB retrieval."""
        if not self.memories:
            return []
            
        try:
            # Get results from ChromaDB
            results = self.retriever.search(query, k)
            
            # Process results
            memories = []
            seen_ids = set()
            
            # Check if we have valid results
            if ('ids' not in results or not results['ids'] or 
                len(results['ids']) == 0 or len(results['ids'][0]) == 0):
                return []
                
            # Helper to deserialize JSON strings
            def safe_deserialize(value, default):
                if isinstance(value, str):
                    try:
                        return json.loads(value)
                    except:
                        return default
                return value if value else default
            
            # Process ChromaDB results
            for i, doc_id in enumerate(results['ids'][0][:k]):
                if doc_id in seen_ids:
                    continue
                    
                if i < len(results['metadatas'][0]):
                    metadata = results['metadatas'][0][i]
                    
                    # Create result dictionary with all metadata fields (deserialize lists + employee fields)
                    memory_dict = {
                        'id': doc_id,
                        'content': metadata.get('content', ''),
                        'context': metadata.get('context', ''),
                        'keywords': safe_deserialize(metadata.get('keywords'), []),
                        'tags': safe_deserialize(metadata.get('tags'), []),
                        'links': safe_deserialize(metadata.get('links'), []),
                        'timestamp': metadata.get('timestamp', ''),
                        'category': metadata.get('category', 'Uncategorized'),
                        'employee_name': metadata.get('employee_name'),
                        'employee_id': metadata.get('employee_id'),
                        'is_neighbor': False
                    }
                    
                    # Add score if available
                    if 'distances' in results and len(results['distances']) > 0 and i < len(results['distances'][0]):
                        memory_dict['score'] = results['distances'][0][i]
                        
                    memories.append(memory_dict)
                    seen_ids.add(doc_id)
            
            # Add linked memories as nested children of their primary parent
            # No limit on number of linked neighbors
            # REMOVED DEDUPLICATION: Show all links even if they appear as primary results
            for memory in memories:
                # Get links (already deserialized above)
                links = memory.get('links', [])
                linked_neighbors = []
                
                for link_id in links:
                    neighbor = self.memories.get(link_id)
                    if neighbor:
                        linked_neighbors.append({
                            'id': link_id,
                            'content': neighbor.content,
                            'context': neighbor.context,
                            'keywords': neighbor.keywords,
                            'tags': neighbor.tags,
                            'timestamp': neighbor.timestamp,
                            'category': neighbor.category,
                            'is_neighbor': True
                        })
                
                # Add linked neighbors as a nested property
                memory['linked_neighbors'] = linked_neighbors
            
            return memories  # Return primary results with nested neighbors
        except Exception as e:
            logger.error(f"Error in search_agentic: {str(e)}")
            return []

    def process_memory(self, note: MemoryNote) -> Tuple[bool, MemoryNote]:
        """Process a memory note and determine if it should evolve.
        
        Args:
            note: The memory note to process
            
        Returns:
            Tuple[bool, MemoryNote]: (should_evolve, processed_note)
        """
        # For first memory or testing, just return the note without evolution
        if not self.memories:
            return False, note
            
        try:
            # Get nearest neighbors (k=10 for richer link networks)
            neighbors_text, doc_ids = self.find_related_memories(note.content, k=10)
            if not neighbors_text or not doc_ids:
                return False, note
                
            # Format neighbors for LLM - in this case, neighbors_text is already formatted
            
            # Query LLM for evolution decision
            prompt = self._evolution_system_prompt.format(
                content=note.content,
                context=note.context,
                keywords=note.keywords,
                nearest_neighbors_memories=neighbors_text,
                neighbor_number=len(doc_ids)
            )
            
            try:
                response = self.llm_controller.llm.get_completion(
                    prompt,
                    response_format={"type": "json_schema", "json_schema": {
                        "name": "response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "should_evolve": {
                                    "type": "boolean"
                                },
                                "actions": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "suggested_connections": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "new_context_neighborhood": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "tags_to_update": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "new_tags_neighborhood": {
                                    "type": "array",
                                    "items": {
                                        "type": "array",
                                        "items": {
                                            "type": "string"
                                        }
                                    }
                                }
                            },
                            "required": ["should_evolve", "actions", "suggested_connections", 
                                      "tags_to_update", "new_context_neighborhood", "new_tags_neighborhood"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }}
                )
                
                # Try to extract JSON from response (in case there's extra text)
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    response = json_match.group()
                else:
                    # No valid JSON found - response may be truncated or malformed
                    logger.warning(f"No valid JSON found in LLM response (length: {len(response)})")
                    logger.debug(f"Response content: {response[:500]}...")
                    return False, note
                
                # Fix Python booleans to JSON booleans
                response_fixed = response.replace('True', 'true').replace('False', 'false')
                
                # Validate that we have complete JSON before parsing
                if not response_fixed.strip():
                    logger.warning("Empty response after JSON extraction")
                    return False, note
                    
                response_json = json.loads(response_fixed)
                should_evolve = response_json["should_evolve"]
                
                if should_evolve:
                    actions = response_json["actions"]
                    for action in actions:
                        if action == "strengthen":
                            suggest_connections = response_json["suggested_connections"]
                            new_tags = response_json["tags_to_update"]
                            
                            # Use the LLM's suggested connections directly (they are now document IDs)
                            # Filter to only include IDs that exist in our doc_ids list
                            actual_ids = [conn_id for conn_id in suggest_connections if conn_id in doc_ids]
                            note.links.extend(actual_ids)
                            note.tags = new_tags
                        elif action == "update_neighbor":
                            new_context_neighborhood = response_json["new_context_neighborhood"]
                            new_tags_neighborhood = response_json["new_tags_neighborhood"]
                            
                            # Update each neighbor using their document ID
                            for i in range(min(len(doc_ids), len(new_tags_neighborhood))):
                                if i >= len(doc_ids):
                                    continue
                                    
                                doc_id = doc_ids[i]
                                if doc_id in self.memories:
                                    neighbor_note = self.memories[doc_id]
                                    neighbor_note.tags = new_tags_neighborhood[i]
                                    if i < len(new_context_neighborhood):
                                        neighbor_note.context = new_context_neighborhood[i]
                                
                return should_evolve, note
                
            except (json.JSONDecodeError, KeyError, Exception) as e:
                print(f"[MEMORY DEBUG] Exception caught in process_memory: {type(e).__name__}")
                print(f"[MEMORY DEBUG] Exception message: {str(e)}")
                print(f"[MEMORY DEBUG] Exception details: {repr(e)}")
                import traceback
                print(f"[MEMORY DEBUG] Traceback:\n{traceback.format_exc()}")
                logger.error(f"Error in memory evolution: {str(e)}")
                return False, note
                
        except Exception as e:
            # For testing purposes, catch all exceptions and return the original note
            logger.error(f"Error in process_memory: {str(e)}")
            return False, note
