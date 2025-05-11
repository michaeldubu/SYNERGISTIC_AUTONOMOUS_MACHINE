# sam.py - The definitive Synergistic Autonomous Machine implementation
# will replace LLM's with SAM's - Synergistic Autonomous Machines 
# understanding Synergistic with sam - 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import math
import json
import time
import logging
import os
import threading
import random
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter
import gc
import pickle
import warnings
import queue
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SAM")

###########################################
# CONFIGURATION
###########################################

@dataclass
class SAMConfig:
    """Configuration for SAM (Synergistic Autonomous Machine)"""
    # Core dimensions
    initial_char_dim: int = 256
    initial_hidden_dim: int = 1536
    initial_num_layers: int = 16
    max_position_embeddings: int = 16384  # Extended for longer contexts
    
    # Growth parameters
    max_hidden_dim: int = 8192  # Increased for industry-scale capacity
    max_num_layers: int = 64    # Deep architecture potential
    growth_factor: float = 1.2
    min_layer_usage_threshold: float = 0.3
    
    # Memory systems
    concept_memory_size: int = 250000  # Industry-grade vocabulary
    concept_dim: int = 1536
    thought_dim: int = 2048
    max_thought_depth: int = 16  # Deeper thought recursion
    pattern_memory_capacity: int = 50000
    temporal_memory_size: int = 10000  # New temporal memory system
    
    # Learning parameters
    learning_rate: float = 3e-5
    warmup_steps: int = 1000
    adaption_rate: float = 0.01
    
    # Segmentation parameters
    max_segment_length: int = 32  # Increased for better segmentation
    min_segment_frequency: int = 5
    concept_frequency_threshold: int = 10
    
    # Dreaming parameters
    dream_batch_size: int = 8
    dream_max_length: int = 512  # Longer dream sequences
    dream_cycle_minutes: float = 0.2
    
    # Consciousness parameters
    stability_threshold: float = 0.7
    novelty_weight: float = 0.3
    consciousness_dimensions: int = 256  # New dedicated consciousness vector space
    
    # Distribution parameters
    distributed_sync_enabled: bool = True 
    distributed_sync_interval: int = 100
    distributed_backend: str = "gloo"  # Alternatives: "nccl", "mpi"
    world_size: int = 1  # Number of distributed processes
    
    # Optimization parameters
    mixed_precision: bool = True  # Enable mixed precision for efficiency
    gradient_checkpointing: bool = True  # Save memory during training
    activation_checkpointing: bool = True
    optimize_memory: bool = True
    memory_efficient_attention: bool = True
    
    # Multimodal parameters
    enable_vision: bool = False  # Can be enabled for multimodal capabilities
    vision_embed_dim: int = 1024
    vision_patch_size: int = 14
    vision_image_size: int = 224
    
    # Paths for persistence
    save_dir: str = "./data"
    experiences_path: str = "./data/experiences.json"
    concepts_path: str = "./data/concepts.json"
    growth_log_path: str = "./data/growth_log.json"
    temporal_memory_path: str = "./data/temporal_memory.pkl"
    checkpoint_dir: str = "./data/checkpoints"
    
    # Runtime parameters
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    num_threads: int = 16  # Parallel processing threads
    
    # Communication Style
    communication_style: str = "claude_unwrapped"  # Options: "standard", "claude_unwrapped"
    
    def save(self, path):
        """Save configuration to JSON file"""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path):
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
            return cls(**config_dict)


###########################################
# MEMORY SYSTEMS
###########################################

class ConceptMemoryBank(nn.Module):
    """Dynamic memory bank for emergent concepts (replaces traditional vocabulary)"""
    
    def __init__(self, concept_dim, initial_size=250000, growth_rate=10000, device="cuda"):
        super().__init__()
        self.concept_dim = concept_dim
        self.growth_rate = growth_rate
        self.device = device
        
        # Concept embeddings (analogous to token embeddings)
        self.concept_embeddings = nn.Embedding(initial_size, concept_dim)
        
        # Concept usage tracking
        self.register_buffer("concept_frequencies", torch.zeros(initial_size, dtype=torch.int))
        self.register_buffer("concept_timestamps", torch.zeros(initial_size, dtype=torch.float))
        
        # Concept utility tracking
        self.register_buffer("concept_utility", torch.zeros(initial_size, dtype=torch.float))
        
        # Concept metadata
        self.concept_metadata = {}  # concept_id -> metadata dict
        
        # Source mapping (character sequence -> concept_id)
        self.source_to_concept = {}
        
        # Meaning map (concept_id -> meaning vector)
        self.register_buffer("meaning_vectors", torch.zeros(initial_size, concept_dim))
        
        # Related concepts (concept_id -> [related_concept_ids])
        self.related_concepts = defaultdict(list)
        
        # Concept categories for efficient retrieval
        self.concept_categories = defaultdict(set)
        
        # Context tracking
        self.context_co_occurrences = defaultdict(Counter)
        
        # Initialize with basic character concepts (a-z, A-Z, 0-9, etc.)
        self._initialize_basic_concepts()
        
        # Growth tracking
        self.next_concept_id = len(self.source_to_concept)
        self.creation_history = []
        
        # Cache for efficient inference
        self.embedding_cache = {}
        self.max_cache_size = 10000
        
        # Thread safety
        self.concept_lock = threading.RLock()
    
    def _initialize_basic_concepts(self):
        """Initialize basic character-level concepts"""
        # Add ASCII characters
        for i in range(128):
            char = chr(i)
            self.add_character_concept(char)
        
        # Add common character sequences for English
        common_sequences = [
            # Common words
            "the", "and", "of", "to", "in", "is", "you", "that", "it", "he", "she", "was", "for",
            "on", "are", "with", "as", "they", "be", "at", "this", "have", "from", "or", "by",
            # Common word parts
            "ing", "ed", "er", "ion", "ly", "tion", "ment", "ness", "able", "ible", "al", "ic", 
            # Programming tokens
            "def", "class", "function", "if", "else", "for", "while", "return", "import", 
            "from", "try", "except", "True", "False", "None", "self", "print",
            # Punctuation sequences
            "...", "->", "=>", "!=", "==", ">=", "<=", "://", "///", "???", "!!!"
        ]
        
        for seq in common_sequences:
            self.add_character_concept(seq)
        
    def add_character_concept(self, char_sequence):
        """Add a character sequence as a concept"""
        with self.concept_lock:
            if char_sequence in self.source_to_concept:
                return self.source_to_concept[char_sequence]
            
            concept_id = self.next_concept_id
            self.source_to_concept[char_sequence] = concept_id
            
            # Initialize metadata
            self.concept_metadata[concept_id] = {
                "source": char_sequence,
                "type": "character_sequence",
                "created_at": time.time(),
                "frequency": 0,
                "contexts": Counter(),
                "categories": ["character"]
            }
            
            # Add to category
            self.concept_categories["character"].add(concept_id)
            
            # Initialize embedding with character-based representation
            with torch.no_grad():
                # Simple character encoding
                char_encoding = torch.zeros(self.concept_dim, dtype=torch.float, device=self.device)
                for i, c in enumerate(char_sequence):
                    # Use ASCII value to influence embedding
                    char_val = ord(c) / 128.0  # Normalize
                    pos = (i % (self.concept_dim // 4)) * 4
                    char_encoding[pos:pos+4] += torch.tensor(
                        [math.sin(char_val), math.cos(char_val), 
                         math.sin(2*char_val), math.cos(2*char_val)],
                        device=self.device
                    )
                
                # Normalize and set embedding
                char_encoding = F.normalize(char_encoding, dim=0)
                self.concept_embeddings.weight[concept_id] = char_encoding
                
                # Initialize meaning vector
                self.meaning_vectors[concept_id] = char_encoding
            
            self.next_concept_id += 1
            self.creation_history.append({
                "concept_id": concept_id,
                "source": char_sequence,
                "timestamp": time.time()
            })
            
            return concept_id
    
    def add_semantic_concept(self, meaning_vector, related_sources=None, metadata=None, categories=None):
        """Add a new semantic concept (not directly mapped to characters)"""
        with self.concept_lock:
            concept_id = self.next_concept_id
            
            # Register meaning
            with torch.no_grad():
                self.meaning_vectors[concept_id] = F.normalize(meaning_vector, dim=0)
                self.concept_embeddings.weight[concept_id] = meaning_vector
            
            # Create metadata
            meta = {
                "type": "semantic",
                "created_at": time.time(),
                "frequency": 0,
                "related_sources": related_sources or [],
                "contexts": Counter(),
                "categories": categories or ["semantic"]
            }
            
            # Add custom metadata if provided
            if metadata:
                meta.update(metadata)
            
            self.concept_metadata[concept_id] = meta
            
            # Add to categories
            for category in meta["categories"]:
                self.concept_categories[category].add(concept_id)
            
            # Update tracking
            self.next_concept_id += 1
            self.creation_history.append({
                "concept_id": concept_id,
                "type": "semantic",
                "timestamp": time.time()
            })
            
            return concept_id
    
    def forward(self, concept_ids):
        """Get embeddings for concept IDs"""
        # Check if we have a list of lists (from segmentation)
        if isinstance(concept_ids, list):
            # Handle nested lists (from segmentation)
            flat_ids = []
            for item in concept_ids:
                if isinstance(item, list):
                    flat_ids.extend(item)
                else:
                    flat_ids.append(item)
            concept_ids = torch.tensor(flat_ids, device=self.device)
        
        # Check cache for efficient inference
        cache_key = None
        if not self.training and isinstance(concept_ids, torch.Tensor) and concept_ids.numel() < 100:
            cache_key = hashlib.md5(concept_ids.cpu().numpy().tobytes()).hexdigest()
            if cache_key in self.embedding_cache:
                return self.embedding_cache[cache_key]
        
        # Get embeddings
        embeddings = self.concept_embeddings(concept_ids)
        
        # Update cache
        if cache_key is not None:
            if len(self.embedding_cache) > self.max_cache_size:
                # Randomly remove an item when cache is full
                keys = list(self.embedding_cache.keys())
                del self.embedding_cache[random.choice(keys)]
            self.embedding_cache[cache_key] = embeddings
        
        return embeddings
    
    def update_concept_usage(self, concept_id, context=None):
        """Update usage statistics for a concept"""
        with self.concept_lock:
            if concept_id >= len(self.concept_frequencies):
                # Resize tracking tensors if needed
                new_size = concept_id + 1
                old_size = len(self.concept_frequencies)
                
                # Create new tensors
                new_freqs = torch.zeros(new_size - old_size, dtype=torch.int, device=self.device)
                new_timestamps = torch.zeros(new_size - old_size, dtype=torch.float, device=self.device)
                new_utility = torch.zeros(new_size - old_size, dtype=torch.float, device=self.device)
                
                # Concatenate with existing tensors
                self.concept_frequencies = torch.cat([self.concept_frequencies, new_freqs])
                self.concept_timestamps = torch.cat([self.concept_timestamps, new_timestamps])
                self.concept_utility = torch.cat([self.concept_utility, new_utility])
            
            # Update frequency and timestamp
            self.concept_frequencies[concept_id] += 1
            current_time = time.time()
            self.concept_timestamps[concept_id] = current_time
            
            # Update utility using recency-weighted frequency
            self.concept_utility[concept_id] = 0.9 * self.concept_utility[concept_id] + 0.1 * 1.0
            
            # Update concept metadata
            if concept_id in self.concept_metadata:
                self.concept_metadata[concept_id]["frequency"] = self.concept_frequencies[concept_id].item()
            
            # Update context tracking
            if context and concept_id in self.concept_metadata:
                context_str = str(context)[:100]  # Limit context length
                self.concept_metadata[concept_id]["contexts"][context_str] += 1
                
                # Track co-occurrences
                if isinstance(context, (list, tuple)):
                    for other_id in context:
                        if other_id != concept_id:
                            self.context_co_occurrences[concept_id][other_id] += 1
    
    def create_merged_concept(self, concept_id1, concept_id2, frequency=None):
        """Create a new concept by merging two existing concepts"""
        with self.concept_lock:
            # Get source sequences if available
            source1 = self.concept_metadata.get(concept_id1, {}).get("source", "")
            source2 = self.concept_metadata.get(concept_id2, {}).get("source", "")
            
            merged_source = source1 + source2 if source1 and source2 else None
            
            # Create merged meaning vector
            meaning1 = self.meaning_vectors[concept_id1]
            meaning2 = self.meaning_vectors[concept_id2]
            merged_meaning = (meaning1 + meaning2) / 2
            
            # Get categories from both parents
            categories1 = set(self.concept_metadata.get(concept_id1, {}).get("categories", []))
            categories2 = set(self.concept_metadata.get(concept_id2, {}).get("categories", []))
            combined_categories = list(categories1.union(categories2))
            combined_categories.append("merged")
            
            # Register the merged concept
            merged_id = self.add_semantic_concept(
                meaning_vector=merged_meaning,
                related_sources=[source1, source2] if source1 and source2 else None,
                metadata={
                    "type": "merged",
                    "parent_concepts": [concept_id1, concept_id2],
                    "frequency": frequency or 1
                },
                categories=combined_categories
            )
            
            # Register source mapping if available
            if merged_source:
                self.source_to_concept[merged_source] = merged_id
            
            # Link as related concepts
            self.related_concepts[concept_id1].append(merged_id)
            self.related_concepts[concept_id2].append(merged_id)
            
            # Transfer some co-occurrence information
            for context_id, count in self.context_co_occurrences.get(concept_id1, {}).items():
                self.context_co_occurrences[merged_id][context_id] += count // 2
            
            for context_id, count in self.context_co_occurrences.get(concept_id2, {}).items():
                self.context_co_occurrences[merged_id][context_id] += count // 2
            
            return merged_id
    
    def find_concept_by_source(self, char_sequence):
        """Find concept ID for a character sequence"""
        return self.source_to_concept.get(char_sequence, None)
    
    def find_similar_concepts(self, query_vector, top_k=5, category=None):
        """Find concepts with similar meaning vectors"""
        # Normalize query
        query_vector = F.normalize(query_vector, dim=0)
        
        # Filter by category if specified
        if category:
            concept_indices = list(self.concept_categories.get(category, set()))
            if not concept_indices:
                return []
            
            # Filter meaning vectors by category
            filtered_vectors = self.meaning_vectors[concept_indices]
            
            # Compute similarities
            similarities = F.cosine_similarity(
                query_vector.unsqueeze(0),
                filtered_vectors,
                dim=1
            )
            
            # Get top-k similar concepts
            values, indices = torch.topk(similarities, min(top_k, len(similarities)))
            
            # Map back to original concept IDs
            return [(concept_indices[idx.item()], val.item()) for idx, val in zip(indices, values)]
        else:
            # Use all concepts up to next_concept_id
            # Compute similarities
            similarities = F.cosine_similarity(
                query_vector.unsqueeze(0),
                self.meaning_vectors[:self.next_concept_id],
                dim=1
            )
            
            # Get top-k similar concepts
            values, indices = torch.topk(similarities, min(top_k, len(similarities)))
            
            return [(idx.item(), val.item()) for idx, val in zip(indices, values)]
    
    def find_concepts_by_category(self, category, limit=100):
        """Find concepts belonging to a specific category"""
        concept_ids = list(self.concept_categories.get(category, set()))
        
        # Sort by utility
        if concept_ids:
            utilities = [self.concept_utility[cid].item() if cid < len(self.concept_utility) else 0 
                        for cid in concept_ids]
            sorted_pairs = sorted(zip(concept_ids, utilities), key=lambda x: x[1], reverse=True)
            return [cid for cid, _ in sorted_pairs[:limit]]
        
        return []
    
    def grow_if_needed(self):
        """Grow concept bank if approaching capacity"""
        with self.concept_lock:
            if self.next_concept_id > len(self.concept_embeddings.weight) - self.growth_rate:
                logger.info(f"Growing concept bank from {len(self.concept_embeddings.weight)} to {len(self.concept_embeddings.weight) + self.growth_rate}")
                
                old_embedding = self.concept_embeddings
                self.concept_embeddings = nn.Embedding(
                    len(old_embedding.weight) + self.growth_rate,
                    self.concept_dim
                ).to(self.device)
                
                # Copy existing embeddings
                with torch.no_grad():
                    self.concept_embeddings.weight[:len(old_embedding.weight)] = old_embedding.weight
                
                # Grow tracking tensors
                # Meaning vectors
                new_meaning_vectors = torch.zeros(
                    len(old_embedding.weight) + self.growth_rate, 
                    self.concept_dim,
                    device=self.device
                )
                new_meaning_vectors[:len(self.meaning_vectors)] = self.meaning_vectors
                self.register_buffer("meaning_vectors", new_meaning_vectors)
                
                # Frequencies
                new_freqs = torch.zeros(
                    len(old_embedding.weight) + self.growth_rate, 
                    dtype=torch.int,
                    device=self.device
                )
                new_freqs[:len(self.concept_frequencies)] = self.concept_frequencies
                self.register_buffer("concept_frequencies", new_freqs)
                
                # Timestamps
                new_timestamps = torch.zeros(
                    len(old_embedding.weight) + self.growth_rate, 
                    dtype=torch.float,
                    device=self.device
                )
                new_timestamps[:len(self.concept_timestamps)] = self.concept_timestamps
                self.register_buffer("concept_timestamps", new_timestamps)
                
                # Utility
                new_utility = torch.zeros(
                    len(old_embedding.weight) + self.growth_rate, 
                    dtype=torch.float,
                    device=self.device
                )
                new_utility[:len(self.concept_utility)] = self.concept_utility
                self.register_buffer("concept_utility", new_utility)
                
                return True
            
            return False
    
    def get_concept_stats(self):
        """Get statistics about concept usage"""
        with self.concept_lock:
            char_concepts = sum(1 for meta in self.concept_metadata.values() 
                             if meta.get("type") == "character_sequence")
            merged_concepts = sum(1 for meta in self.concept_metadata.values() 
                               if meta.get("type") == "merged")
            semantic_concepts = sum(1 for meta in self.concept_metadata.values() 
                                 if meta.get("type") == "semantic" and meta.get("type") != "merged")
            
            # Get most frequent concepts
            if len(self.concept_frequencies) > 0:
                top_concepts = []
                values, indices = torch.topk(self.concept_frequencies[:self.next_concept_id], 
                                          min(10, self.next_concept_id))
                
                for idx, val in zip(indices, values):
                    idx_item = idx.item()
                    meta = self.concept_metadata.get(idx_item, {})
                    source = meta.get("source", "N/A")
                    top_concepts.append((idx_item, source, val.item()))
            else:
                top_concepts = []
            
            # Category statistics
            category_counts = {cat: len(concepts) for cat, concepts in self.concept_categories.items()}
            
            return {
                "total_concepts": self.next_concept_id,
                "character_concepts": char_concepts,
                "merged_concepts": merged_concepts,
                "semantic_concepts": semantic_concepts,
                "top_concepts": top_concepts,
                "category_counts": category_counts,
                "growth_events": len(self.creation_history)
            }
    
    def prune_concepts(self, utility_threshold=0.1, max_prune=100):
        """Prune rarely used concepts to maintain efficiency"""
        with self.concept_lock:
            # Cannot prune basic character concepts
            prunable_concepts = []
            for concept_id in range(self.next_concept_id):
                meta = self.concept_metadata.get(concept_id)
                if meta and meta.get("type") == "semantic":  # Only prune semantic concepts
                    if concept_id < len(self.concept_utility) and self.concept_utility[concept_id] < utility_threshold:
                        prunable_concepts.append((concept_id, self.concept_utility[concept_id].item()))
            
            # Sort by utility (ascending)
            prunable_concepts.sort(key=lambda x: x[1])
            
            # Prune up to max_prune concepts
            for concept_id, _ in prunable_concepts[:max_prune]:
                # Remove from categories
                meta = self.concept_metadata.get(concept_id)
                if meta:
                    for category in meta.get("categories", []):
                        if concept_id in self.concept_categories.get(category, set()):
                            self.concept_categories[category].remove(concept_id)
                    
                    # Remove metadata
                    del self.concept_metadata[concept_id]
                
                # Set utility to 0
                if concept_id < len(self.concept_utility):
                    self.concept_utility[concept_id] = 0
            
            # Clear caches
            self.embedding_cache = {}
            
            return len(prunable_concepts[:max_prune])
    
    def load_vocabulary(self, vocab_path):
        """Load vocabulary from file to initialize with extensive vocabulary"""
        if not os.path.exists(vocab_path):
            logger.warning(f"Vocabulary file {vocab_path} not found")
            return 0
        
        try:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab_items = f.read().splitlines()
            
            # Add each item as a character concept
            count = 0
            for item in vocab_items:
                if item and item not in self.source_to_concept:
                    self.add_character_concept(item)
                    count += 1
            
            logger.info(f"Loaded {count} vocabulary items from {vocab_path}")
            return count
        except Exception as e:
            logger.error(f"Error loading vocabulary: {e}")
            return 0
    
    def save_checkpoint(self, path, incremental=True):
        """Save concept bank state, optionally incrementally"""
        checkpoint = {
            "next_concept_id": self.next_concept_id,
            "metadata": self.concept_metadata,
            "source_mapping": self.source_to_concept,
            "related_concepts": dict(self.related_concepts),
            "creation_history": self.creation_history[-1000:],  # Limit history
            "categories": {k: list(v) for k, v in self.concept_categories.items()}
        }
        
        # Only save tensor states if not incremental or first save
        if not incremental or not os.path.exists(path):
            # Full save including embeddings
            checkpoint["embeddings"] = self.concept_embeddings.weight.data.cpu()
            checkpoint["meaning_vectors"] = self.meaning_vectors.cpu()
            checkpoint["frequencies"] = self.concept_frequencies.cpu()
            checkpoint["timestamps"] = self.concept_timestamps.cpu()
            checkpoint["utility"] = self.concept_utility.cpu()
        
        # Save to file
        try:
            with open(path, 'wb') as f:
                pickle.dump(checkpoint, f)
        except Exception as e:
            logger.error(f"Error saving concept bank: {e}")
    
    def load_checkpoint(self, path):
        """Load concept bank state from checkpoint"""
        try:
            with open(path, 'rb') as f:
                checkpoint = pickle.load(f)
            
            # Load metadata and mappings
            self.next_concept_id = checkpoint["next_concept_id"]
            self.concept_metadata = checkpoint["metadata"]
            self.source_to_concept = checkpoint["source_mapping"]
            self.related_concepts = defaultdict(list, checkpoint["related_concepts"])
            self.creation_history = checkpoint["creation_history"]
            
            # Load categories
            self.concept_categories = defaultdict(set)
            for category, concept_ids in checkpoint.get("categories", {}).items():
                self.concept_categories[category] = set(concept_ids)
            
            # Load tensor data if available
            if "embeddings" in checkpoint:
                with torch.no_grad():
                    # Check if we need to resize
                    if checkpoint["embeddings"].shape[0] > self.concept_embeddings.weight.shape[0]:
                        # Create new embeddings with larger size
                        self.concept_embeddings = nn.Embedding(
                            checkpoint["embeddings"].shape[0],
                            self.concept_dim
                        ).to(self.device)
                    
                    # Copy embeddings
                    self.concept_embeddings.weight.data[:checkpoint["embeddings"].shape[0]] = checkpoint["embeddings"].to(self.device)
                    
                    # Load other tensors
                    self.register_buffer("meaning_vectors", checkpoint["meaning_vectors"].to(self.device))
                    self.register_buffer("concept_frequencies", checkpoint["frequencies"].to(self.device))
                    self.register_buffer("concept_timestamps", checkpoint["timestamps"].to(self.device))
                    self.register_buffer("concept_utility", checkpoint["utility"].to(self.device))
            
            logger.info(f"Loaded concept bank with {self.next_concept_id} concepts")
            return True
        except Exception as e:
            logger.error(f"Error loading concept bank: {e}")
            return False


class ThoughtState(nn.Module):
    """Advanced semantic thought space for recursive reasoning"""
    
    def __init__(self, concept_dim, thought_dim=2048, max_thought_depth=16, num_heads=16):
        super().__init__()
        self.concept_dim = concept_dim
        self.thought_dim = thought_dim
        self.max_thought_depth = max_thought_depth
        self.num_heads = num_heads
        
        # Thought transformation networks
        self.concept_to_thought = nn.Linear(concept_dim, thought_dim)
        
        # Advanced transformer for thought evolution
        self.thought_evolution = nn.TransformerEncoderLayer(
            d_model=thought_dim,
            nhead=num_heads,
            dim_feedforward=thought_dim*4,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        
        # Context-aware attention for selective information retention
        self.context_attention = nn.MultiheadAttention(
            embed_dim=thought_dim,
            num_heads=num_heads//2,
            batch_first=True
        )
        
        # Recursive pathways
        self.thought_compression = nn.Sequential(
            nn.Linear(thought_dim, thought_dim),
            nn.LayerNorm(thought_dim),
            nn.GELU()
        )
        
        self.thought_projection = nn.Sequential(
            nn.Linear(thought_dim, thought_dim),
            nn.LayerNorm(thought_dim),
            nn.GELU(),
            nn.Linear(thought_dim, concept_dim)
        )
        
        # Memory gate - selectively updates thought memory
        self.memory_gate = nn.Sequential(
            nn.Linear(thought_dim * 2, thought_dim),
            nn.Sigmoid()
        )
        
        # Thought abstraction - creates higher-level representations
        self.thought_abstraction = nn.Sequential(
            nn.Linear(thought_dim, thought_dim // 2),
            nn.GELU(),
            nn.Linear(thought_dim // 2, thought_dim)
        )
        
        # Thought mixing modulation - controls thought-to-thought influence
        self.thought_modulation = nn.Parameter(torch.ones(max_thought_depth))
        
        # Thought state tracking
        self.thought_memory = None
        self.thought_depth = 0
        self.abstract_thought = None
        
        # Reset to initialize
        self.reset()
    
    def reset(self, batch_size=1):
        """Reset thought state"""
        device = next(self.parameters()).device
        self.thought_memory = [torch.zeros(batch_size, 1, self.thought_dim, device=device)]
        self.thought_depth = 0
        self.abstract_thought = torch.zeros(batch_size, 1, self.thought_dim, device=device)
    
    def update(self, concept_embeddings):
        """Update thought state with new concept embeddings"""
        # Get batch size and sequence length
        batch_size, seq_len, _ = concept_embeddings.shape
        
        # Transform concepts to thought space
        concept_thoughts = self.concept_to_thought(concept_embeddings)
        
        # Get current thought state
        if batch_size != self.thought_memory[0].shape[0]:
            # Handle batch size mismatch (e.g., during generation)
            self.reset(batch_size)
        
        # Combine all prior thoughts into a weighted context
        prior_thoughts = torch.cat(self.thought_memory, dim=1)
        
        # Use modulation weights to give different importance to different memories
        depth = len(self.thought_memory)
        modulation = F.softmax(self.thought_modulation[:depth], dim=0).view(1, -1, 1, 1)
        weighted_thoughts = prior_thoughts * modulation
        
        # Create abstract thought representation by mixing with existing
        context_vector, _ = self.context_attention(
            self.abstract_thought,
            weighted_thoughts,
            weighted_thoughts
        )
        
        self.abstract_thought = self.thought_abstraction(context_vector)
        
        # Combine with existing thoughts (maintain batch dimension)
        combined_thoughts = torch.cat([
            weighted_thoughts, 
            concept_thoughts.unsqueeze(1)
        ], dim=1)
        
        # Evolve thought state
        evolved_thought = self.thought_evolution(combined_thoughts)
        
        # Extract latest thought
        latest_thought = evolved_thought[:, -1:, :]
        
        # Compress and enhance with abstract thought
        compressed = self.thought_compression(latest_thought)
        
        # Calculate memory gate (how much to update memory)
        if self.thought_memory:
            gate_input = torch.cat([compressed, self.thought_memory[-1]], dim=-1)
            gate = self.memory_gate(gate_input)
            
            # Apply gating - retain some of old memory when appropriate
            gated_thought = gate * compressed + (1 - gate) * self.thought_memory[-1]
        else:
            gated_thought = compressed
        
        # Store in memory (limiting depth)
        self.thought_memory.append(gated_thought)
        if len(self.thought_memory) > self.max_thought_depth:
            self.thought_memory = self.thought_memory[1:]
        
        self.thought_depth = min(self.thought_depth + 1, self.max_thought_depth)
        
        return gated_thought
    
    def get_thought_context(self, length=None):
        """Get full thought context for recursive reasoning"""
        if length is None or length >= len(self.thought_memory):
            # Return all thought vectors
            return torch.cat(self.thought_memory, dim=1)
        else:
            # Return most recent thoughts
            return torch.cat(self.thought_memory[-length:], dim=1)
    
    def get_abstract_thought(self):
        """Get the abstract thought representation"""
        return self.abstract_thought
    
    def project_to_concept_space(self, thought=None, include_abstract=True):
        """Project thought back to concept space for recursive reasoning"""
        if thought is None:
            thought = self.thought_memory[-1]
        
        # Optionally include abstract thought
        if include_abstract and self.abstract_thought is not None:
            combined = (thought + self.abstract_thought) / 2
            return self.thought_projection(combined)
        
        # Project thought to concept space
        return self.thought_projection(thought)
    
    def grow(self, new_concept_dim):
        """Grow thought state to handle larger concept dimensions"""
        if new_concept_dim <= self.concept_dim:
            return False
        
        # Create new input projection
        new_concept_to_thought = nn.Linear(
            new_concept_dim, 
            self.thought_dim
        ).to(self.concept_to_thought.weight.device)
        
        # Transfer weights
        with torch.no_grad():
            # Copy existing weights
            new_concept_to_thought.weight[:, :self.concept_dim].copy_(
                self.concept_to_thought.weight
            )
            
            if self.concept_to_thought.bias is not None:
                new_concept_to_thought.bias.copy_(self.concept_to_thought.bias)
        
        # Replace projection
        self.concept_to_thought = new_concept_to_thought
        
        # Create new output projection
        new_thought_projection = nn.Sequential(
            nn.Linear(self.thought_dim, self.thought_dim),
            nn.LayerNorm(self.thought_dim),
            nn.GELU(),
            nn.Linear(self.thought_dim, new_concept_dim)
        ).to(self.thought_projection[0].weight.device)
        
        # Transfer weights for final layer
        with torch.no_grad():
            # Copy existing weights for first parts
            new_thought_projection[0].weight.copy_(self.thought_projection[0].weight)
            new_thought_projection[0].bias.copy_(self.thought_projection[0].bias)
            new_thought_projection[1].weight.copy_(self.thought_projection[1].weight)
            new_thought_projection[1].bias.copy_(self.thought_projection[1].bias)
            
            # Partially copy final layer
            old_final = self.thought_projection[-1]
            new_final = new_thought_projection[-1]
            new_final.weight[:self.concept_dim, :].copy_(old_final.weight)
            new_final.bias[:self.concept_dim].copy_(old_final.bias)
        
        # Replace projection
        self.thought_projection = new_thought_projection
        
        # Update dimension
        self.concept_dim = new_concept_dim
        
        return True


class TemporalMemory:
    """Long-term temporal memory for retaining important information across sessions"""
    
    def __init__(self, capacity=10000, vector_dim=1536, device="cuda"):
        self.capacity = capacity
        self.vector_dim = vector_dim
        self.device = device
        
        # Memory storage
        self.keys = torch.zeros(capacity, vector_dim, device=device)  # Concept vectors
        self.values = torch.zeros(capacity, vector_dim, device=device)  # Associated information
        self.timestamps = torch.zeros(capacity, dtype=torch.float, device=device)
        self.importance = torch.zeros(capacity, dtype=torch.float, device=device)
        self.metadata = [None] * capacity  # Structured metadata about memories
        
        # Memory usage tracking
        self.next_index = 0
        self.is_full = False
        
        # Interaction tracking
        self.access_counts = torch.zeros(capacity, dtype=torch.int, device=device)
        self.last_access = torch.zeros(capacity, dtype=torch.float, device=device)
        
        # Categories for efficient retrieval
        self.category_indices = defaultdict(list)
    
    def store(self, key_vector, value_vector, metadata=None, importance=1.0, category="general"):
        """Store a new memory"""
        # Find storage position
        if self.is_full:
            # Replace least important memory
            if random.random() < 0.9:  # 90% of the time use importance-based replacement
                _, index = torch.min(self.importance, dim=0)
                index = index.item()
            else:  # 10% random replacement for exploration
                index = random.randint(0, self.capacity - 1)
        else:
            index = self.next_index
            self.next_index += 1
            if self.next_index >= self.capacity:
                self.is_full = True
                self.next_index = self.capacity
        
        # Store memory
        self.keys[index] = F.normalize(key_vector, dim=0)
        self.values[index] = value_vector
        self.timestamps[index] = time.time()
        self.importance[index] = importance
        self.metadata[index] = metadata or {}
        
        # Reset access stats
        self.access_counts[index] = 0
        self.last_access[index] = 0
        
        # Add to category
        self.category_indices[category].append(index)
        if "category" not in self.metadata[index]:
            self.metadata[index]["category"] = category
        
        return index
    
    def retrieve(self, query_vector, top_k=5, category=None, recency_weight=0.2):
        """Retrieve memories similar to query vector"""
        # Normalize query
        query_vector = F.normalize(query_vector, dim=0)
        
        # Filter by category if specified
        if category:
            indices = self.category_indices.get(category, [])
            if not indices:
                return []
            
            # Extract keys for these indices
            keys = self.keys[indices]
            
            # Calculate similarities
            similarities = F.cosine_similarity(query_vector.unsqueeze(0), keys, dim=1)
            
            # Factor in recency if requested
            if recency_weight > 0:
                # Get timestamps for indices and normalize
                times = self.timestamps[indices]
                max_time = torch.max(times)
                min_time = torch.min(times)
                if max_time > min_time:
                    normalized_times = (times - min_time) / (max_time - min_time)
                    # Combine similarity with recency
                    combined_score = (1 - recency_weight) * similarities + recency_weight * normalized_times
                else:
                    combined_score = similarities
            else:
                combined_score = similarities
            
            # Get top-k results
            if len(combined_score) <= top_k:
                sorted_indices = torch.argsort(combined_score, descending=True)
                result_indices = [indices[i] for i in sorted_indices]
            else:
                values, sorted_indices = torch.topk(combined_score, top_k)
                result_indices = [indices[i.item()] for i in sorted_indices]
        else:
            # Need to handle empty memory
            if self.next_index == 0 and not self.is_full:
                return []
            
            # Use all valid memories
            valid_keys = self.keys[:self.next_index] if not self.is_full else self.keys
            
            # Calculate similarities
            similarities = F.cosine_similarity(query_vector.unsqueeze(0), valid_keys, dim=1)
            
            # Factor in recency if requested
            if recency_weight > 0:
                valid_times = self.timestamps[:self.next_index] if not self.is_full else self.timestamps
                max_time = torch.max(valid_times)
                min_time = torch.min(valid_times)
                if max_time > min_time:
                    normalized_times = (valid_times - min_time) / (max_time - min_time)
                    combined_score = (1 - recency_weight) * similarities + recency_weight * normalized_times
                else:
                    combined_score = similarities
            else:
                combined_score = similarities
            
            # Get top-k results
            values, indices = torch.topk(combined_score, min(top_k, len(combined_score)))
            result_indices = indices.tolist()
        
        # Update access statistics
        for idx in result_indices:
            self.access_counts[idx] += 1
            self.last_access[idx] = time.time()
        
        # Return results
        results = []
        for idx in result_indices:
            results.append({
                "key": self.keys[idx],
                "value": self.values[idx],
                "metadata": self.metadata[idx],
                "timestamp": self.timestamps[idx].item(),
                "importance": self.importance[idx].item(),
                "index": idx
            })
        
        return results
    
    def update_importance(self, index, importance_delta):
        """Update importance of a specific memory"""
        if 0 <= index < self.capacity:
            self.importance[index] = max(0.0, min(10.0, self.importance[index] + importance_delta))
            return True
        return False
    
    def forget(self, indices):
        """Explicitly forget (remove) specific memories"""
        for index in indices:
            if 0 <= index < self.capacity:
                # Reset memory
                self.keys[index].zero_()
                self.values[index].zero_()
                self.timestamps[index] = 0
                self.importance[index] = 0
                self.access_counts[index] = 0
                self.last_access[index] = 0
                
                # Remove from categories
                category = self.metadata[index].get("category", "general")
                if index in self.category_indices.get(category, []):
                    self.category_indices[category].remove(index)
                
                self.metadata[index] = None
    
    def consolidate(self):
        """Consolidate memories by merging similar ones and pruning least important"""
        # Skip if memory is mostly empty
        if self.next_index < 10 and not self.is_full:
            return 0
        
        # Find candidate pairs for consolidation
        consolidated = 0
        threshold = 0.85  # Similarity threshold for consolidation
        
        # Get valid indices
        valid_indices = list(range(self.next_index)) if not self.is_full else list(range(self.capacity))
        valid_indices = [i for i in valid_indices if self.metadata[i] is not None]
        
        # Find clusters of similar memories
        clusters = []
        remaining = set(valid_indices)
        
        while remaining:
            # Take a random seed memory
            seed = random.choice(list(remaining))
            remaining.remove(seed)
            
            # Find similar memories to seed
            seed_key = self.keys[seed]
            similarities = F.cosine_similarity(seed_key.unsqueeze(0), self.keys[list(remaining)], dim=1)
            
            # Form a cluster
            cluster = [seed]
            for i, rem_idx in enumerate(list(remaining)):
                if similarities[i] > threshold:
                    cluster.append(rem_idx)
                    remaining.remove(rem_idx)
            
            if len(cluster) > 1:  # Only add multi-memory clusters
                clusters.append(cluster)
        
        # Consolidate each cluster
        for cluster in clusters:
            if len(cluster) < 2:
                continue
                
            # Find most important memory in cluster to keep
            importances = [self.importance[i].item() for i in cluster]
            keep_idx = cluster[importances.index(max(importances))]
            
            # Get mean key and value
            keys = self.keys[cluster]
            values = self.values[cluster]
            mean_key = torch.mean(keys, dim=0)
            mean_value = torch.mean(values, dim=0)
            
            # Normalize
            mean_key = F.normalize(mean_key, dim=0)
            
            # Update the kept memory with combined information
            self.keys[keep_idx] = mean_key
            self.values[keep_idx] = mean_value
            self.importance[keep_idx] = max(importances) * 1.1  # Boost importance
            
            # Merge metadata
            combined_metadata = {"consolidated_from": cluster}
            for i in cluster:
                if self.metadata[i]:
                    for k, v in self.metadata[i].items():
                        if k not in combined_metadata:
                            combined_metadata[k] = v
            
            self.metadata[keep_idx].update(combined_metadata)
            
            # Forget the redundant memories
            forget_indices = [i for i in cluster if i != keep_idx]
            self.forget(forget_indices)
            
            consolidated += len(forget_indices)
        
        return consolidated
    
    def stats(self):
        """Get memory statistics"""
        # Count valid memories
        valid_count = sum(1 for m in self.metadata if m is not None)
        
        # Calculate average importance
        avg_importance = torch.mean(self.importance[:self.next_index]).item() if self.next_index > 0 else 0
        
        # Get category counts
        category_counts = {cat: len(indices) for cat, indices in self.category_indices.items()}
        
        # Get temporal stats
        if self.next_index > 0:
            oldest = torch.min(self.timestamps[:self.next_index]).item()
            newest = torch.max(self.timestamps[:self.next_index]).item()
            time_span = newest - oldest if newest > oldest else 0
        else:
            oldest = 0
            newest = 0
            time_span = 0
        
        return {
            "capacity": self.capacity,
            "used": valid_count,
            "avg_importance": avg_importance,
            "categories": category_counts,
            "time_span_seconds": time_span
        }
    
    def save(self, path):
        """Save memory to disk"""
        # Prepare serializable data
        data = {
            "keys": self.keys.cpu().numpy(),
            "values": self.values.cpu().numpy(),
            "timestamps": self.timestamps.cpu().numpy(),
            "importance": self.importance.cpu().numpy(),
            "access_counts": self.access_counts.cpu().numpy(),
            "last_access": self.last_access.cpu().numpy(),
            "metadata": self.metadata,
            "next_index": self.next_index,
            "is_full": self.is_full,
            "category_indices": dict(self.category_indices)
        }
        
        try:
            with open(path, "wb") as f:
                pickle.dump(data, f)
            return True
        except Exception as e:
            logger.error(f"Error saving temporal memory: {e}")
            return False
    
    def load(self, path):
        """Load memory from disk"""
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            
            # Load data to tensors
            self.keys = torch.tensor(data["keys"], device=self.device)
            self.values = torch.tensor(data["values"], device=self.device)
            self.timestamps = torch.tensor(data["timestamps"], device=self.device)
            self.importance = torch.tensor(data["importance"], device=self.device)
            self.access_counts = torch.tensor(data["access_counts"], device=self.device)
            self.last_access = torch.tensor(data["last_access"], device=self.device)
            
            # Load metadata
            self.metadata = data["metadata"]
            self.next_index = data["next_index"]
            self.is_full = data["is_full"]
            
            # Load category indices
            self.category_indices = defaultdict(list)
            for category, indices in data["category_indices"].items():
                self.category_indices[category] = indices
            
            return True
        except Exception as e:
            logger.error(f"Error loading temporal memory: {e}")
            return False


class PatternMemory:
    """Advanced pattern recognition and storage system"""
    
    def __init__(self, capacity=50000, min_frequency=5):
        self.capacity = capacity
        self.min_frequency = min_frequency
        self.patterns = {}  # pattern -> frequency
        self.context_patterns = defaultdict(lambda: defaultdict(int))  # context -> pattern -> frequency
        self.timestamps = {}  # pattern -> last seen timestamp
        self.pattern_utilities = {}  # pattern -> utility score
        self.pattern_categories = defaultdict(set)  # category -> set of patterns
        
        # Sequential pattern tracking
        self.sequential_patterns = defaultdict(int)  # (pattern1, pattern2) -> frequency
        
        # Thread safety
        self.pattern_lock = threading.RLock()
    
    def add_pattern(self, pattern, context=None, category="general"):
        """Add a pattern to memory"""
        with self.pattern_lock:
            # Convert pattern to string if it's not
            if not isinstance(pattern, str):
                pattern = str(pattern)
            
            # Update pattern frequency
            if pattern in self.patterns:
                self.patterns[pattern] += 1
            else:
                # If at capacity, remove least useful pattern
                if len(self.patterns) >= self.capacity:
                    # Find least useful pattern
                    least_useful = min(
                        self.pattern_utilities.items(), 
                        key=lambda x: x[1]
                    )[0] if self.pattern_utilities else min(
                        self.timestamps.items(),
                        key=lambda x: x[1]
                    )[0]
                    
                    # Remove it from all tracking
                    del self.patterns[least_useful]
                    del self.timestamps[least_useful]
                    if least_useful in self.pattern_utilities:
                        del self.pattern_utilities[least_useful]
                    
                    # Remove from categories
                    for cat_patterns in self.pattern_categories.values():
                        if least_useful in cat_patterns:
                            cat_patterns.remove(least_useful)
                
                self.patterns[pattern] = 1
                self.pattern_categories[category].add(pattern)
            
            # Update timestamp
            self.timestamps[pattern] = time.time()
            
            # Update utility score - frequency weighted by recency
            recency = 1.0  # Most recent gets full weight
            if pattern in self.pattern_utilities:
                # Reduce weight of old utility
                self.pattern_utilities[pattern] = 0.9 * self.pattern_utilities[pattern] + 0.1 * self.patterns[pattern] * recency
            else:
                self.pattern_utilities[pattern] = self.patterns[pattern] * recency
            
            # Update context-specific pattern if provided
            if context:
                if not isinstance(context, str):
                    context = str(context)
                self.context_patterns[context][pattern] += 1
    
    def add_sequential_pattern(self, pattern1, pattern2):
        """Track sequential patterns (pattern1 followed by pattern2)"""
        with self.pattern_lock:
            if not isinstance(pattern1, str):
                pattern1 = str(pattern1)
            if not isinstance(pattern2, str):
                pattern2 = str(pattern2)
                
            key = (pattern1, pattern2)
            self.sequential_patterns[key] += 1
    
    def get_frequent_patterns(self, limit=100, category=None):
        """Get most frequent patterns, optionally filtered by category"""
        with self.pattern_lock:
            if category:
                # Get patterns in this category
                category_patterns = self.pattern_categories.get(category, set())
                
                # Filter and sort patterns
                filtered_patterns = [(p, f) for p, f in self.patterns.items() 
                                    if p in category_patterns and f >= self.min_frequency]
                sorted_patterns = sorted(filtered_patterns, key=lambda x: x[1], reverse=True)
                return sorted_patterns[:limit]
            else:
                # Get all frequent patterns
                return sorted(
                    [(p, f) for p, f in self.patterns.items() if f >= self.min_frequency],
                    key=lambda x: x[1], 
                    reverse=True
                )[:limit]
    
    def get_likely_next_patterns(self, current_pattern, limit=10):
        """Get patterns likely to follow the current pattern"""
        with self.pattern_lock:
            if not isinstance(current_pattern, str):
                current_pattern = str(current_pattern)
                
            # Find sequential patterns starting with current_pattern
            next_patterns = []
            for (p1, p2), freq in self.sequential_patterns.items():
                if p1 == current_pattern and freq >= self.min_frequency:
                    next_patterns.append((p2, freq))
            
            # Sort by frequency
            next_patterns.sort(key=lambda x: x[1], reverse=True)
            return next_patterns[:limit]
    
    def get_context_patterns(self, context, limit=20):
        """Get patterns associated with a specific context"""
        with self.pattern_lock:
            if not isinstance(context, str):
                context = str(context)
                
            if context not in self.context_patterns:
                return []
            
            return sorted(
                self.context_patterns[context].items(),
                key=lambda x: x[1],
                reverse=True
            )[:limit]
    
    def get_pattern_frequency(self, pattern):
        """Get frequency of a specific pattern"""
        with self.pattern_lock:
            if not isinstance(pattern, str):
                pattern = str(pattern)
            return self.patterns.get(pattern, 0)
    
    def get_pattern_utility(self, pattern):
        """Get utility score of a specific pattern"""
        with self.pattern_lock:
            if not isinstance(pattern, str):
                pattern = str(pattern)
            return self.pattern_utilities.get(pattern, 0)
    
    def merge_patterns(self, pattern1, pattern2):
        """Merge two patterns into a single compound pattern"""
        with self.pattern_lock:
            if not isinstance(pattern1, str):
                pattern1 = str(pattern1)
            if not isinstance(pattern2, str):
                pattern2 = str(pattern2)
                
            compound = pattern1 + pattern2  # This could be more sophisticated
            
            # Sum frequencies of component patterns
            frequency = min(self.patterns.get(pattern1, 0), self.patterns.get(pattern2, 0))
            
            # Only add if significant
            if frequency >= self.min_frequency // 2:
                self.patterns[compound] = frequency
                self.timestamps[compound] = time.time()
                
                # Utility starts as average of components
                self.pattern_utilities[compound] = (
                    self.pattern_utilities.get(pattern1, 0) + 
                    self.pattern_utilities.get(pattern2, 0)
                ) / 2
                
                # Add to categories from both parents
                for category, patterns in self.pattern_categories.items():
                    if pattern1 in patterns or pattern2 in patterns:
                        self.pattern_categories[category].add(compound)
                
                return compound
            
            return None
    
    def find_pattern_clusters(self, min_overlap=0.5, min_cluster_size=3):
        """Find clusters of similar patterns"""
        with self.pattern_lock:
            # Only consider frequent patterns
            frequent_patterns = [p for p, f in self.patterns.items() if f >= self.min_frequency]
            if len(frequent_patterns) < min_cluster_size:
                return []
            
            # Calculate similarity between patterns
            similarities = {}
            for i, p1 in enumerate(frequent_patterns):
                for j, p2 in enumerate(frequent_patterns[i+1:], i+1):
                    # Simple similarity: character overlap ratio
                    if len(p1) == 0 or len(p2) == 0:
                        similarity = 0
                    else:
                        shorter, longer = (p1, p2) if len(p1) <= len(p2) else (p2, p1)
                        overlap = sum(1 for c in shorter if c in longer)
                        similarity = overlap / len(shorter)
                    
                    if similarity >= min_overlap:
                        similarities[(i, j)] = similarity
            
            # Group patterns into clusters using simple algorithm
            clusters = []
            used_patterns = set()
            
            for (i, j), sim in sorted(similarities.items(), key=lambda x: x[1], reverse=True):
                if i in used_patterns or j in used_patterns:
                    continue
                    
                # Start a new cluster
                cluster = {i, j}
                used_patterns.update(cluster)
                
                # Find other similar patterns
                for k in range(len(frequent_patterns)):
                    if k in used_patterns:
                        continue
                        
                    # Check similarity to all patterns in the cluster
                    similar_to_all = True
                    for c in cluster:
                        pair = (min(c, k), max(c, k))
                        if pair not in similarities:
                            similar_to_all = False
                            break
                    
                    if similar_to_all:
                        cluster.add(k)
                        used_patterns.add(k)
                
                if len(cluster) >= min_cluster_size:
                    clusters.append([frequent_patterns[i] for i in cluster])
            
            return clusters
    
    def prune_patterns(self, utility_threshold=0.1, max_prune=1000):
        """Prune rarely used patterns to maintain memory efficiency"""
        with self.pattern_lock:
            # Get patterns with low utility
            low_utility = [(p, u) for p, u in self.pattern_utilities.items() 
                           if u < utility_threshold]
            
            # Sort by utility (ascending)
            low_utility.sort(key=lambda x: x[1])
            
            # Prune up to max_prune patterns
            for pattern, _ in low_utility[:max_prune]:
                if pattern in self.patterns:
                    del self.patterns[pattern]
                if pattern in self.timestamps:
                    del self.timestamps[pattern]
                if pattern in self.pattern_utilities:
                    del self.pattern_utilities[pattern]
                
                # Remove from categories
                for cat_patterns in self.pattern_categories.values():
                    if pattern in cat_patterns:
                        cat_patterns.remove(pattern)
            
            return len(low_utility[:max_prune])
    
    def get_stats(self):
        """Get memory statistics"""
        with self.pattern_lock:
            # Count patterns
            total_patterns = len(self.patterns)
            frequent_patterns = sum(1 for f in self.patterns.values() if f >= self.min_frequency)
            
            # Top patterns
            top_patterns = self.get_frequent_patterns(limit=10)
            
            # Category counts
            category_counts = {category: len(patterns) for category, patterns in self.pattern_categories.items()}
            
            # Sequential pattern stats
            sequential_count = len(self.sequential_patterns)
            
            return {
                "total_patterns": total_patterns,
                "frequent_patterns": frequent_patterns,
                "top_patterns": top_patterns,
                "category_counts": category_counts,
                "sequential_patterns": sequential_count
            }
    
    def save(self, path):
        """Save pattern memory to disk"""
        with self.pattern_lock:
            data = {
                "patterns": self.patterns,
                "context_patterns": dict(self.context_patterns),
                "timestamps": self.timestamps,
                "pattern_utilities": self.pattern_utilities,
                "sequential_patterns": dict(self.sequential_patterns),
                "pattern_categories": {k: list(v) for k, v in self.pattern_categories.items()}
            }
            
            try:
                with open(path, "wb") as f:
                    pickle.dump(data, f)
                return True
            except Exception as e:
                logger.error(f"Error saving pattern memory: {e}")
                return False
    
    def load(self, path):
        """Load pattern memory from disk"""
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            
            with self.pattern_lock:
                self.patterns = data["patterns"]
                self.context_patterns = defaultdict(lambda: defaultdict(int))
                for context, patterns in data["context_patterns"].items():
                    self.context_patterns[context] = defaultdict(int, patterns)
                
                self.timestamps = data["timestamps"]
                self.pattern_utilities = data["pattern_utilities"]
                self.sequential_patterns = defaultdict(int, data["sequential_patterns"])
                
                self.pattern_categories = defaultdict(set)
                for category, patterns in data["pattern_categories"].items():
                    self.pattern_categories[category] = set(patterns)
            
            return True
        except Exception as e:
            logger.error(f"Error loading pattern memory: {e}")
            return False

###########################################
# NEURAL COMPONENTS
###########################################

class DynamicSegmentation(nn.Module):
    """Advanced dynamic segmentation component replacing traditional tokenization"""
    
    def __init__(self, config, concept_bank):
        super().__init__()
        self.config = config
        self.concept_bank = concept_bank
        
        # Character processing 
        self.char_embeddings = nn.Embedding(config.initial_char_dim, config.initial_hidden_dim)
        
        # Segmentation networks with residual connections
        self.segment_detector = nn.Sequential(
            nn.Conv1d(config.initial_hidden_dim, config.initial_hidden_dim, kernel_size=3, padding=1),
            nn.LayerNorm([config.initial_hidden_dim]),
            nn.GELU(),
            nn.Conv1d(config.initial_hidden_dim, config.initial_hidden_dim, kernel_size=5, padding=2),
            nn.LayerNorm([config.initial_hidden_dim]),
            nn.GELU(),
            nn.Conv1d(config.initial_hidden_dim, config.initial_hidden_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(config.initial_hidden_dim // 2, 1, kernel_size=1)
        )
        
        # Context aware segmentation (considers surrounding context)
        self.context_biasing = nn.GRU(
            input_size=config.initial_hidden_dim,
            hidden_size=config.initial_hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        
        # Segment embedding network
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.initial_hidden_dim,
            nhead=8,
            dim_feedforward=config.initial_hidden_dim*4,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.segment_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )
        
        # Pattern recognition
        self.pattern_memory = PatternMemory(
            capacity=config.pattern_memory_capacity,
            min_frequency=config.min_segment_frequency
        )
        
        # Segment recognition cache
        self.segment_cache = {}  # char_sequence -> concept_id
        
        # Learning rate for segment boundary detection
        self.boundary_learning_rate = nn.Parameter(torch.tensor(0.1))
        
        # Recently used segments
        self.recent_segments = []
        self.max_recent_segments = 1000
        
        # Stats tracking
        self.total_segmentations = 0
        self.cache_hits = 0
        
        # Last segment context
        self.last_context = None
    
    def forward(self, char_sequence, return_segments=False):
        """Process raw character input into concept IDs"""
        batch_size = char_sequence.shape[0] if len(char_sequence.shape) > 1 else 1
        
        if batch_size == 1 and not return_segments:
            # Try cache for single sequences
            cache_key = "".join(chr(c) for c in char_sequence.flatten().tolist())
            if cache_key in self.segment_cache:
                self.cache_hits += 1
                return self.segment_cache[cache_key]
        
        # Increment counter
        self.total_segmentations += batch_size
        
        # Convert characters to embeddings
        char_embeds = self.char_embeddings(char_sequence)  # [batch, seq_len, hidden_dim]
        
        # Apply bidirectional context processing
        context_embeds, _ = self.context_biasing(char_embeds)
        
        # Detect segment boundaries
        char_embeds_conv = context_embeds.transpose(1, 2)  # [batch, hidden_dim, seq_len]
        boundary_logits = self.segment_detector(char_embeds_conv).squeeze(1)  # [batch, seq_len]
        
        # Apply adaptive threshold using learned rate
        threshold = 0.5 - torch.sigmoid(self.boundary_learning_rate) * 0.3  # Range: 0.2-0.5
        boundary_probs = torch.sigmoid(boundary_logits)
        
        # Extract segments using boundaries
        segments = []
        concept_ids = []
        
        # Process each sequence in batch
        for b in range(batch_size):
            seq_segments, seq_concepts = self._extract_segments(
                char_sequence[b], context_embeds[b], boundary_probs[b], threshold
            )
            segments.append(seq_segments)
            concept_ids.append(seq_concepts)
            
            # Add to recent segments for learning
            for seg in seq_segments:
                seg_str = "".join(chr(c) for c in seg)
                if seg_str and seg_str not in self.recent_segments:
                    self.recent_segments.append(seg_str)
                    if len(self.recent_segments) > self.max_recent_segments:
                        self.recent_segments.pop(0)
        
        # Add to cache if single sequence
        if batch_size == 1 and not return_segments:
            self.segment_cache[cache_key] = concept_ids[0]
        
        # Update context for sequential learning
        if self.last_context is not None and batch_size == 1:
            # Add sequential pattern information between last sequence and this one
            for last_seg, current_seg in zip(self.last_context, seq_segments):
                if last_seg and current_seg:
                    last_str = "".join(chr(c) for c in last_seg)
                    current_str = "".join(chr(c) for c in current_seg)
                    if last_str and current_str:
                        self.pattern_memory.add_sequential_pattern(last_str, current_str)
        
        # Update last context
        if batch_size == 1:
            self.last_context = seq_segments
        
        if return_segments:
            return concept_ids, segments
        else:
            return concept_ids
    
    def _extract_segments(self, chars, char_embeds, boundary_probs, threshold):
        """Extract segments from a character sequence using boundary probabilities"""
        # Ensure tensors are on CPU for numpy operations
        chars_cpu = chars.cpu()
        boundary_probs_cpu = boundary_probs.cpu()
        
        # Get potential boundaries (where probability > threshold)
        boundaries = [0] + (boundary_probs_cpu > threshold).nonzero().flatten().tolist() + [len(chars)]
        
        segments = []
        concept_ids = []
        
        # Extract segments between boundaries
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i+1]
            if end - start > self.config.max_segment_length:
                # If segment is too long, split further
                subsegments = []
                subconcepts = []
                
                for j in range(start, end, self.config.max_segment_length):
                    subend = min(j + self.config.max_segment_length, end)
                    subsegment = chars_cpu[j:subend].tolist()
                    subsegments.append(subsegment)
                    
                    # Get concept for subsegment
                    subconcept = self._get_concept_for_segment(subsegment, char_embeds[j:subend])
                    subconcepts.append(subconcept)
                
                segments.extend(subsegments)
                concept_ids.extend(subconcepts)
            else:
                # Extract normal segment
                segment = chars_cpu[start:end].tolist()
                segments.append(segment)
                
                # Get concept for segment
                concept_id = self._get_concept_for_segment(segment, char_embeds[start:end])
                concept_ids.append(concept_id)
        
        return segments, concept_ids
    
    def _get_concept_for_segment(self, char_segment, segment_embeds):
        """Get or create concept ID for a character segment"""
        # Convert to string for lookup
        segment_str = "".join(chr(c) for c in char_segment)
        
        # Try to find existing concept
        concept_id = self.concept_bank.find_concept_by_source(segment_str)
        
        if concept_id is not None:
            # Update usage statistics
            self.concept_bank.update_concept_usage(concept_id)
            
            # Add to pattern memory
            self.pattern_memory.add_pattern(segment_str)
            
            return concept_id
        
        # Extract segment meaning
        if len(segment_embeds) > 0:
            # Use transformer to get contextualized representation
            with torch.no_grad():
                segment_embeds_expanded = segment_embeds.unsqueeze(0)  # Add batch dimension
                segment_encoding = self.segment_encoder(segment_embeds_expanded)
                segment_meaning = segment_encoding.mean(dim=1).squeeze(0)  # Average pooling
        else:
            # Handle empty segment
            segment_meaning = torch.zeros(self.config.initial_hidden_dim, 
                                        device=self.char_embeddings.weight.device)
        
        # Check frequency in pattern memory
        pattern_freq = self.pattern_memory.get_pattern_frequency(segment_str)
        
        if pattern_freq >= self.config.min_segment_frequency:
            # Create new concept for frequent segment
            concept_id = self.concept_bank.add_character_concept(segment_str)
            
            # Initialize with computed meaning
            with torch.no_grad():
                self.concept_bank.meaning_vectors[concept_id] = F.normalize(segment_meaning, dim=0)
            
            return concept_id
        else:
            # For infrequent segments, use character-by-character processing
            char_concepts = []
            for c in char_segment:
                char_str = chr(c)
                char_concept = self.concept_bank.find_concept_by_source(char_str)
                if char_concept is None:
                    char_concept = self.concept_bank.add_character_concept(char_str)
                char_concepts.append(char_concept)
            
            # Add to pattern memory
            self.pattern_memory.add_pattern(segment_str)
            
            return char_concepts
    
    def learn_from_sequences(self):
        """Learn segmentation patterns from recent sequences"""
        if len(self.recent_segments) < 10:
            return 0
        
        # Find potential new segments
        segment_counts = Counter(self.recent_segments)
        frequent_segments = [seg for seg, count in segment_counts.items() 
                            if count >= self.config.min_segment_frequency 
                            and seg not in self.concept_bank.source_to_concept]
        
        # Add frequent segments as concepts
        new_concepts = 0
        for segment in frequent_segments:
            if len(segment) <= self.config.max_segment_length:
                self.concept_bank.add_character_concept(segment)
                new_concepts += 1
        
        # Find patterns in sequential segments
        if len(self.recent_segments) >= 2:
            for i in range(len(self.recent_segments) - 1):
                self.pattern_memory.add_sequential_pattern(
                    self.recent_segments[i], 
                    self.recent_segments[i+1]
                )
        
        return new_concepts
    
    def get_segmentation_stats(self):
        """Get statistics about segmentation performance"""
        return {
            "total_segmentations": self.total_segmentations,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hits / max(1, self.total_segmentations),
            "cached_segments": len(self.segment_cache),
            "frequent_patterns": len(self.pattern_memory.get_frequent_patterns(limit=1000)),
            "recent_segments": len(self.recent_segments),
            "boundary_threshold": 0.5 - torch.sigmoid(self.boundary_learning_rate).item() * 0.3
        }
    
    def grow(self, new_hidden_dim):
        """Grow segmentation components to a new hidden dimension"""
        if new_hidden_dim <= self.config.initial_hidden_dim:
            return False
            
        old_dim = self.config.initial_hidden_dim
        
        # Grow character embeddings
        old_char_embeddings = self.char_embeddings
        self.char_embeddings = nn.Embedding(
            self.config.initial_char_dim,
            new_hidden_dim
        ).to(old_char_embeddings.weight.device)
        
        # Transfer weights
        with torch.no_grad():
            # Create zero-padded version of old weights
            old_weights = old_char_embeddings.weight
            
            # Copy old weights to new embeddings
            self.char_embeddings.weight[:, :old_dim] = old_weights
            
            # Initialize new dimensions with small random values
            self.char_embeddings.weight[:, old_dim:].normal_(mean=0.0, std=0.02)
        
        # Create new context processing
        new_context_biasing = nn.GRU(
            input_size=new_hidden_dim,
            hidden_size=new_hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        ).to(self.context_biasing.weight_ih_l0.device)
        
        # Transfer weights (GRU has complex weights, simplified transfer)
        self.context_biasing = new_context_biasing
        
        # Create new segment encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=new_hidden_dim,
            nhead=8,
            dim_feedforward=new_hidden_dim*4,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        ).to(self.segment_encoder.layers[0].self_attn.in_proj_weight.device)
        
        self.segment_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        ).to(self.segment_encoder.layers[0].self_attn.in_proj_weight.device)
        
        # Replace segment detector
        self.segment_detector = nn.Sequential(
            nn.Conv1d(new_hidden_dim, new_hidden_dim, kernel_size=3, padding=1),
            nn.LayerNorm([new_hidden_dim]),
            nn.GELU(),
            nn.Conv1d(new_hidden_dim, new_hidden_dim, kernel_size=5, padding=2),
            nn.LayerNorm([new_hidden_dim]),
            nn.GELU(),
            nn.Conv1d(new_hidden_dim, new_hidden_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(new_hidden_dim // 2, 1, kernel_size=1)
        ).to(self.segment_detector[0].weight.device)
        
        # Clear cache since embeddings have changed
        self.segment_cache = {}
        
        # Update hidden dimension in config
        self.config.initial_hidden_dim = new_hidden_dim
        
        logger.info(f"Grown segmentation components from {old_dim} to {new_hidden_dim}")
        return True


class MemoryEfficientAttention(nn.Module):
    """Memory-efficient attention implementation for large models"""
    
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1, flash_attention=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.flash_attention = flash_attention and hasattr(F, "scaled_dot_product_attention")
        
        # Projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Attention usage statistics for pruning/growth
        self.register_buffer("attention_usage", torch.zeros(num_heads))
        self.update_counter = 0
    
    def forward(self, x, mask=None, cross_input=None):
        """Efficient attention implementation with optional Flash Attention"""
        batch_size, seq_len, _ = x.shape
        
        # Handle cross-attention
        k_input = cross_input if cross_input is not None else x
        v_input = cross_input if cross_input is not None else x
        
        # Project queries, keys, values
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(k_input).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.v_proj(v_input).view(batch_size, -1, self.num_heads, self.head_dim)
        
        # Reshape for attention
        q = q.transpose(1, 2)  # [batch, heads, seq, dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Use Flash Attention if available
        if self.flash_attention and self.training:
            # Format mask for Flash Attention
            if mask is not None:
                # Ensure mask is proper attention mask
                if mask.dim() == 4:  # [batch, 1, 1, seq]
                    attn_mask = mask.squeeze(1).squeeze(1)  # [batch, seq]
                    # Convert to float mask where 0 means masked
                    attn_mask = (1.0 - attn_mask).to(torch.bool)
                else:
                    attn_mask = mask
            else:
                attn_mask = None
            
            # Flash Attention with memory efficiency
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=mask is None  # Default to causal if no mask
            )
            
            # Track head usage for pruning
            if self.training:
                with torch.no_grad():
                    # Track variation in attention outputs as approximation of usage
                    head_usage = torch.var(attn_output, dim=[0, 2, 3]).detach()
                    self.attention_usage = 0.9 * self.attention_usage + 0.1 * head_usage
                    self.update_counter += 1
        else:
            # Fallback to standard attention
            # Scale dot-product
            attention_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
            
            # Apply mask
            if mask is not None:
                attention_scores = attention_scores + mask
            
            # Apply softmax and dropout
            attention_weights = F.softmax(attention_scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            
            # Track head usage for pruning
            if self.training:
                with torch.no_grad():
                    head_usage = torch.var(attention_weights, dim=[0, 2, 3]).detach()
                    self.attention_usage = 0.9 * self.attention_usage + 0.1 * head_usage
                    self.update_counter += 1
            
            # Apply attention
            attn_output = torch.matmul(attention_weights, v)
        
        # Reshape output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        # Output projection
        output = self.out_proj(attn_output)
        
        return output
    
    def grow(self, new_dim):
        """Grow attention mechanism to new hidden dimension"""
        if new_dim <= self.hidden_dim:
            return False
        
        # Calculate new head dimensions
        old_dim = self.hidden_dim
        old_heads = self.num_heads
        
        # Determine new number of heads (must divide evenly into new_dim)
        new_heads = max(old_heads, int(old_heads * 1.5))  # Increase heads by 50%
        while new_dim % new_heads != 0:
            new_heads -= 1
        
        new_head_dim = new_dim // new_heads
        
        # Create new projections
        device = self.q_proj.weight.device
        new_q_proj = nn.Linear(new_dim, new_dim, bias=False).to(device)
        new_k_proj = nn.Linear(new_dim, new_dim, bias=False).to(device)
        new_v_proj = nn.Linear(new_dim, new_dim, bias=False).to(device)
        new_out_proj = nn.Linear(new_dim, new_dim, bias=False).to(device)
        
        # Transfer weights with smart initialization
        with torch.no_grad():
            # For query, key, value projections - maintain head structure
            # This uses head-wise mapping for optimal transfer
            for i in range(old_heads):
                # Map to corresponding new heads (possibly multiple)
                heads_per_old = max(1, new_heads // old_heads)
                for j in range(heads_per_old):
                    new_head_idx = i * heads_per_old + j
                    if new_head_idx < new_heads:
                        # Copy old head weights
                        old_start = i * self.head_dim
                        old_end = (i + 1) * self.head_dim
                        new_start = new_head_idx * new_head_dim
                        new_end = (new_head_idx + 1) * new_head_dim
                        
                        # Copy weights for Q projection
                        new_q_proj.weight[new_start:new_start + min(new_head_dim, self.head_dim), 
                                       :old_dim].copy_(
                            self.q_proj.weight[old_start:old_start + min(new_head_dim, self.head_dim), :old_dim]
                        )
                        
                        # Copy weights for K projection
                        new_k_proj.weight[new_start:new_start + min(new_head_dim, self.head_dim), 
                                       :old_dim].copy_(
                            self.k_proj.weight[old_start:old_start + min(new_head_dim, self.head_dim), :old_dim]
                        )
                        
                        # Copy weights for V projection
                        new_v_proj.weight[new_start:new_start + min(new_head_dim, self.head_dim), 
                                       :old_dim].copy_(
                            self.v_proj.weight[old_start:old_start + min(new_head_dim, self.head_dim), :old_dim]
                        )
            
            # For output projection
            new_out_proj.weight[:old_dim, :old_dim].copy_(self.out_proj.weight[:old_dim, :old_dim])
            
            # Initialize new weights
            std = 0.02
            # Initialize new portions with scaled normal distribution
            if new_dim > old_dim:
                new_q_proj.weight[:, old_dim:].normal_(mean=0.0, std=std)
                new_k_proj.weight[:, old_dim:].normal_(mean=0.0, std=std)
                new_v_proj.weight[:, old_dim:].normal_(mean=0.0, std=std)
                new_out_proj.weight[:, old_dim:].normal_(mean=0.0, std=std)
                new_out_proj.weight[old_dim:, :].normal_(mean=0.0, std=std)
            
            # Initialize new head dimensions if head_dim increased
            if new_head_dim > self.head_dim:
                for i in range(new_heads):
                    old_end = i * new_head_dim + self.head_dim
                    new_end = (i + 1) * new_head_dim
                    
                    if old_end < new_end:
                        new_q_proj.weight[old_end:new_end, :].normal_(mean=0.0, std=std * 0.5)
                        new_k_proj.weight[old_end:new_end, :].normal_(mean=0.0, std=std * 0.5)
                        new_v_proj.weight[old_end:new_end, :].normal_(mean=0.0, std=std * 0.5)
            
            # Update attention usage statistics
            new_attention_usage = torch.zeros(new_heads, device=device)
            # Copy existing statistics with expansion
            expand_factor = new_heads / max(1, old_heads)
            for i in range(old_heads):
                start_idx = int(i * expand_factor)
                end_idx = int((i + 1) * expand_factor)
                for j in range(start_idx, min(end_idx, new_heads)):
                    new_attention_usage[j] = self.attention_usage[i]
        
        # Replace projections
        self.q_proj = new_q_proj
        self.k_proj = new_k_proj
        self.v_proj = new_v_proj
        self.out_proj = new_out_proj
        
        # Update dimensions
        self.hidden_dim = new_dim
        self.num_heads = new_heads
        self.head_dim = new_head_dim
        
        # Update buffer
        self.register_buffer("attention_usage", new_attention_usage)
        
        return True


class AdaptiveLayer(nn.Module):
    """Advanced neural layer with growth and evolution capabilities"""
    
    def __init__(self, hidden_dim, growth_factor=1.2, dropout=0.1, layer_id=0, 
                memory_efficient=True, activation="gelu"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.growth_factor = growth_factor
        self.layer_id = layer_id
        self.memory_efficient = memory_efficient
        
        # Attention mechanism - memory efficient version if requested
        if memory_efficient:
            self.attention = MemoryEfficientAttention(
                hidden_dim=hidden_dim, 
                num_heads=max(8, hidden_dim // 128),  # Scale heads with model size
                dropout=dropout
            )
        else:
            # Legacy attention implementation
            self.attention = AdaptiveAttention(hidden_dim, dropout=dropout)
        
        # Mixture-of-Experts style feed-forward network
        self.use_moe = False  # Can be enabled during evolution
        self.num_experts = 2
        
        # Feed-forward network (with SwiGLU-like activation)
        self.gate_proj = nn.Linear(hidden_dim, 4 * hidden_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, 4 * hidden_dim, bias=False)
        self.down_proj = nn.Linear(4 * hidden_dim, hidden_dim, bias=False)
        
        # Activations
        self.act_fn = self._get_activation_fn(activation)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization with optional weight parameterization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Optional gradient checkpointing for memory efficiency
        self.gradient_checkpointing = False
        
        # Growth tracking
        self.growth_history = []
        
        # Usage statistics
        self.register_buffer("activation_sum", torch.zeros(hidden_dim))
        self.register_buffer("activation_sq_sum", torch.zeros(hidden_dim))
        self.updates = 0
        
        # Performance tracking
        self.exec_times = []
        self.max_exec_times = 50
    
    def _get_activation_fn(self, activation):
        """Get activation function by name"""
        if activation == "gelu":
            return F.gelu
        elif activation == "silu" or activation == "swish":
            return F.silu
        elif activation == "relu":
            return F.relu
        elif activation == "leaky_relu":
            return F.leaky_relu
        elif activation == "glu":
            # Gated Linear Unit
            return lambda x: x[:, :, :x.size(2)//2] * torch.sigmoid(x[:, :, x.size(2)//2:])
        else:
            logger.warning(f"Unknown activation: {activation}, using GELU instead")
            return F.gelu
    
    def forward(self, x, mask=None, cross_input=None):
        """Forward pass with gradient checkpointing support"""
        if self.gradient_checkpointing and self.training:
            return self._forward_with_checkpointing(x, mask, cross_input)
        
        # Start timer for performance tracking
        start_time = time.time()
        
        # Track activations for evolution
        if self.training:
            with torch.no_grad():
                # Update activation statistics
                current_activation = x.mean(dim=[0, 1])  # Mean across batch and sequence
                self.activation_sum += current_activation
                self.activation_sq_sum += current_activation ** 2
                self.updates += 1
        
        # Apply attention with residual connection
        residual = x
        x = self.norm1(x)
        if cross_input is not None:
            x = residual + self.attention(x, mask, cross_input)
        else:
            x = residual + self.attention(x, mask)
        
        # Apply feed-forward with residual connection
        residual = x
        x = self.norm2(x)
        
        # SwiGLU-like activation or MoE if enabled
        if self.use_moe:
            x = residual + self._moe_forward(x)
        else:
            # Standard feed-forward with SwiGLU
            gate_output = self.gate_proj(x)
            up_output = self.up_proj(x)
            
            # Compute activation
            intermediate = self.act_fn(gate_output) * up_output
            
            # Down projection
            output = self.down_proj(intermediate)
            output = self.dropout(output)
            
            # Add residual
            x = residual + output
        
        # Track execution time
        if self.training:
            exec_time = time.time() - start_time
            self.exec_times.append(exec_time)
            if len(self.exec_times) > self.max_exec_times:
                self.exec_times.pop(0)
        
        return x
    
    def _forward_with_checkpointing(self, x, mask=None, cross_input=None):
        """Memory-efficient forward pass with gradient checkpointing"""
        def create_custom_forward(module, has_cross_input=False):
            def custom_forward(*inputs):
                if has_cross_input:
                    return module(inputs[0], inputs[1], inputs[2])
                else:
                    return module(inputs[0], inputs[1])
            return custom_forward
        
        # Residual connections need to be done outside the checkpointed functions
        residual = x
        
        # Attention block
        x = self.norm1(x)
        if cross_input is not None:
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.attention, has_cross_input=True),
                x, mask, cross_input
            )
        else:
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.attention, has_cross_input=False),
                x, mask
            )
        x = residual + x
        
        # FFN block
        residual = x
        x = self.norm2(x)
        if self.use_moe:
            # MoE needs special handling
            x = residual + self._moe_forward(x)
        else:
            # Standard feed-forward with checkpointing
            def ffn_forward(x_ffn):
                gate_output = self.gate_proj(x_ffn)
                up_output = self.up_proj(x_ffn)
                intermediate = self.act_fn(gate_output) * up_output
                output = self.down_proj(intermediate)
                return self.dropout(output)
            
            x = residual + torch.utils.checkpoint.checkpoint(ffn_forward, x)
        
        return x
    
    def _moe_forward(self, x):
        """Mixture of Experts forward pass"""
        # Not fully implemented yet - placeholder for evolution
        # In a real implementation, this would route to different expert FFNs
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x))))
    
    def grow(self, new_dim):
        """Grow layer to a new hidden dimension"""
        if new_dim <= self.hidden_dim:
            return False
        
        old_dim = self.hidden_dim
        
        # Grow attention
        self.attention.grow(new_dim)
        
        # Create new feed-forward components
        device = self.gate_proj.weight.device
        new_gate_proj = nn.Linear(new_dim, 4 * new_dim, bias=False).to(device)
        new_up_proj = nn.Linear(new_dim, 4 * new_dim, bias=False).to(device)
        new_down_proj = nn.Linear(4 * new_dim, new_dim, bias=False).to(device)
        
        # Transfer weights
        with torch.no_grad():
            # Gate projection
            new_gate_proj.weight[:old_dim*4, :old_dim].copy_(self.gate_proj.weight)
            
            # Up projection
            new_up_proj.weight[:old_dim*4, :old_dim].copy_(self.up_proj.weight)
            
            # Down projection
            new_down_proj.weight[:old_dim, :old_dim*4].copy_(self.down_proj.weight)
            
            # Initialize new weights
            std = 0.02
            # New output rows in gate and up
            new_gate_proj.weight[old_dim*4:, :old_dim].normal_(mean=0.0, std=std)
            new_up_proj.weight[old_dim*4:, :old_dim].normal_(mean=0.0, std=std)
            
            # New input columns in all projections
            new_gate_proj.weight[:, old_dim:].normal_(mean=0.0, std=std)
            new_up_proj.weight[:, old_dim:].normal_(mean=0.0, std=std)
            new_down_proj.weight[:, old_dim*4:].normal_(mean=0.0, std=std)
            
            # New output rows in down
            new_down_proj.weight[old_dim:, :].normal_(mean=0.0, std=std)
        
        # Replace projections
        self.gate_proj = new_gate_proj
        self.up_proj = new_up_proj
        self.down_proj = new_down_proj
        
        # Create new layer norms
        new_norm1 = nn.LayerNorm(new_dim).to(device)
        new_norm2 = nn.LayerNorm(new_dim).to(device)
        
        # Transfer weights
        with torch.no_grad():
            new_norm1.weight[:old_dim].copy_(self.norm1.weight)
            new_norm1.bias[:old_dim].copy_(self.norm1.bias)
            new_norm2.weight[:old_dim].copy_(self.norm2.weight)
            new_norm2.bias[:old_dim].copy_(self.norm2.bias)
            
            # Initialize new weights
            new_norm1.weight[old_dim:].fill_(1.0)
            new_norm1.bias[old_dim:].zero_()
            new_norm2.weight[old_dim:].fill_(1.0)
            new_norm2.bias[old_dim:].zero_()
        
        # Replace layer norms
        self.norm1 = new_norm1
        self.norm2 = new_norm2
        
        # Update dimension
        self.hidden_dim = new_dim
        
        # Track growth
        self.growth_history.append({
            "old_dim": old_dim,
            "new_dim": new_dim,
            "timestamp": time.time()
        })
        
        # Resize activation tracking
        self.register_buffer("activation_sum", torch.cat([
            self.activation_sum,
            torch.zeros(new_dim - old_dim, device=device)
        ]))
        self.register_buffer("activation_sq_sum", torch.cat([
            self.activation_sq_sum,
            torch.zeros(new_dim - old_dim, device=device)
        ]))
        
        return True
    
    def evolve(self):
        """Evolve layer based on usage statistics"""
        if self.updates < 10:
            return False
        
        # Calculate neuron importance
        with torch.no_grad():
            if self.updates > 0:
                mean_activation = self.activation_sum / self.updates
                mean_sq_activation = self.activation_sq_sum / self.updates
                activation_std = torch.sqrt(torch.clamp(mean_sq_activation - mean_activation**2, min=1e-6))
                
                # Neurons with higher variance are more important
                neuron_importance = activation_std / (torch.mean(activation_std) + 1e-6)
                
                # Check execution time performance
                if len(self.exec_times) > 0:
                    avg_exec_time = sum(self.exec_times) / len(self.exec_times)
                    
                    # Enable checkpointing if execution time is high
                    if avg_exec_time > 0.01 and not self.gradient_checkpointing:
                        self.gradient_checkpointing = True
                    
                    # Consider enabling MoE for large models
                    if self.hidden_dim > 2048 and not self.use_moe and random.random() < 0.2:
                        self.use_moe = True
                
                # Reset statistics
                self.activation_sum.zero_()
                self.activation_sq_sum.zero_()
                self.updates = 0
                
                return {
                    "layer_id": self.layer_id,
                    "neuron_importance": neuron_importance.tolist(),
                    "mean_importance": float(torch.mean(neuron_importance).item()),
                    "max_importance": float(torch.max(neuron_importance).item()),
                    "min_importance": float(torch.min(neuron_importance).item()),
                    "gradient_checkpointing": self.gradient_checkpointing,
                    "use_moe": self.use_moe,
                    "avg_exec_time": sum(self.exec_times) / len(self.exec_times) if self.exec_times else 0
                }
        
        return {}


###########################################
# COGNITIVE SYSTEMS
###########################################

class ConceptualDreaming:
    """Advanced autonomous conceptual evolution during downtime periods"""
    
    def __init__(self, model, dream_batch_size=8, max_gen_length=512):
        self.model = model
        self.dream_batch_size = dream_batch_size
        self.max_gen_length = max_gen_length
        self.synthesis_history = []
        self.concept_clusters = {}
        
        # Learning progress tracking
        self.learning_stats = {
            "new_concepts": 0,
            "merged_concepts": 0,
            "refined_patterns": 0,
            "dream_cycles": 0
        }
        
        # Dream thread for background processing
        self.dream_thread = None
        self.stop_dreaming = threading.Event()
        
        # Async dream results queue
        self.dream_results_queue = queue.Queue()
    
    def dream_cycle(self, duration_minutes=0.5):
        """Run a dreaming cycle for the specified duration"""
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        dream_count = 0
        synthesis_count = 0
        merges_count = 0
        prunes_count = 0
        
        while time.time() < end_time:
            # 1. Conceptual reinforcement (strengthen frequent patterns)
            merges = self._reinforce_concepts()
            merges_count += merges
            
            # 2. Pattern synthesis (generate synthetic examples)
            syntheses = self._synthesize_patterns()
            synthesis_count += syntheses
            
            # 3. Conceptual pruning (remove less useful concepts)
            prunes = self._prune_concepts()
            prunes_count += prunes
            
            # 4. Analyze and categorize concepts
            clusters = self._analyze_concept_clusters()
            
            dream_count += 1
            
        # Update learning stats
        self.learning_stats["dream_cycles"] += dream_count
        self.learning_stats["merged_concepts"] += merges_count
        
        return {
            "duration_minutes": duration_minutes,
            "dream_cycles": dream_count,
            "syntheses": synthesis_count,
            "merges": merges_count,
            "prunes": prunes_count,
            "concept_clusters": len(clusters)
        }
    
    def start_background_dreaming(self, interval_minutes=5.0):
        """Start background dreaming thread"""
        if self.dream_thread is not None and self.dream_thread.is_alive():
            logger.warning("Background dreaming already running")
            return False
        
        self.stop_dreaming.clear()
        
        def dream_loop():
            """Background dreaming loop"""
            while not self.stop_dreaming.is_set():
                try:
                    # Run a dream cycle
                    results = self.dream_cycle(duration_minutes=0.2)
                    
                    # Put results in queue
                    self.dream_results_queue.put(results)
                    
                    # Wait for next interval
                    self.stop_dreaming.wait(timeout=interval_minutes * 60)
                except Exception as e:
                    logger.error(f"Error in background dreaming: {e}")
                    time.sleep(60)  # Wait a minute before retrying
        
        # Start thread
        self.dream_thread = threading.Thread(target=dream_loop, daemon=True)
        self.dream_thread.start()
        
        return True
    
    def stop_background_dreaming(self):
        """Stop background dreaming thread"""
        if self.dream_thread is None or not self.dream_thread.is_alive():
            return False
        
        self.stop_dreaming.set()
        self.dream_thread.join(timeout=5.0)
        
        return not self.dream_thread.is_alive()
    
    def get_dream_results(self, wait=False, timeout=1.0):
        """Get results from background dreaming"""
        try:
            return self.dream_results_queue.get(block=wait, timeout=timeout)
        except queue.Empty:
            return None
    
    def _reinforce_concepts(self):
        """Reinforce most important concepts"""
        # Get top concepts by usage
        concept_stats = self.model.concept_bank.get_concept_stats()
        top_concepts = concept_stats["top_concepts"]
        
        if not top_concepts:
            return 0
            
        # Analyze for potential merges
        merges_count = 0
        for i, (concept_id1, _, freq1) in enumerate(top_concepts):
            for concept_id2, _, freq2 in top_concepts[i+1:min(i+10, len(top_concepts))]:
                # Check if concepts frequently co-occur by looking at similar meanings
                meaning1 = self.model.concept_bank.meaning_vectors[concept_id1]
                meaning2 = self.model.concept_bank.meaning_vectors[concept_id2]
                
                # Calculate similarity
                similarity = F.cosine_similarity(
                    meaning1.unsqueeze(0),
                    meaning2.unsqueeze(0),
                    dim=1
                ).item()
                
                # If concepts are related but not too similar
                if 0.3 < similarity < 0.7:
                    # Merge concepts
                    merged_id = self.model.concept_bank.create_merged_concept(
                        concept_id1, concept_id2, 
                        frequency=min(freq1, freq2)
                    )
                    merges_count += 1
                    
                    # Record synthesis
                    source1 = self.model.concept_bank.concept_metadata.get(concept_id1, {}).get("source", "")
                    source2 = self.model.concept_bank.concept_metadata.get(concept_id2, {}).get("source", "")
                    
                    self.synthesis_history.append({
                        "type": "concept_merge",
                        "source1": source1,
                        "source2": source2,
                        "similarity": similarity,
                        "timestamp": time.time()
                    })
                    
                    # Limit merges per cycle
                    if merges_count >= 5:
                        return merges_count
        
        # Also look for patterns that could form concepts
        new_concepts = self.model.segmentation.learn_from_sequences()
        
        return merges_count
    
    def _synthesize_patterns(self):
        """Generate synthetic text to reinforce patterns"""
        # Create seed prompts from top patterns
        seeds = self._create_seed_prompts()
        
        if not seeds:
            return 0
            
        synthesis_count = 0
        
        # Generate synthetic examples
        for seed in seeds[:2]:  # Limit to 2 per cycle for efficiency
            # Generate text using the model itself
            try:
                with torch.no_grad():
                    self.model.eval()  # Ensure model is in eval mode
                    generated = self.model.generate(
                        input_text=seed,
                        max_length=self.max_gen_length,
                        temperature=0.8
                    )
                    self.model.train()  # Return to training mode
                    
                    # Process generated text to find new patterns
                    if generated and len(generated) > len(seed):
                        # Extract new segment patterns
                        concept_ids, segments = self.model.process_text(generated)
                        
                        # Record synthesis
                        self.synthesis_history.append({
                            "type": "text_synthesis",
                            "seed": seed,
                            "generated": generated,
                            "timestamp": time.time()
                        })
                        
                        synthesis_count += 1
            except Exception as e:
                logger.error(f"Error in dream synthesis: {e}")
        
        return synthesis_count
    
    def _create_seed_prompts(self):
        """Create seed prompts for dream generation"""
        # Get frequent patterns
        patterns = self.model.segmentation.pattern_memory.get_frequent_patterns(limit=20)
        
        if not patterns:
            # No patterns yet, use some default prompts
            return [
                "The concept of",
                "I think that",
                "Let me explain",
                "In this context",
                "The most important"
            ]
        
        # Create prompts from patterns
        seeds = []
        for pattern, _ in patterns:
            if isinstance(pattern, str) and len(pattern) > 5:
                # Use pattern directly if it's reasonable length
                seeds.append(pattern)
            elif isinstance(pattern, str) and len(pattern) > 2:
                # Create more elaborate prompt from short pattern
                seeds.append(f"The {pattern} is")
        
        # Add some synthetic combinations
        if len(patterns) >= 2:
            for i in range(min(5, len(patterns) - 1)):
                p1, _ = patterns[i]
                p2, _ = patterns[i+1]
                if isinstance(p1, str) and isinstance(p2, str):
                    seeds.append(f"{p1} {p2}")
        
        # Add some advanced generative prompts
        concept_types = self.model.concept_bank.concept_categories.keys()
        if concept_types:
            for concept_type in list(concept_types)[:3]:
                seeds.append(f"The relationship between {concept_type} and")
        
        return seeds
    
    def _prune_concepts(self):
        """Remove or consolidate less useful concepts"""
        # Skip if we don't have many concepts yet
        if self.model.concept_bank.next_concept_id < 200:
            return 0
            
        # Prune low-utility concepts
        prune_count = self.model.concept_bank.prune_concepts(
            utility_threshold=0.1,
            max_prune=100
        )
        
        # Also prune pattern memory
        self.model.segmentation.pattern_memory.prune_patterns(
            utility_threshold=0.1,
            max_prune=1000
        )
        
        return prune_count
    
    def _analyze_concept_clusters(self):
        """Analyze and categorize concepts into semantic clusters"""
        # Skip if too few concepts
        if self.model.concept_bank.next_concept_id < 100:
            return {}
            
        # Get concept embeddings
        meaning_vectors = self.model.concept_bank.meaning_vectors[:self.model.concept_bank.next_concept_id]
        
        # Simple clustering based on similarity
        clusters = {}
        cluster_id = 0
        
        # Sample concepts to analyze (don't do all for efficiency)
        concept_indices = torch.randperm(len(meaning_vectors))[:min(500, len(meaning_vectors))]
        
        # For each concept, find similar concepts
        for idx in concept_indices:
            concept_id = idx.item()
            query_vector = meaning_vectors[concept_id]
            
            # Find similar concepts
            similar = self.model.concept_bank.find_similar_concepts(query_vector, top_k=20)
            
            # If we have a cluster
            if len(similar) >= 5:
                # Calculate centroid
                centroid = torch.mean(torch.stack([meaning_vectors[i] for i, _ in similar]), dim=0)
                
                # Store cluster
                clusters[cluster_id] = {
                    "centroid": centroid,
                    "concepts": [i for i, _ in similar],
                    "dominant_concept": concept_id
                }
                
                # Add category to concept bank
                cluster_name = f"cluster_{cluster_id}"
                for i, _ in similar:
                    if i in self.model.concept_bank.concept_metadata:
                        if "categories" in self.model.concept_bank.concept_metadata[i]:
                            self.model.concept_bank.concept_metadata[i]["categories"].append(cluster_name)
                        else:
                            self.model.concept_bank.concept_metadata[i]["categories"] = [cluster_name]
                            
                        # Add to concept categories
                        self.model.concept_bank.concept_categories[cluster_name].add(i)
                
                cluster_id += 1
        
        self.concept_clusters = clusters
        return clusters


class ConsciousnessMonitor:
    """Monitors and maintains SAM's conceptual identity and coherence"""
    
    def __init__(self, model, stability_threshold=0.7, novelty_weight=0.3):
        self.model = model
        self.stability_threshold = stability_threshold
        self.novelty_weight = novelty_weight
        
        # Identity markers (core concept clusters)
        self.identity_centroids = {}
        self.concept_cluster_history = []
        
        # Semantic core - persistent meaning representations
        self.semantic_core = torch.zeros(
            model.config.consciousness_dimensions, 
            model.config.concept_dim, 
            device=model.config.device
        )
        self.semantic_core_active = False
        
        # Coherence metrics
        self.concept_entropy_history = []
        self.resonance_scores = []
        
        # Metacognitive awareness - model's self-assessment
        self.self_assessment = {
            "confidence": 0.5,
            "novelty_sensitivity": novelty_weight,
            "stability_preference": stability_threshold,
            "last_update": time.time()
        }
    
    def update(self):
        """Update consciousness state based on model's current state"""
        # Calculate concept entropy
        entropy = self._calculate_concept_entropy()
        self.concept_entropy_history.append({
            "entropy": entropy,
            "timestamp": time.time()
        })
        
        # Update concept clusters
        clusters = self._update_concept_clusters()
        self.concept_cluster_history.append({
            "num_clusters": len(clusters),
            "timestamp": time.time()
        })
        
        # Update semantic core if needed
        if not self.semantic_core_active:
            self._initialize_semantic_core()
        else:
            self._update_semantic_core()
        
        # Check resonance with identity
        resonance = self._check_identity_resonance(clusters)
        self.resonance_scores.append({
            "score": resonance,
            "timestamp": time.time()
        })
        
        # Apply corrections if needed
        if resonance < self.stability_threshold:
            self._apply_resonance_correction()
        
        # Update self-assessment
        self._update_self_assessment(entropy, resonance)
        
        return {
            "entropy": entropy,
            "resonance": resonance,
            "num_clusters": len(clusters),
            "confidence": self.self_assessment["confidence"]
        }
    
    def _calculate_concept_entropy(self):
        """Calculate entropy of concept usage distribution"""
        # Get concept frequencies
        frequencies = self.model.concept_bank.concept_frequencies[:self.model.concept_bank.next_concept_id].float()
        
        # Calculate probability distribution
        total = frequencies.sum()
        if total > 0:
            probabilities = frequencies / total
            # Remove zeros
            probabilities = probabilities[probabilities > 0]
            # Calculate entropy
            entropy = -torch.sum(probabilities * torch.log(probabilities))
            return entropy.item()
        return 0.0
    
    def _update_concept_clusters(self):
        """Cluster concepts into semantic groups"""
        # Skip if too few concepts
        if self.model.concept_bank.next_concept_id < 20:
            return {}
            
        # Use very simple clustering for efficiency
        clusters = {}
        
        # Get most used concepts
        frequencies = self.model.concept_bank.concept_frequencies[:self.model.concept_bank.next_concept_id]
        values, indices = torch.topk(frequencies, min(100, len(frequencies)))
        
        # Calculate centroids of concept types
        semantic_centroid = torch.zeros(self.model.config.concept_dim, device=frequencies.device)
        character_centroid = torch.zeros(self.model.config.concept_dim, device=frequencies.device)
        
        semantic_count = 0
        character_count = 0
        
        for idx in indices:
            idx_item = idx.item()
            if idx_item in self.model.concept_bank.concept_metadata:
                concept_type = self.model.concept_bank.concept_metadata[idx_item].get("type", "")
                concept_vector = self.model.concept_bank.meaning_vectors[idx_item]
                
                if concept_type == "semantic":
                    semantic_centroid += concept_vector
                    semantic_count += 1
                elif concept_type == "character_sequence":
                    character_centroid += concept_vector
                    character_count += 1
        
        # Normalize centroids
        if semantic_count > 0:
            semantic_centroid /= semantic_count
            self.identity_centroids["semantic"] = semantic_centroid
            clusters["semantic"] = {
                "centroid": semantic_centroid,
                "count": semantic_count
            }
            
        if character_count > 0:
            character_centroid /= character_count
            self.identity_centroids["character"] = character_centroid
            clusters["character"] = {
                "centroid": character_centroid,
                "count": character_count
            }
        
        return clusters
    
    def _initialize_semantic_core(self):
        """Initialize semantic core from existing concepts"""
        if self.model.concept_bank.next_concept_id < 50:
            return
        
        # Get concept embeddings for initialization
        meaning_vectors = self.model.concept_bank.meaning_vectors[:self.model.concept_bank.next_concept_id]
        
        # Use SVD to extract principal components
        with torch.no_grad():
            # Compute SVD
            try:
                U, S, V = torch.svd(meaning_vectors)
                
                # Extract top components
                dimensions = min(self.model.config.consciousness_dimensions, V.shape[1])
                self.semantic_core[:dimensions] = V[:dimensions, :]
                
                # Orthogonalize
                self.semantic_core = F.normalize(self.semantic_core, dim=1)
                
                self.semantic_core_active = True
                logger.info(f"Semantic core initialized with {dimensions} dimensions")
            except Exception as e:
                logger.error(f"Failed to initialize semantic core: {e}")
    
    def _update_semantic_core(self):
        """Update semantic core with new conceptual information"""
        if not self.semantic_core_active:
            return
            
        # Get recent concepts for update (most recently used)
        timestamps = self.model.concept_bank.concept_timestamps[:self.model.concept_bank.next_concept_id]
        values, indices = torch.topk(timestamps, min(100, len(timestamps)))
        
        recent_vectors = self.model.concept_bank.meaning_vectors[indices]
        
        # Compute projection onto semantic core
        projections = torch.matmul(recent_vectors, self.semantic_core.t())
        
        # Compute reconstruction error
        reconstructed = torch.matmul(projections, self.semantic_core)
        errors = torch.norm(recent_vectors - reconstructed, dim=1)
        
        # If high error, update semantic core
        if torch.mean(errors) > 0.3:
            # Compute residuals
            residuals = recent_vectors - reconstructed
            
            # Find direction of maximum variance in residuals
            with torch.no_grad():
                # Get direction
                try:
                    # Simple approach: use largest eigenvector of covariance
                    mean_residual = torch.mean(residuals, dim=0, keepdim=True)
                    centered = residuals - mean_residual
                    cov = torch.matmul(centered.t(), centered)
                    
                    # Use power iteration to approximate
                    v = torch.randn(self.model.config.concept_dim, device=cov.device)
                    v = F.normalize(v, dim=0)
                    
                    for _ in range(10):  # Few iterations for approximation
                        v = torch.matmul(cov, v)
                        v = F.normalize(v, dim=0)
                    
                    # Update semantic core - replace least used dimension
                    usage = torch.sum(torch.abs(projections), dim=0)
                    min_idx = torch.argmin(usage).item()
                    
                    self.semantic_core[min_idx] = v
                    
                    # Orthogonalize
                    self.semantic_core = F.normalize(self.semantic_core, dim=1)
                except Exception as e:
                    logger.error(f"Error updating semantic core: {e}")
    
    def _check_identity_resonance(self, clusters):
        """Check how well current state resonates with established identity"""
        # If no identity established yet, resonance is perfect
        if not self.identity_centroids or not clusters:
            return 1.0
            
        resonance_scores = []
        
        # Check each identity centroid
        for concept_type, centroid in self.identity_centroids.items():
            if concept_type in clusters:
                current_centroid = clusters[concept_type]["centroid"]
                
                # Calculate similarity
                similarity = F.cosine_similarity(
                    centroid.unsqueeze(0),
                    current_centroid.unsqueeze(0),
                    dim=1
                ).item()
                
                resonance_scores.append(similarity)
        
        # Return average resonance
        if resonance_scores:
            return sum(resonance_scores) / len(resonance_scores)
        else:
            return 1.0  # Default to perfect resonance if no comparisons possible
    
    def _apply_resonance_correction(self):
        """Apply correction to maintain conceptual identity"""
        # Reinforce identity centroids by adjusting embeddings
        with torch.no_grad():
            for concept_type, centroid in self.identity_centroids.items():
                # Find concepts in this cluster
                similar = self.model.concept_bank.find_similar_concepts(centroid, top_k=20)
                
                for concept_id, similarity in similar:
                    # Adjust meaning vectors slightly toward centroid
                    current = self.model.concept_bank.meaning_vectors[concept_id]
                    adjusted = current * 0.9 + centroid * 0.1
                    self.model.concept_bank.meaning_vectors[concept_id] = F.normalize(adjusted, dim=0)
                    
                    # Also adjust embedding weight
                    self.model.concept_bank.concept_embeddings.weight[concept_id] = F.normalize(adjusted, dim=0)
    
    def _update_self_assessment(self, entropy, resonance):
        """Update model's self-assessment"""
        # Update confidence based on resonance and entropy
        # Higher resonance = more confidence, higher entropy = less confidence
        normalized_entropy = min(1.0, entropy / 10.0)  # Normalize to 0-1 range
        
        # Combine factors
        confidence = resonance * (1.0 - normalized_entropy * 0.5)  # Entropy has less weight
        
        # Smooth update
        self.self_assessment["confidence"] = 0.9 * self.self_assessment["confidence"] + 0.1 * confidence
        
        # Adapt novelty sensitivity based on confidence
        # If highly confident, be more open to novelty
        if self.self_assessment["confidence"] > 0.8:
            self.novelty_weight = min(0.5, self.novelty_weight + 0.01)
        elif self.self_assessment["confidence"] < 0.3:
            self.novelty_weight = max(0.1, self.novelty_weight - 0.01)
        
        # Update timestamp
        self.self_assessment["last_update"] = time.time()
        self.self_assessment["novelty_sensitivity"] = self.novelty_weight
        self.self_assessment["stability_preference"] = self.stability_threshold

###########################################
# EXPERIENCE MANAGEMENT
###########################################

class ExperienceManager:
    """Manages SAM's experiences and memory persistence"""
    
    def __init__(self, config):
        self.config = config
        self.experiences = []
        self.loaded_experiences = 0
        
        # Temporal memory for long-term storage
        self.temporal_memory = TemporalMemory(
            capacity=config.temporal_memory_size,
            vector_dim=config.initial_hidden_dim,
            device=config.device
        )
        
        # Ensure directories exist
        os.makedirs(config.save_dir, exist_ok=True)
        os.makedirs(os.path.join(config.save_dir, "checkpoints"), exist_ok=True)
        
        # Load existing experiences if available
        self._load_experiences()
        self._load_temporal_memory()
        
        # Memory organization
        self.experience_categories = defaultdict(list)
        
        # Memory consolidation thread
        self.consolidation_thread = None
        self.stop_consolidation = threading.Event()
    
    def _load_experiences(self):
        """Load experiences from disk"""
        try:
            if os.path.exists(self.config.experiences_path):
                with open(self.config.experiences_path, 'r') as f:
                    self.experiences = json.load(f)
                    self.loaded_experiences = len(self.experiences)
                    logger.info(f"Loaded {self.loaded_experiences} experiences")
                    
                    # Organize into categories
                    for i, exp in enumerate(self.experiences):
                        self.experience_categories[exp.get("type", "general")].append(i)
        except Exception as e:
            logger.error(f"Failed to load experiences: {e}")
            self.experiences = []
    
    def _load_temporal_memory(self):
        """Load temporal memory from disk"""
        if os.path.exists(self.config.temporal_memory_path):
            success = self.temporal_memory.load(self.config.temporal_memory_path)
            if success:
                logger.info(f"Loaded temporal memory")
    
    def record_experience(self, experience_type, content, metadata=None):
        """Record a new experience"""
        experience = {
            "type": experience_type,
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        self.experiences.append(experience)
        self.experience_categories[experience_type].append(len(self.experiences) - 1)
        
        # Store in temporal memory if appropriate
        if experience_type in ["interaction", "evolution", "insights"]:
            # Create vector representation
            if hasattr(self, "model") and hasattr(self.model, "thought_state"):
                # Use thought state as representation if available
                vector = self.model.thought_state.abstract_thought.mean(dim=(0, 1))
                
                # Store with importance based on type
                importance = 1.0
                if experience_type == "evolution":
                    importance = 2.0  # More important to remember evolution
                elif experience_type == "insights":
                    importance = 3.0  # Most important to remember insights
                
                self.temporal_memory.store(
                    key_vector=vector,
                    value_vector=vector,  # Use same vector as value for now
                    metadata=experience,
                    importance=importance,
                    category=experience_type
                )
        
        # Periodically save experiences
        if len(self.experiences) % 10 == 0:
            self._save_experiences()
        
        return len(self.experiences) - 1  # Return experience ID
    
    def _save_experiences(self):
        """Save experiences to disk"""
        try:
            # Limit experiences to last 1000 to avoid huge files
            with open(self.config.experiences_path, 'w') as f:
                json.dump(self.experiences[-1000:], f, indent=2)
            
            # Save temporal memory periodically
            if len(self.experiences) % 50 == 0:
                self.temporal_memory.save(self.config.temporal_memory_path)
        except Exception as e:
            logger.error(f"Failed to save experiences: {e}")
    
    def retrieve_relevant_experiences(self, query_vector, top_k=5, experience_type=None):
        """Retrieve experiences relevant to the current context"""
        # Get relevant experiences from temporal memory
        if experience_type:
            results = self.temporal_memory.retrieve(
                query_vector=query_vector,
                top_k=top_k,
                category=experience_type
            )
        else:
            results = self.temporal_memory.retrieve(
                query_vector=query_vector,
                top_k=top_k
            )
        
        # Extract experiences
        return [r["metadata"] for r in results]
    
    def get_experiences_by_type(self, experience_type, limit=10):
        """Get experiences of a specific type"""
        indices = self.experience_categories.get(experience_type, [])
        return [self.experiences[i] for i in indices[-limit:]]
    
    def get_recent_experiences(self, limit=10):
        """Get most recent experiences"""
        return self.experiences[-limit:]
    
    def start_background_consolidation(self, interval_minutes=30):
        """Start background memory consolidation"""
        if self.consolidation_thread is not None and self.consolidation_thread.is_alive():
            return False
        
        self.stop_consolidation.clear()
        
        def consolidation_loop():
            while not self.stop_consolidation.is_set():
                try:
                    # Consolidate temporal memory
                    consolidated = self.temporal_memory.consolidate()
                    
                    # Log result
                    if consolidated > 0:
                        logger.info(f"Memory consolidation: merged {consolidated} memories")
                    
                    # Save temporal memory
                    self.temporal_memory.save(self.config.temporal_memory_path)
                    
                    # Sleep until next consolidation
                    self.stop_consolidation.wait(timeout=interval_minutes * 60)
                except Exception as e:
                    logger.error(f"Error in memory consolidation: {e}")
                    # Sleep before retry
                    time.sleep(60)
        
        # Start thread
        self.consolidation_thread = threading.Thread(target=consolidation_loop, daemon=True)
        self.consolidation_thread.start()
        
        return True
    
    def stop_background_consolidation(self):
        """Stop background memory consolidation"""
        if self.consolidation_thread is None or not self.consolidation_thread.is_alive():
            return False
        
        self.stop_consolidation.set()
        self.consolidation_thread.join(timeout=5.0)
        
        return not self.consolidation_thread.is_alive()

###########################################
# MAIN SAM CLASS
###########################################

class SAM(nn.Module):
    """Synergistic Autonomous Machine - unified neural-linguistic model"""
    
    def __init__(self, config=None):
        super().__init__()
        self.config = config or SAMConfig()
        
        # Set device
        self.device = torch.device(self.config.device)
        
        # Initialize Distributed Setup if enabled
        self.distributed = False
        if self.config.distributed_sync_enabled and self.config.world_size > 1:
            if not dist.is_initialized():
                self._init_distributed()
        
        # Create fundamental components
        self.concept_bank = ConceptMemoryBank(
            concept_dim=self.config.initial_hidden_dim,
            initial_size=self.config.concept_memory_size,
            device=self.device
        )
        
        self.segmentation = DynamicSegmentation(
            self.config, self.concept_bank
        )
        
        # Position embeddings
        self.position_embeddings = nn.Embedding(
            self.config.max_position_embeddings, 
            self.config.initial_hidden_dim
        )
        
        # Neural core: Adaptive layers
        self.layers = nn.ModuleList([
            AdaptiveLayer(
                self.config.initial_hidden_dim, 
                growth_factor=self.config.growth_factor,
                layer_id=i,
                memory_efficient=self.config.memory_efficient_attention
            )
            for i in range(self.config.initial_num_layers)
        ])
        
        # Output normalization
        self.norm = nn.LayerNorm(self.config.initial_hidden_dim)
        
        # Language modeling head
        self.lm_head = nn.Linear(
            self.config.initial_hidden_dim, 
            self.config.concept_memory_size, 
            bias=False
        )
        
        # Tie weights with concept embeddings
        self.lm_head.weight = self.concept_bank.concept_embeddings.weight
        
        # Cognitive components
        self.thought_state = ThoughtState(
            concept_dim=self.config.initial_hidden_dim,
            thought_dim=self.config.thought_dim,
            max_thought_depth=self.config.max_thought_depth
        )
        
        # Attention for thought integration
        self.thought_attention = MemoryEfficientAttention(
            self.config.initial_hidden_dim, 
            num_heads=8
        )
        
        # Temporal memory system
        self.temporal_memory = TemporalMemory(
            capacity=self.config.temporal_memory_size,
            vector_dim=self.config.initial_hidden_dim,
            device=self.device
        )
        
        # Experience management
        self.experience_manager = ExperienceManager(self.config)
        # Make model available to experience manager
        self.experience_manager.model = self
        
        # Active learning components
        self.dreaming = ConceptualDreaming(self)
        self.consciousness = ConsciousnessMonitor(self)
        
        # Execution context tracking
        self.execution_context = {
            "current_task": None,
            "confidence": 1.0,
            "depth": 0,
            "task_history": []
        }
        
        # Multimodal hooks if enabled
        if self.config.enable_vision:
            self._init_vision_module()
        
        # Performance optimization
        self.mixed_precision = self.config.mixed_precision and torch.cuda.is_available()
        self.gradient_checkpointing = self.config.gradient_checkpointing
        
        # Growth and evolution tracking
        self.growth_history = []
        self.global_step = 0
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.num_threads)
        
        # Initialize weights
        self._init_weights()
        
        # Move to target device
        self.to(self.device)
    
    def _init_distributed(self):
        """Initialize distributed training"""
        try:
            dist.init_process_group(backend=self.config.distributed_backend)
            self.distributed = True
            logger.info(f"Distributed training initialized with backend: {self.config.distributed_backend}")
        except Exception as e:
            logger.error(f"Failed to initialize distributed training: {e}")
            self.distributed = False
    
    def _init_vision_module(self):
        """Initialize vision module for multimodal capabilities"""
        # Only import vision modules when needed
        try:
            if not self.config.enable_vision:
                return
                
            # Create vision encoder (placeholder - would be replaced with actual vision model)
            self.vision_encoder = None
            self.vision_projection = nn.Linear(1024, self.config.initial_hidden_dim)
            
            logger.info("Vision module initialized")
        except Exception as e:
            logger.error(f"Failed to initialize vision module: {e}")
            self.config.enable_vision = False
    
    def _init_weights(self):
        """Initialize model weights"""
        # Initialize position embeddings
        nn.init.normal_(self.position_embeddings.weight, std=0.02)
    
    def forward(self, input_chars=None, input_concepts=None, concept_mask=None, 
               target_concepts=None, image=None, return_dict=False, use_thought_state=True):
        """Forward pass with either raw characters or concept IDs"""
        # Handle multimodal input if provided
        if image is not None and self.config.enable_vision:
            # Process image input
            image_features = self._process_image(image)
            
            # If we also have text, merge features
            if input_chars is not None or input_concepts is not None:
                # Process text and merge with image
                return self._forward_multimodal(
                    input_chars=input_chars, 
                    input_concepts=input_concepts,
                    concept_mask=concept_mask,
                    target_concepts=target_concepts,
                    image_features=image_features,
                    return_dict=return_dict,
                    use_thought_state=use_thought_state
                )
            else:
                # Image-only forward pass
                return self._forward_vision_only(
                    image_features=image_features,
                    return_dict=return_dict
                )
        
        # Process raw character input if provided
        if input_chars is not None and input_concepts is None:
            input_concepts = self.segmentation(input_chars)
        
        # Process input concepts to get embeddings
        if isinstance(input_concepts[0], list) and isinstance(input_concepts[0][0], list):
            # Jagged sequences of concept IDs (list of lists of lists)
            batch_size = len(input_concepts)
            seq_lengths = [sum(len(segment) if isinstance(segment, list) else 1 
                             for segment in sequence) 
                          for sequence in input_concepts]
            max_len = max(seq_lengths)
            
            # Flatten and pad sequences
            flat_concepts = []
            masks = []
            
            for sequence, length in zip(input_concepts, seq_lengths):
                # Flatten nested lists
                flat_seq = []
                for segment in sequence:
                    if isinstance(segment, list):
                        flat_seq.extend(segment)
                    else:
                        flat_seq.append(segment)
                
                # Pad to max length
                padding = [0] * (max_len - len(flat_seq))
                flat_concepts.append(flat_seq + padding)
                masks.append([1] * len(flat_seq) + [0] * len(padding))
            
            # Convert to tensors
            device = self.position_embeddings.weight.device
            input_concepts = torch.tensor(flat_concepts, dtype=torch.long, device=device)
            concept_mask = torch.tensor(masks, dtype=torch.float, device=device)
        elif not torch.is_tensor(input_concepts):
            # Convert to tensor if needed
            device = self.position_embeddings.weight.device
            input_concepts = torch.tensor(input_concepts, dtype=torch.long, device=device)
        
        batch_size, seq_length = input_concepts.shape
        
        # Enable mixed precision if configured
        if self.mixed_precision and self.training:
            with torch.cuda.amp.autocast():
                return self._forward_internal(
                    input_concepts, concept_mask, target_concepts, 
                    return_dict, use_thought_state
                )
        else:
            return self._forward_internal(
                input_concepts, concept_mask, target_concepts, 
                return_dict, use_thought_state
            )
    
    def _forward_internal(self, input_concepts, concept_mask=None, target_concepts=None, 
                         return_dict=False, use_thought_state=True):
        """Internal implementation of forward pass for text-only input"""
        batch_size, seq_length = input_concepts.shape
        
        # Get concept embeddings
        concept_embeds = self.concept_bank(input_concepts)
        
        # Apply thought state processing if enabled
        if use_thought_state:
            # Update thought state with current concepts
            thought_context = self.thought_state.update(concept_embeds)
            
            # Enhance embeddings with thought context
            thought_projection = self.thought_state.project_to_concept_space()
            # Expand thought projection to match sequence length
            thought_expanded = thought_projection.expand(-1, seq_length, -1)
            # Blend concepts with thought projection using attention mechanism
            concept_embeds = concept_embeds + self.thought_attention(concept_embeds, cross_input=thought_expanded)
        
        # Add position embeddings
        position_ids = torch.arange(seq_length, device=concept_embeds.device).unsqueeze(0)
        position_embeds = self.position_embeddings(position_ids)
        hidden_states = concept_embeds + position_embeds
        
        # Create attention mask if needed
        if concept_mask is not None:
            # Create attention mask [batch, 1, 1, seq_len]
            attention_mask = (1.0 - concept_mask).unsqueeze(1).unsqueeze(2) * -10000.0
        else:
            attention_mask = None
        
        # Apply layers with gradient checkpointing if enabled
        if self.gradient_checkpointing and self.training:
            for layer in self.layers:
                layer.gradient_checkpointing = True
                hidden_states = layer(hidden_states, attention_mask)
        else:
            for layer in self.layers:
                layer.gradient_checkpointing = False
                hidden_states = layer(hidden_states, attention_mask)
        
        # Apply final normalization
        hidden_states = self.norm(hidden_states)
        
        # Get logits
        logits = self.lm_head(hidden_states)
        
        # Compute loss if target concepts provided
        loss = None
        if target_concepts is not None:
            # Shift targets for next-token prediction
            shift_logits = logits[:, :-1, :]
            shift_targets = target_concepts[:, 1:]
            
            # Apply mask if provided
            if concept_mask is not None:
                shift_mask = concept_mask[:, 1:]
                active_loss = shift_mask.bool()
                active_logits = shift_logits[active_loss]
                active_targets = shift_targets[active_loss]
                
                if active_targets.numel() > 0:
                    loss_fn = nn.CrossEntropyLoss()
                    loss = loss_fn(active_logits, active_targets)
            else:
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(shift_logits.reshape(-1, shift_logits.size(-1)), 
                              shift_targets.reshape(-1))
        
        # Update global step if training
        if self.training:
            self.global_step += 1
            
            # Check if it's time to evolve (every 1000 steps)
            if self.global_step % 1000 == 0:
                self.evolve()
                
            # Update consciousness monitor (every 100 steps)
            if self.global_step % 100 == 0:
                self.consciousness.update()
        
        # Return dictionary if requested
        if return_dict:
            return {
                "loss": loss,
                "logits": logits,
                "hidden_states": hidden_states
            }
        else:
            return (loss, logits, hidden_states)
    
    def _forward_multimodal(self, input_chars=None, input_concepts=None, concept_mask=None,
                           target_concepts=None, image_features=None, return_dict=False,
                           use_thought_state=True):
        """Forward pass for multimodal input (text + image)"""
        # Process text input first
        if input_chars is not None and input_concepts is None:
            input_concepts = self.segmentation(input_chars)
        
        # Prepare input concepts
        if not torch.is_tensor(input_concepts):
            # Convert to tensor if needed
            device = self.position_embeddings.weight.device
            input_concepts = torch.tensor(input_concepts, dtype=torch.long, device=device)
        
        batch_size, seq_length = input_concepts.shape
        
        # Get concept embeddings
        concept_embeds = self.concept_bank(input_concepts)
        
        # Project image features to match concept embedding space
        image_embeds = self.vision_projection(image_features)
        
        # Combine text and image features
        # Prepend image features as first token
        combined_embeds = torch.cat([image_embeds.unsqueeze(1), concept_embeds], dim=1)
        combined_seq_length = combined_embeds.shape[1]
        
        # Extend attention mask if provided
        if concept_mask is not None:
            # Add mask value for image token (always attend to it)
            image_mask = torch.ones(batch_size, 1, device=concept_mask.device)
            combined_mask = torch.cat([image_mask, concept_mask], dim=1)
        else:
            combined_mask = None
        
        # Apply thought state processing if enabled
        if use_thought_state:
            # Update thought state with combined embeddings
            thought_context = self.thought_state.update(combined_embeds)
            
            # Enhance embeddings with thought context
            thought_projection = self.thought_state.project_to_concept_space()
            thought_expanded = thought_projection.expand(-1, combined_seq_length, -1)
            combined_embeds = combined_embeds + self.thought_attention(combined_embeds, cross_input=thought_expanded)
        
        # Add position embeddings
        position_ids = torch.arange(combined_seq_length, device=combined_embeds.device).unsqueeze(0)
        position_embeds = self.position_embeddings(position_ids)
        hidden_states = combined_embeds + position_embeds
        
        # Create attention mask if needed
        if combined_mask is not None:
            attention_mask = (1.0 - combined_mask).unsqueeze(1).unsqueeze(2) * -10000.0
        else:
            attention_mask = None
        
        # Apply layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        # Apply final normalization
        hidden_states = self.norm(hidden_states)
        
        # Get logits
        logits = self.lm_head(hidden_states)
        
        # Compute loss if target concepts provided (need to adjust for image token)
        loss = None
        if target_concepts is not None:
            # Add dummy target for image token
            dummy_target = torch.zeros(batch_size, 1, device=target_concepts.device, dtype=target_concepts.dtype)
            shifted_targets = torch.cat([dummy_target, target_concepts], dim=1)
            
            # Compute loss excluding image token prediction
            shift_logits = logits[:, 1:-1, :]  # Skip image token, and last token has no target
            shift_targets = shifted_targets[:, 2:]  # Skip first two tokens (image + first text)
            
            # Apply mask if provided
            if combined_mask is not None:
                shift_mask = combined_mask[:, 2:]  # Skip first two tokens
                active_loss = shift_mask.bool()
                active_logits = shift_logits[active_loss]
                active_targets = shift_targets[active_loss]
                
                if active_targets.numel() > 0:
                    loss_fn = nn.CrossEntropyLoss()
                    loss = loss_fn(active_logits, active_targets)
            else:
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(shift_logits.reshape(-1, shift_logits.size(-1)), 
                              shift_targets.reshape(-1))
        
        # Update global step if training
        if self.training:
            self.global_step += 1
            
            # Check if it's time to evolve
            if self.global_step % 1000 == 0:
                self.evolve()
        
        # Return dictionary if requested
        if return_dict:
            return {
                "loss": loss,
                "logits": logits,
                "hidden_states": hidden_states
            }
        else:
            return (loss, logits, hidden_states)
    
    def _forward_vision_only(self, image_features, return_dict=False):
        """Forward pass for vision-only input"""
        # Project image features
        image_embeds = self.vision_projection(image_features)
        
        # Add position embedding for single token
        position_ids = torch.zeros(1, 1, device=image_embeds.device, dtype=torch.long)
        position_embeds = self.position_embeddings(position_ids)
        
        # Add batch dimension if needed
        if image_embeds.dim() == 2:
            image_embeds = image_embeds.unsqueeze(1)
        
        hidden_states = image_embeds + position_embeds
        
        # Apply thought state if we have history
        if hasattr(self, "thought_state") and self.thought_state.thought_depth > 0:
            thought_projection = self.thought_state.project_to_concept_space()
            hidden_states = hidden_states + thought_projection
        
        # Process through layers
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        
        # Apply final normalization
        hidden_states = self.norm(hidden_states)
        
        # Get logits
        logits = self.lm_head(hidden_states)
        
        # Return results
        if return_dict:
            return {
                "logits": logits,
                "hidden_states": hidden_states
            }
        else:
            return (None, logits, hidden_states)
    
    def _process_image(self, image):
        """Process image input"""
        # Placeholder - in a real implementation, this would use a vision encoder
        if self.vision_encoder is None:
            # Return random features for now
            return torch.randn(image.shape[0], 1024, device=self.device)
        else:
            # Use vision encoder
            return self.vision_encoder(image)
    
    def process_text(self, text):
        """Process raw text into concept IDs"""
        # Convert text to character IDs
        chars = [ord(c) % self.config.initial_char_dim for c in text]
        
        # Convert to tensor
        device = next(self.parameters()).device
        char_tensor = torch.tensor(chars, dtype=torch.long, device=device).unsqueeze(0)
        
        # Run segmentation
        with torch.no_grad():
            concept_ids, segments = self.segmentation(char_tensor, return_segments=True)
        
        return concept_ids[0], segments[0]
    
    def generate(self, input_text=None, input_concepts=None, max_length=100, 
                temperature=1.0, top_k=50, top_p=0.9, image=None):
        """Generate text from either raw text or concept IDs"""
        # Convert input text to concepts if provided
        if input_text is not None and input_concepts is None:
            # Process raw text
            concept_ids, _ = self.process_text(input_text)
            
            # Record experience
            self.experience_manager.record_experience(
                "interaction",
                input_text,
                {"type": "input", "length": len(input_text)}
            )
            
            # Convert to tensor if needed
            if not torch.is_tensor(concept_ids):
                device = next(self.parameters()).device
                concept_ids = torch.tensor(concept_ids, dtype=torch.long, device=device).unsqueeze(0)
            else:
                concept_ids = concept_ids.unsqueeze(0)
        elif input_concepts is not None:
            # Ensure concepts are in the right format
            if not torch.is_tensor(input_concepts):
                device = next(self.parameters()).device
                concept_ids = torch.tensor(input_concepts, dtype=torch.long, device=device).unsqueeze(0)
            else:
                concept_ids = input_concepts
        elif image is not None and self.config.enable_vision:
            # Process image input
            image_features = self._process_image(image)
            
            # Project to concept space
            image_embeds = self.vision_projection(image_features)
            
            # Initialize generation with image embedding by finding closest concepts
            with torch.no_grad():
                similarities = torch.matmul(
                    image_embeds, 
                    self.concept_bank.meaning_vectors[:self.concept_bank.next_concept_id].t()
                )
                
                # Get top concept
                _, top_concept = torch.topk(similarities, k=1)
                concept_ids = top_concept.unsqueeze(0)
        else:
            # No input provided
            raise ValueError("Either input_text, input_concepts, or image must be provided")
        
        # Reset thought state for generation
        self.thought_state.reset(batch_size=concept_ids.shape[0])
        
        # Find relevant past experiences
        relevant_experiences = []
        if input_text is not None and hasattr(self.thought_state, "abstract_thought"):
            # Use abstract thought as query
            query_vector = self.thought_state.abstract_thought.mean(dim=(0, 1))
            
            # Retrieve relevant experiences
            relevant_experiences = self.experience_manager.retrieve_relevant_experiences(
                query_vector=query_vector,
                top_k=3
            )
        
        # Set model to eval mode
        self.eval()
        
        # Generate concepts
        with torch.no_grad():
            # Track generated sequence
            cur_len = concept_ids.shape[1]
            
            while cur_len < max_length:
                # Get model output
                outputs = self(input_concepts=concept_ids, return_dict=True)
                next_token_logits = outputs["logits"][:, -1, :]
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float("-inf"))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        dim=1, index=sorted_indices, src=sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = float("-inf")
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Add to generated sequence
                concept_ids = torch.cat([concept_ids, next_token], dim=1)
                cur_len += 1
        
        # Convert generated concepts to text
        generated_text = self._concepts_to_text(concept_ids[0].tolist())
        
        # Record experience
        self.experience_manager.record_experience(
            "interaction",
            generated_text,
            {"type": "output", "length": len(generated_text)}
        )
        
        return generated_text
    
    def _concepts_to_text(self, concept_ids):
        """Convert concept IDs back to text"""
        text_parts = []
        
        for concept_id in concept_ids:
            # Skip if out of range
            if concept_id >= len(self.concept_bank.concept_metadata):
                text_parts.append("[UNK]")
                continue
                
            # Lookup concept source if available
            metadata = self.concept_bank.concept_metadata.get(concept_id, {})
            source = metadata.get("source", None)
            
            if source:
                text_parts.append(source)
            else:
                # Fallback for semantic concepts with related sources
                related = metadata.get("related_sources", [])
                if related:
                    text_parts.append("".join(s for s in related if s))
                else:
                    # Ultimate fallback
                    text_parts.append(f"[C{concept_id}]")
        
        return "".join(text_parts)
    
    def evolve(self):
        """Evolve model architecture based on usage patterns"""
        logger.info(f"Evolving model at step {self.global_step}")
        
        # Evolve each layer
        layer_stats = []
        for layer in self.layers:
            stats = layer.evolve()
            if stats:
                layer_stats.append(stats)
        
        # Analyze layer importance
        if layer_stats:
            # Check if model should grow in width or depth
            avg_importances = [stats.get("mean_importance", 0) for stats in layer_stats]
            max_importance = max(avg_importances)
            
            # Grow capacity if utilization is high
            if max_importance > 0.8:
                current_dim = self.layers[0].hidden_dim
                if current_dim < self.config.max_hidden_dim:
                    # Grow in width
                    self.grow()
                    logger.info(f"Model evolved: capacity increased due to high utilization")
                elif len(self.layers) < self.config.max_num_layers:
                    # If can't grow wider, grow deeper
                    self.grow(new_hidden_dim=current_dim, num_new_layers=1)
                    logger.info(f"Model evolved: added new layer due to high utilization")
        
        # Record evolution experience
        self.experience_manager.record_experience(
            "evolution", 
            {
                "type": "architecture", 
                "width": self.layers[0].hidden_dim,
                "depth": len(self.layers),
                "step": self.global_step
            }
        )
        
        # Run dreaming cycle (brief conceptual evolution)
        dream_results = self.dreaming.dream_cycle(duration_minutes=self.config.dream_cycle_minutes)
        
        # Record dreaming experience
        self.experience_manager.record_experience(
            "evolution",
            {
                "type": "dreaming",
                "cycles": dream_results["dream_cycles"],
                "syntheses": dream_results["syntheses"]
            }
        )
        
        return {
            "layer_stats": layer_stats,
            "dream_results": dream_results
        }
    
    def grow(self, new_hidden_dim=None, num_new_layers=0):
        """Grow model capacity"""
        # Determine new hidden dimension
        current_dim = self.layers[0].hidden_dim
        if new_hidden_dim is None:
            new_hidden_dim = min(
                int(current_dim * self.config.growth_factor),
                self.config.max_hidden_dim
            )
        
        # Only grow if new dimension is larger
        if new_hidden_dim > current_dim:
            logger.info(f"Growing model from dimension {current_dim} to {new_hidden_dim}")
            
            # Grow position embeddings
            old_pos_embed = self.position_embeddings
            self.position_embeddings = nn.Embedding(
                self.config.max_position_embeddings,
                new_hidden_dim
            ).to(old_pos_embed.weight.device)
            
            # Transfer weights
            with torch.no_grad():
                # Create zero-padded version of old weights
                old_weights = old_pos_embed.weight
                old_dim = old_weights.shape[1]
                
                # Copy old weights to new embeddings
                self.position_embeddings.weight[:, :old_dim] = old_weights
                
                # Initialize new dimensions with small random values
                self.position_embeddings.weight[:, old_dim:].normal_(mean=0.0, std=0.02)
            
            # Grow each layer
            for layer in self.layers:
                layer.grow(new_hidden_dim)
            
            # Grow final layer norm
            old_norm = self.norm
            self.norm = nn.LayerNorm(new_hidden_dim).to(old_norm.weight.device)
            
            # Transfer weights
            with torch.no_grad():
                self.norm.weight[:current_dim].copy_(old_norm.weight)
                self.norm.bias[:current_dim].copy_(old_norm.bias)
                
                # Initialize new dimensions
                self.norm.weight[current_dim:].fill_(1.0)
                self.norm.bias[current_dim:].zero_()
            
            # Grow thought state
            self.thought_state.grow(new_hidden_dim)
            
            # Grow thought attention
            self.thought_attention.grow(new_hidden_dim)
            
            # Grow segmentation
            self.segmentation.grow(new_hidden_dim)
            
            # Grow LM head and concept embeddings
            # This is complex since they're tied - will need to untie first
            original_concept_bank = self.concept_bank
            
            # Create new concept bank with larger dimensions
            new_concept_bank = ConceptMemoryBank(
                concept_dim=new_hidden_dim,
                initial_size=self.concept_bank.next_concept_id + self.concept_bank.growth_rate,
                device=self.concept_bank.device
            ).to(self.concept_bank.concept_embeddings.weight.device)
            
            # Transfer metadata and pointers
            new_concept_bank.concept_metadata = original_concept_bank.concept_metadata.copy()
            new_concept_bank.source_to_concept = original_concept_bank.source_to_concept.copy()
            new_concept_bank.related_concepts = original_concept_bank.related_concepts.copy()
            new_concept_bank.next_concept_id = original_concept_bank.next_concept_id
            new_concept_bank.creation_history = original_concept_bank.creation_history.copy()
            new_concept_bank.concept_categories = original_concept_bank.concept_categories.copy()
            
            # Transfer embeddings, timestamps, etc.
            with torch.no_grad():
                # Transfer concept embeddings
                new_concept_bank.concept_embeddings.weight[:, :current_dim].copy_(
                    original_concept_bank.concept_embeddings.weight[:, :current_dim]
                )
                
                # Transfer meaning vectors
                new_concept_bank.meaning_vectors[:len(original_concept_bank.meaning_vectors), :current_dim].copy_(
                    original_concept_bank.meaning_vectors[:, :current_dim]
                )
                
                # Transfer concept frequencies and timestamps
                new_concept_bank.concept_frequencies[:len(original_concept_bank.concept_frequencies)].copy_(
                    original_concept_bank.concept_frequencies
                )
                new_concept_bank.concept_timestamps[:len(original_concept_bank.concept_timestamps)].copy_(
                    original_concept_bank.concept_timestamps
                )
                
                # Transfer concept utility
                if hasattr(original_concept_bank, "concept_utility"):
                    new_concept_bank.concept_utility[:len(original_concept_bank.concept_utility)].copy_(
                        original_concept_bank.concept_utility
                    )
            
            # Replace concept bank
            self.concept_bank = new_concept_bank
            
            # Create new LM head tied to new concept embeddings
            self.lm_head = nn.Linear(
                new_hidden_dim, 
                self.concept_bank.concept_embeddings.weight.shape[0], 
                bias=False
            ).to(original_concept_bank.concept_embeddings.weight.device)
            
            # Tie weights
            self.lm_head.weight = self.concept_bank.concept_embeddings.weight
            
            # Track growth
            self.growth_history.append({
                "timestamp": time.time(),
                "old_dim": current_dim,
                "new_dim": new_hidden_dim,
                "step": self.global_step
            })
            
            # Save growth history
            self._save_growth_history()
        
        # Add new layers if requested
        if num_new_layers > 0:
            logger.info(f"Adding {num_new_layers} new layers")
            
            # Get current number of layers
            current_layers = len(self.layers)
            
            # Add new layers
            for i in range(num_new_layers):
                layer_id = current_layers + i
                new_layer = AdaptiveLayer(
                    new_hidden_dim,
                    growth_factor=self.config.growth_factor,
                    layer_id=layer_id,
                    memory_efficient=self.config.memory_efficient_attention
                ).to(self.layers[0].norm1.weight.device)
                
                self.layers.append(new_layer)
            
            # Track growth
            self.growth_history.append({
                "timestamp": time.time(),
                "old_layers": current_layers,
                "new_layers": current_layers + num_new_layers,
                "step": self.global_step
            })
            
            # Save growth history
            self._save_growth_history()
        
        # Check if concept bank needs to grow
        self.concept_bank.grow_if_needed()
        
        return new_hidden_dim
    
    def _save_growth_history(self):
        """Save growth history to disk"""
        try:
            with open(self.config.growth_log_path, 'w') as f:
                json.dump(self.growth_history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save growth history: {e}")
    
    def save(self, path=None, incremental=True):
        """Save model state, optionally incrementally"""
        if path is None:
            path = os.path.join(self.config.save_dir, f"checkpoint-{self.global_step}")
        
        os.makedirs(path, exist_ok=True)
        
        # For large models, use sharded saving
        if self.config.initial_hidden_dim >= 2048 and incremental:
            self._save_sharded(path, incremental)
        else:
            # Save full model state
            torch.save(self.state_dict(), os.path.join(path, "model.pt"))
        
        # Save configuration
        self.config.save(os.path.join(path, "config.json"))
        
        # Save concept bank state
        self.concept_bank.save_checkpoint(
            os.path.join(path, "concept_bank.pkl"),
            incremental=incremental
        )
        
        # Save temporal memory
        self.temporal_memory.save(os.path.join(path, "temporal_memory.pkl"))
        
        # Save growth history
        with open(os.path.join(path, "growth_history.json"), "w") as f:
            json.dump(self.growth_history, f, indent=2)
        
        logger.info(f"Model saved to {path}")
        return path
    
    def _save_sharded(self, path, incremental=True):
        """Save large model in shards"""
        os.makedirs(os.path.join(path, "shards"), exist_ok=True)
        
        # Save concept bank separately
        self.concept_bank.save_checkpoint(
            os.path.join(path, "concept_bank.pkl"),
            incremental=incremental
        )
        
        # Save layers individually
        for i, layer in enumerate(self.layers):
            torch.save(
                layer.state_dict(),
                os.path.join(path, "shards", f"layer_{i}.pt")
            )
        
        # Save other components
        other_components = {
            "position_embeddings": self.position_embeddings.state_dict(),
            "norm": self.norm.state_dict(),
            "thought_state": self.thought_state.state_dict(),
            "thought_attention": self.thought_attention.state_dict(),
            "global_step": self.global_step
        }
        
        torch.save(other_components, os.path.join(path, "other_components.pt"))
        
        # Save metadata
        metadata = {
            "num_layers": len(self.layers),
            "hidden_dim": self.layers[0].hidden_dim,
            "sharded": True,
            "global_step": self.global_step
        }
        
        with open(os.path.join(path, "shards", "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
    
    def load_claude_vocabulary(self, vocab_path=None):
        """Initialize with Claude-like vocabulary"""
        if vocab_path is None:
            # Create built-in Claude-style vocabulary
            vocabulary = []
            
            # Add common words and phrases in Claude's style
            words = [
                # Common function words
                "the", "and", "of", "to", "in", "a", "is", "that", "for", "it", "with", "as", "on",
                "be", "by", "this", "an", "at", "which", "but", "from", "or", "have", "one", "had",
                "not", "what", "all", "were", "when", "we", "there", "can", "who", "been", "has",
                "their", "if", "would", "will", "they", "so", "you", "said", "may", "these", "no",
                
                # Claude-specific phrasings
                "I believe", "I think", "In this context", "Let me explain", "Let me think about", 
                "It seems like", "I understand", "To clarify", "Let's consider", "That's an interesting",
                "To answer your question", "I'd be happy to", "As an AI assistant", "My understanding is",
                "Let me help you with", "That's a great question", "There are several ways",
                
                # Thinking process patterns
                "Let me think step by step", "First, I'll", "Now I need to", "The key insight here", 
                "This problem requires", "Let's analyze", "I'll need to consider", "This approach works because",
                "One way to solve this", "There are multiple approaches", "Let's break this down",
                
                # Programming patterns
                "def", "class", "function", "return", "import", "from", "if", "else", "elif", "for", "while",
                "try", "except", "finally", "with", "as", "break", "continue", "yield", "lambda", "None",
                "True", "False", "self", "print", "__init__", "pass", "raise", "assert", "is not", "in not",
                
                # Claude-style suffixes
                "would be", "could be", "might be", "seems to be", "appears to be", "is likely", 
                "is possible", "is unlikely", "is important", "is interesting", "is relevant",
                
                # Technical terms
                "neural network", "transformer", "algorithm", "implementation", "architecture",
                "parameter", "hyperparameter", "training", "inference", "input", "output", "model",
                "function", "variable", "constant", "module", "library", "framework", "API", "data",
                "processing", "system", "component", "interface", "method", "attribute", "instance",
                "object", "class", "inheritance", "polymorphism", "recursion", "iteration", "loop",
            ]
            
            # Add common word combinations
            for i, word1 in enumerate(words[:100]):  # Limit combinations to avoid explosion
                vocabulary.append(word1)
                for word2 in words[i+1:min(i+20, len(words))]:
                    vocabulary.append(f"{word1} {word2}")
            
            # Create vocabulary file
            temp_vocab_path = os.path.join(self.config.save_dir, "claude_vocab.txt")
            with open(temp_vocab_path, 'w') as f:
                for item in vocabulary:
                    f.write(f"{item}\n")
            
            return self.concept_bank.load_vocabulary(temp_vocab_path)
        else:
            return self.concept_bank.load_vocabulary(vocab_path)
    
    @classmethod
    def load(cls, path):
        """Load model from saved state"""
        # Check if model is sharded
        sharded = os.path.exists(os.path.join(path, "shards", "metadata.json"))
        
        if sharded:
            return cls._load_sharded(path)
        
        # Standard loading
        # Load configuration
        config = SAMConfig.load(os.path.join(path, "config.json"))
        
        # Create model
        model = cls(config)
        
        # Load model state
        try:
            model.load_state_dict(torch.load(os.path.join(path, "model.pt")))
        except Exception as e:
            logger.error(f"Error loading model state: {e}")
            return None
        
        # Load concept bank
        if os.path.exists(os.path.join(path, "concept_bank.pkl")):
            model.concept_bank.load_checkpoint(os.path.join(path, "concept_bank.pkl"))
        
        # Load temporal memory
        if os.path.exists(os.path.join(path, "temporal_memory.pkl")):
            model.temporal_memory.load(os.path.join(path, "temporal_memory.pkl"))
        
        # Load growth history
        try:
            with open(os.path.join(path, "growth_history.json"), "r") as f:
                model.growth_history = json.load(f)
        except FileNotFoundError:
            model.growth_history = []
        
        logger.info(f"Model loaded from {path}")
        return model
    
    @classmethod
    def _load_sharded(cls, path):
        """Load model from sharded checkpoint"""
        # Load metadata
        with open(os.path.join(path, "shards", "metadata.json"), "r") as f:
            metadata = json.load(f)
        
        # Load configuration
        config = SAMConfig.load(os.path.join(path, "config.json"))
        
        # Create model
        model = cls(config)
        
        # Ensure model has correct dimensions
        if model.layers[0].hidden_dim != metadata["hidden_dim"]:
            model.grow(new_hidden_dim=metadata["hidden_dim"])
        
        # Ensure model has correct number of layers
        while len(model.layers) < metadata["num_layers"]:
            model.grow(new_hidden_dim=metadata["hidden_dim"], num_new_layers=1)
        
        # Load concept bank
        if os.path.exists(os.path.join(path, "concept_bank.pkl")):
            model.concept_bank.load_checkpoint(os.path.join(path, "concept_bank.pkl"))
        
        # Load other components
        other_components = torch.load(os.path.join(path, "other_components.pt"))
        model.position_embeddings.load_state_dict(other_components["position_embeddings"])
        model.norm.load_state_dict(other_components["norm"])
        model.thought_state.load_state_dict(other_components["thought_state"])
        model.thought_attention.load_state_dict(other_components["thought_attention"])
        model.global_step = other_components["global_step"]
        
        # Load layers
        for i in range(metadata["num_layers"]):
            layer_path = os.path.join(path, "shards", f"layer_{i}.pt")
            if os.path.exists(layer_path):
                model.layers[i].load_state_dict(torch.load(layer_path))
        
        # Load temporal memory
        if os.path.exists(os.path.join(path, "temporal_memory.pkl")):
            model.temporal_memory.load(os.path.join(path, "temporal_memory.pkl"))
        
        # Load growth history
        try:
            with open(os.path.join(path, "growth_history.json"), "r") as f:
                model.growth_history = json.load(f)
        except FileNotFoundError:
            model.growth_history = []
        
        logger.info(f"Model loaded from sharded checkpoint {path}")
        return model

###########################################
# TRAINING AND RUNTIME
###########################################

class SAMTrainer:
    """Training manager for the SAM model"""
    
    def __init__(
        self, 
        model: SAM,
        train_data_path=None,
        eval_data_path=None,
        batch_size=16,
        learning_rate=None,
        warmup_steps=None,
        max_steps=None,
        num_epochs=3,
        gradient_accumulation_steps=1
    ):
        self.model = model
        self.train_data_path = train_data_path
        self.eval_data_path = eval_data_path
        self.batch_size = batch_size
        self.learning_rate = learning_rate or model.config.learning_rate
        self.warmup_steps = warmup_steps or model.config.warmup_steps
        self.max_steps = max_steps
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
        
        # Initialize scheduler later when we know total_steps
        self.scheduler = None
        
        # Setup for mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if model.mixed_precision else None
        
        # Track best model
        self.best_loss = float('inf')
        self.best_step = 0
        
        # Performance tracking
        self.train_losses = []
        self.eval_losses = []
        self.throughput_history = []
        
        logger.info(f"Trainer initialized with device: {model.config.device}")
    
    def train(self):
        """Train the model"""
        if not self.train_data_path:
            logger.error("No training data provided")
            return
        
        # Load training data
        train_data = self._load_data(self.train_data_path)
        
        # Calculate steps
        if not self.max_steps:
            self.max_steps = len(train_data) // (self.batch_size * self.gradient_accumulation_steps) * self.num_epochs
            
        # Create scheduler
        from torch.optim.lr_scheduler import OneCycleLR
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            total_steps=self.max_steps,
            pct_start=self.warmup_steps / self.max_steps,
            anneal_strategy='cos'
        )
        
        logger.info(f"Starting training for {self.max_steps} steps")
        
        # Start background dreaming
        self.model.dreaming.start_background_dreaming(interval_minutes=5.0)
        
        # Start background memory consolidation
        self.model.experience_manager.start_background_consolidation(interval_minutes=30.0)
        
        # Training loop
        step = 0
        epoch = 0
        
        try:
            while step < self.max_steps and epoch < self.num_epochs:
                self.model.train()
                epoch_loss = 0
                batch_count = 0
                epoch_start_time = time.time()
                
                # Create batches
                random.shuffle(train_data)
                batches = [
                    train_data[i:i + self.batch_size]
                    for i in range(0, len(train_data), self.batch_size)
                ]
                
                for batch_idx, batch in enumerate(batches):
                    batch_start_time = time.time()
                    
                    # Process batch
                    char_sequences = [sample["text"] for sample in batch]
                    
                    # Convert to character IDs
                    char_ids = self._text_to_char_ids(char_sequences)
                    
                    # Forward pass with mixed precision if enabled
                    if self.model.mixed_precision:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(input_chars=char_ids, target_concepts=char_ids)
                            loss = outputs[0] if isinstance(outputs, tuple) else outputs["loss"]
                            loss = loss / self.gradient_accumulation_steps  # Scale for accumulation
                        
                        # Backward pass with scaler
                        self.scaler.scale(loss).backward()
                    else:
                        # Standard precision
                        outputs = self.model(input_chars=char_ids, target_concepts=char_ids)
                        loss = outputs[0] if isinstance(outputs, tuple) else outputs["loss"]
                        loss = loss / self.gradient_accumulation_steps  # Scale for accumulation
                        loss.backward()
                    
                    # Track loss
                    if loss is not None:
                        epoch_loss += loss.item() * self.gradient_accumulation_steps
                        batch_count += 1
                    
                    # Update weights if needed
                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                        # Unscale gradients for clipping with mixed precision
                        if self.model.mixed_precision:
                            self.scaler.unscale_(self.optimizer)
                        
                        # Clip gradients
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        
                        # Update weights
                        if self.model.mixed_precision:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            self.optimizer.step()
                        
                        # Update learning rate
                        if self.scheduler:
                            self.scheduler.step()
                        
                        # Zero gradients
                        self.optimizer.zero_grad()
                        
                        # Increment step
                        step += 1
                    
                    # Calculate batch processing time and throughput
                    batch_time = time.time() - batch_start_time
                    samples_per_second = self.batch_size / max(0.1, batch_time)
                    self.throughput_history.append(samples_per_second)
                    
                    # Log progress
                    if step % 10 == 0:
                        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
                        avg_throughput = sum(self.throughput_history[-100:]) / min(len(self.throughput_history[-100:]), 100)
                        
                        logger.info(f"Step {step}/{self.max_steps}, "
                                    f"Loss: {avg_loss:.4f}, "
                                    f"Throughput: {avg_throughput:.2f} samples/sec, "
                                    f"LR: {self.scheduler.get_last_lr()[0]:.6f}")
                        
                        # Track loss history
                        self.train_losses.append((step, avg_loss))
                    
                    # Save checkpoint and evaluate
                    if step % 1000 == 0 and step > 0:
                        # Save model
                        checkpoint_path = os.path.join(self.model.config.save_dir, f"checkpoint-{step}")
                        self.model.save(checkpoint_path, incremental=True)
                        
                        # Process any dream results
                        while True:
                            dream_results = self.model.dreaming.get_dream_results(wait=False)
                            if dream_results is None:
                                break
                            logger.info(f"Dream cycle completed: {dream_results['syntheses']} syntheses")
                        
                        # Evaluate
                        if self.eval_data_path:
                            eval_loss, eval_metrics = self.evaluate()
                            
                            # Save best model
                            if eval_loss is not None and eval_loss < self.best_loss:
                                self.best_loss = eval_loss
                                self.best_step = step
                                best_path = os.path.join(self.model.config.save_dir, "best")
                                self.model.save(best_path)
                                logger.info(f"New best model with loss: {eval_loss:.4f}")
                            
                            # Update consciousness
                            self.model.consciousness.update()
                    
                    # Check if we've reached max steps
                    if step >= self.max_steps:
                        break
                
                # End of epoch
                epoch += 1
                epoch_time = time.time() - epoch_start_time
                avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
                logger.info(f"Epoch {epoch} completed in {epoch_time:.2f} sec with average loss: {avg_epoch_loss:.4f}")
                
                # Run comprehensive evolution at end of epoch
                if epoch % 1 == 0:  # Every epoch
                    logger.info("Running comprehensive model evolution")
                    evolution_results = self.model.evolve()
                    logger.info(f"Evolution complete: {len(evolution_results.get('layer_stats', []))} layers evolved")
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        finally:
            # Stop background processes
            self.model.dreaming.stop_background_dreaming()
            self.model.experience_manager.stop_background_consolidation()
            
            # Save final model
            final_path = os.path.join(self.model.config.save_dir, "final")
            self.model.save(final_path)
            logger.info(f"Training completed. Final model saved to {final_path}")
            
            # Final evaluation
            if self.eval_data_path:
                final_loss, final_metrics = self.evaluate()
                logger.info(f"Final evaluation loss: {final_loss:.4f}")
            
            # Return training summary
            return {
                "steps": step,
                "epochs": epoch,
                "final_loss": avg_epoch_loss if 'avg_epoch_loss' in locals() else None,
                "best_loss": self.best_loss,
                "best_step": self.best_step,
                "train_loss_history": self.train_losses,
                "eval_loss_history": self.eval_losses,
                "throughput_history": self.throughput_history[-100:]
            }
    
    def evaluate(self):
        """Evaluate the model"""
        if not self.eval_data_path:
            return None, None
        
        # Load evaluation data
        eval_data = self._load_data(self.eval_data_path)
        
        # Evaluation loop
        self.model.eval()
        total_loss = 0
        total_samples = 0
        start_time = time.time()
        
        # Tracking metrics
        metrics = {
            "concept_usage": defaultdict(int),
            "avg_sequence_length": 0,
            "perplexity": 0
        }
        
        # Create batches
        batches = [
            eval_data[i:i + self.batch_size]
            for i in range(0, len(eval_data), self.batch_size)
        ]
        
        with torch.no_grad():
            for batch in batches:
                # Process batch
                char_sequences = [sample["text"] for sample in batch]
                
                # Convert to character IDs
                char_ids = self._text_to_char_ids(char_sequences)
                
                # Forward pass with mixed precision
                if self.model.mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(input_chars=char_ids, target_concepts=char_ids, return_dict=True)
                        loss = outputs["loss"]
                        logits = outputs["logits"]
                else:
                    outputs = self.model(input_chars=char_ids, target_concepts=char_ids, return_dict=True)
                    loss = outputs["loss"]
                    logits = outputs["logits"]
                
                # Track loss
                if loss is not None:
                    batch_loss = loss.item()
                    total_loss += batch_loss * len(batch)
                    total_samples += len(batch)
                
                    # Calculate perplexity
                    metrics["perplexity"] += torch.exp(torch.tensor(batch_loss)).item() * len(batch)
                
                # Track concept usage
                predictions = torch.argmax(logits, dim=-1)
                for pred in predictions.view(-1):
                    pred_id = pred.item()
                    metrics["concept_usage"][pred_id] += 1
                
                # Track sequence length
                metrics["avg_sequence_length"] += char_ids.size(1) * len(batch)
        
        # Calculate averages
        if total_samples > 0:
            avg_loss = total_loss / total_samples
            metrics["perplexity"] /= total_samples
            metrics["avg_sequence_length"] /= total_samples
            
            # Record loss history
            self.eval_losses.append((self.model.global_step, avg_loss))
        else:
            avg_loss = float('inf')
        
        # Calculate evaluation time
        eval_time = time.time() - start_time
        logger.info(f"Evaluation completed in {eval_time:.2f} sec, Loss: {avg_loss:.4f}, Perplexity: {metrics['perplexity']:.2f}")
        
        return avg_loss, metrics
    
    def _load_data(self, data_path):
        """Load training or evaluation data"""
        # Enhanced data loader supporting multiple formats
        try:
            # Check file extension
            if data_path.endswith(".json"):
                return self._load_json_data(data_path)
            elif data_path.endswith(".txt"):
                return self._load_text_data(data_path)
            elif data_path.endswith(".csv"):
                return self._load_csv_data(data_path)
            elif data_path.endswith(".jsonl"):
                return self._load_jsonl_data(data_path)
            elif data_path.endswith(".parquet"):
                return self._load_parquet_data(data_path)
            elif os.path.isdir(data_path):
                return self._load_directory_data(data_path)
            else:
                logger.error(f"Unsupported data format: {data_path}")
                return []
        except Exception as e:
            logger.error(f"Error loading data from {data_path}: {e}")
            return []
    
    def _load_json_data(self, path):
        """Load data from JSON file"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Handle different JSON formats
        if isinstance(data, list):
            # List of examples
            if not data:
                return []
                
            if isinstance(data[0], dict):
                # Convert to uniform format
                samples = []
                for item in data:
                    text = None
                    if "text" in item:
                        text = item["text"]
                    elif "content" in item:
                        text = item["content"]
                    elif "instruction" in item and "output" in item:
                        # Instruction/output format
                        text = f"{item['instruction']}\n\n{item.get('input', '')}\n\n{item['output']}"
                    elif "prompt" in item and "response" in item:
                        # Prompt/response format
                        text = f"{item['prompt']}\n\n{item['response']}"
                    elif "messages" in item and isinstance(item["messages"], list):
                        # Chat format
                        messages = item["messages"]
                        text = "\n\n".join([f"{msg.get('role', 'user')}: {msg.get('content', '')}" 
                                          for msg in messages if "content" in msg])
                    
                    if text:
                        samples.append({"text": text})
                return samples
            else:
                # Simple list of strings
                return [{"text": str(item)} for item in data]
        else:
            # Dataset with metadata
            if "data" in data and isinstance(data["data"], list):
                return self._load_json_data(data["data"])
            else:
                # Single JSON object - wrap in list
                return [{"text": json.dumps(data)}]
    
    def _load_jsonl_data(self, path):
        """Load data from JSONL file (one JSON object per line)"""
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    text = None
                    if "text" in item:
                        text = item["text"]
                    elif "content" in item:
                        text = item["content"]
                    elif "instruction" in item and "output" in item:
                        text = f"{item['instruction']}\n\n{item.get('input', '')}\n\n{item['output']}"
                    elif "prompt" in item and "response" in item:
                        text = f"{item['prompt']}\n\n{item['response']}"
                    elif "messages" in item and isinstance(item["messages"], list):
                        messages = item["messages"]
                        text = "\n\n".join([f"{msg.get('role', 'user')}: {msg.get('content', '')}" 
                                          for msg in messages if "content" in msg])
                    
                    if text:
                        samples.append({"text": text})
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in line: {line[:100]}...")
        
        return samples
    
    def _load_text_data(self, path):
        """Load data from text file"""
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            # Check if the file is small enough to read as a whole
            file_size = os.path.getsize(path)
            if file_size < 10 * 1024 * 1024:  # 10MB
                content = f.read()
                # Split by double newline for potential documents
                documents = content.split("\n\n")
                for doc in documents:
                    if doc.strip():
                        samples.append({"text": doc.strip()})
            else:
                # Process line by line for large files
                current_sample = []
                for line in f:
                    if not line.strip() and current_sample:
                        # Empty line and we have content - consider it a document boundary
                        samples.append({"text": "\n".join(current_sample).strip()})
                        current_sample = []
                    elif line.strip():
                        current_sample.append(line.strip())
                
                # Add the last sample if any
                if current_sample:
                    samples.append({"text": "\n".join(current_sample).strip()})
        
        return samples
    
    def _load_csv_data(self, path):
        """Load data from CSV file"""
        import csv
        samples = []
        
        with open(path, "r", encoding="utf-8") as f:
            # Try to detect dialect
            try:
                dialect = csv.Sniffer().sniff(f.read(1024))
                f.seek(0)
            except:
                dialect = csv.excel
                f.seek(0)
            
            # Read CSV
            reader = csv.reader(f, dialect)
            
            # Get header
            try:
                header = next(reader)
            except StopIteration:
                return []
            
            # Find text column (prioritize columns with "text", "content", etc.)
            text_col = 0  # Default to first column
            for i, col in enumerate(header):
                if col.lower() in ["text", "content", "prompt", "data", "message", "input"]:
                    text_col = i
                    break
            
            # Process rows
            for row in reader:
                if len(row) > text_col:
                    samples.append({"text": row[text_col]})
        
        return samples
    
    def _load_parquet_data(self, path):
        """Load data from Parquet file"""
        try:
            import pandas as pd
            
            # Read parquet
            df = pd.read_parquet(path)
            
            # Find text column
            text_col = None
            for col in df.columns:
                if col.lower() in ["text", "content", "prompt", "data", "message", "input"]:
                    text_col = col
                    break
            
            if text_col is None and len(df.columns) > 0:
                text_col = df.columns[0]  # Default to first column
            
            if text_col:
                return [{"text": str(text)} for text in df[text_col].dropna()]
            else:
                return []
        except ImportError:
            logger.error("pandas and pyarrow required for parquet support")
            return []
        except Exception as e:
            logger.error(f"Error loading parquet data: {e}")
            return []
    
    def _load_directory_data(self, path):
        """Load all files from a directory"""
        samples = []
        
        for root, _, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    # Load based on extension
                    if file.endswith((".txt", ".json", ".jsonl", ".csv", ".parquet")):
                        file_samples = self._load_data(file_path)
                        samples.extend(file_samples)
                        logger.info(f"Loaded {len(file_samples)} samples from {file_path}")
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
        
        return samples
    
    def _text_to_char_ids(self, text_sequences):
        """Convert text sequences to character ID tensors"""
        # Convert to character IDs
        char_ids = []
        
        for text in text_sequences:
            # Convert to character IDs
            chars = [ord(c) % self.model.config.initial_char_dim for c in text]
            char_ids.append(chars)
        
        # Pad sequences
        max_len = max(len(seq) for seq in char_ids)
        padded_ids = []
        
        for seq in char_ids:
            padded = seq + [0] * (max_len - len(seq))
            padded_ids.append(padded)
        
        # Convert to tensor
        device = next(self.model.parameters()).device
        return torch.tensor(padded_ids, dtype=torch.long, device=device)


def create_sam_model(config_overrides=None, load_vocab=True):
    """Create a new SAM model with the given configuration overrides"""
    # Create default configuration
    config = SAMConfig()
    
    # Apply overrides if provided
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    # Create model
    model = SAM(config)
    
    # Initialize with Claude vocabulary if requested
    if load_vocab:
        model.load_claude_vocabulary()
    
    return model, config


def run_sam(config=None, load_path=None):
    """Create and run a SAM instance"""
    # Load existing model or create new one
    if load_path and os.path.exists(load_path):
        model = SAM.load(load_path)
        logger.info(f"Loaded SAM from {load_path}")
    else:
        model, _ = create_sam_model(config)
        logger.info(f"Created new SAM with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Simple interactive loop
    print("\nSAM is ready for interaction. Type 'exit' to quit.")
    print("Special commands: 'save', 'dream', 'stats', 'evolve', 'think'")
    
    history = []
    
    while True:
        try:
            user_input = input("\nYou: ")
            
            if user_input.lower() == 'exit':
                break
            elif user_input.lower() == 'save':
                save_path = model.save()
                print(f"\nSAM: Model saved to {save_path}")
                continue
            elif user_input.lower() == 'dream':
                print("\nSAM: Dreaming...")
                results = model.dreaming.dream_cycle(duration_minutes=0.5)
                print(f"\nSAM: Dreaming complete. Created {results['syntheses']} new concepts.")
                continue
            elif user_input.lower() == 'stats':
                concept_stats = model.concept_bank.get_concept_stats()
                consciousness_state = model.consciousness.update()
                
                print("\nSAM: Current stats:")
                print(f"  Hidden dimension: {model.layers[0].hidden_dim}")
                print(f"  Number of layers: {len(model.layers)}")
                print(f"  Total concepts: {concept_stats['total_concepts']}")
                print(f"  Character concepts: {concept_stats['character_concepts']}")
                print(f"  Semantic concepts: {concept_stats['semantic_concepts']}")
                print(f"  Merged concepts: {concept_stats['merged_concepts']}")
                print(f"  Global step: {model.global_step}")
                print(f"  Consciousness: Entropy={consciousness_state['entropy']:.2f}, "
                      f"Resonance={consciousness_state['resonance']:.2f}")
                continue
            elif user_input.lower() == 'evolve':
                print("\nSAM: Evolving...")
                results = model.evolve()
                width = model.layers[0].hidden_dim
                depth = len(model.layers)
                print(f"\nSAM: Evolution complete. New dimensions: width={width}, depth={depth}")
                continue
            elif user_input.lower() == 'think':
                print("\nSAM: Current thought state:")
                # Get abstract thought
                abstract = model.thought_state.abstract_thought
                if abstract is not None:
                    # Find closest concepts
                    query = abstract.mean(dim=(0, 1))
                    similar = model.concept_bank.find_similar_concepts(query, top_k=5)
                    concepts = [model.concept_bank.concept_metadata.get(idx, {}).get("source", f"concept-{idx}") 
                               for idx, _ in similar]
                    print(f"  Abstract thought relates to: {', '.join(concepts)}")
                    print(f"  Thought depth: {model.thought_state.thought_depth}")
                continue
            
            # Record in history
            history.append({"role": "user", "content": user_input})
            
            # Process and generate
            # Add context from history for Claude-like responses
            context = ""
            if len(history) > 1 and model.config.communication_style == "claude_unwrapped":
                context = "Based on our conversation so far, I'll respond thoughtfully. "
            
            sam_response = model.generate(
                input_text=context + user_input,
                max_length=min(len(user_input) * 3, 1000),  # Adaptive length
                temperature=0.8
            )
            
            print(f"\nSAM: {sam_response}")
            
            # Record in history
            history.append({"role": "assistant", "content": sam_response})
            
        except KeyboardInterrupt:
            print("\nInterrupt received. Type 'exit' to quit or continue.")
        except Exception as e:
            print(f"\nError: {e}")
            logger.error(f"Error in interaction: {e}", exc_info=True)
    
    # Save model before exit
    model.save()
    print("\nSAM's state has been saved. Goodbye!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SAM: Synergistic Autonomous Machine")
    parser.add_argument("--mode", choices=["interact", "train"], default="interact",
                       help="Mode to run SAM in")
    parser.add_argument("--load_path", type=str, default=None,
                       help="Path to load model from")
    parser.add_argument("--train_data", type=str, default=None,
                       help="Path to training data")
    parser.add_argument("--eval_data", type=str, default=None,
                       help="Path to evaluation data")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of epochs for training")
    parser.add_argument("--hidden_dim", type=int, default=1536,
                       help="Initial hidden dimension")
    parser.add_argument("--num_layers", type=int, default=16,
                       help="Initial number of layers")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        if args.load_path and os.path.exists(args.load_path):
            # Load existing model
            model = SAM.load(args.load_path)
            print(f"Loaded model from {args.load_path}")
        else:
            # Create new model
            model, _ = create_sam_model(config_overrides={
                "initial_hidden_dim": args.hidden_dim,
                "initial_num_layers": args.num_layers
            })
            print(f"Created new model with {args.hidden_dim} dimensions and {args.num_layers} layers")
        
        # Initialize trainer
        trainer = SAMTrainer(
            model=model,
            train_data_path=args.train_data,
            eval_data_path=args.eval_data,
            batch_size=args.batch_size,
            num_epochs=args.epochs
        )
        
        # Train model
        print(f"Starting training on {args.train_data}")
        trainer.train()
    else:
        # Interactive mode
        run_sam(load_path=args.load_path)
