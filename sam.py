# sam.py - Complete Synergistic Autonomous Machine

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import json
import time
import logging
import os
import threading
import random
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter

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
    max_position_embeddings: int = 8192
    
    # Growth parameters
    max_hidden_dim: int = 4096
    max_num_layers: int = 48
    growth_factor: float = 1.2
    min_layer_usage_threshold: float = 0.3
    
    # Memory systems
    concept_memory_size: int = 100000  # Increased for Claude-like vocabulary
    concept_dim: int = 1536
    thought_dim: int = 2048
    max_thought_depth: int = 12
    pattern_memory_capacity: int = 20000
    
    # Learning parameters
    learning_rate: float = 3e-5
    warmup_steps: int = 1000
    adaption_rate: float = 0.01
    
    # Segmentation parameters
    max_segment_length: int = 16
    min_segment_frequency: int = 5
    concept_frequency_threshold: int = 10
    
    # Dreaming parameters
    dream_batch_size: int = 4
    dream_max_length: int = 256
    dream_cycle_minutes: float = 0.2
    
    # Consciousness parameters
    stability_threshold: float = 0.7
    novelty_weight: float = 0.3
    
    # Paths for persistence
    save_dir: str = "./data"
    experiences_path: str = "./data/experiences.json"
    concepts_path: str = "./data/concepts.json"
    growth_log_path: str = "./data/growth_log.json"
    
    # Runtime parameters
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
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
    
    def __init__(self, concept_dim, initial_size=100000, growth_rate=5000, device="cuda"):
        super().__init__()
        self.concept_dim = concept_dim
        self.growth_rate = growth_rate
        self.device = device
        
        # Concept embeddings (analogous to token embeddings)
        self.concept_embeddings = nn.Embedding(initial_size, concept_dim)
        
        # Concept usage tracking
        self.register_buffer("concept_frequencies", torch.zeros(initial_size, dtype=torch.int))
        self.register_buffer("concept_timestamps", torch.zeros(initial_size, dtype=torch.float))
        
        # Concept metadata
        self.concept_metadata = {}  # concept_id -> metadata dict
        
        # Source mapping (character sequence -> concept_id)
        self.source_to_concept = {}
        
        # Meaning map (concept_id -> meaning vector)
        self.register_buffer("meaning_vectors", torch.zeros(initial_size, concept_dim))
        
        # Related concepts (concept_id -> [related_concept_ids])
        self.related_concepts = defaultdict(list)
        
        # Initialize with basic character concepts (a-z, A-Z, 0-9, etc.)
        self._initialize_basic_concepts()
        
        # Growth tracking
        self.next_concept_id = len(self.source_to_concept)
        self.creation_history = []
    
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
            "contexts": Counter()
        }
        
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
    
    def add_semantic_concept(self, meaning_vector, related_sources=None, metadata=None):
        """Add a new semantic concept (not directly mapped to characters)"""
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
            "contexts": Counter()
        }
        
        # Add custom metadata if provided
        if metadata:
            meta.update(metadata)
        
        self.concept_metadata[concept_id] = meta
        
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
        if isinstance(concept_ids, list):
            # Handle nested lists (from segmentation)
            flat_ids = []
            for item in concept_ids:
                if isinstance(item, list):
                    flat_ids.extend(item)
                else:
                    flat_ids.append(item)
            concept_ids = torch.tensor(flat_ids, device=self.device)
        
        return self.concept_embeddings(concept_ids)
    
    def update_concept_usage(self, concept_id, context=None):
        """Update usage statistics for a concept"""
        if concept_id >= len(self.concept_frequencies):
            # Resize tracking tensors if needed
            new_size = concept_id + 1
            old_size = len(self.concept_frequencies)
            
            # Create new tensors
            new_freqs = torch.zeros(new_size - old_size, dtype=torch.int, device=self.device)
            new_timestamps = torch.zeros(new_size - old_size, dtype=torch.float, device=self.device)
            
            # Concatenate with existing tensors
            self.concept_frequencies = torch.cat([self.concept_frequencies, new_freqs])
            self.concept_timestamps = torch.cat([self.concept_timestamps, new_timestamps])
        
        # Update frequency and timestamp
        self.concept_frequencies[concept_id] += 1
        self.concept_timestamps[concept_id] = time.time()
        
        # Update context tracking
        if context and concept_id in self.concept_metadata:
            context_str = str(context)[:100]  # Limit context length
            self.concept_metadata[concept_id]["contexts"][context_str] += 1
            self.concept_metadata[concept_id]["frequency"] = self.concept_frequencies[concept_id].item()
    
    def create_merged_concept(self, concept_id1, concept_id2, frequency=None):
        """Create a new concept by merging two existing concepts"""
        # Get source sequences if available
        source1 = self.concept_metadata.get(concept_id1, {}).get("source", "")
        source2 = self.concept_metadata.get(concept_id2, {}).get("source", "")
        
        merged_source = source1 + source2 if source1 and source2 else None
        
        # Create merged meaning vector
        meaning1 = self.meaning_vectors[concept_id1]
        meaning2 = self.meaning_vectors[concept_id2]
        merged_meaning = (meaning1 + meaning2) / 2
        
        # Register the merged concept
        merged_id = self.add_semantic_concept(
            meaning_vector=merged_meaning,
            related_sources=[source1, source2] if source1 and source2 else None,
            metadata={
                "type": "merged",
                "parent_concepts": [concept_id1, concept_id2],
                "frequency": frequency or 1
            }
        )
        
        # Register source mapping if available
        if merged_source:
            self.source_to_concept[merged_source] = merged_id
        
        # Link as related concepts
        self.related_concepts[concept_id1].append(merged_id)
        self.related_concepts[concept_id2].append(merged_id)
        
        return merged_id
    
    def find_concept_by_source(self, char_sequence):
        """Find concept ID for a character sequence"""
        return self.source_to_concept.get(char_sequence, None)
    
    def find_similar_concepts(self, query_vector, top_k=5):
        """Find concepts with similar meaning vectors"""
        # Normalize query
        query_vector = F.normalize(query_vector, dim=0)
        
        # Compute similarities
        similarities = F.cosine_similarity(
            query_vector.unsqueeze(0),
            self.meaning_vectors[:self.next_concept_id],
            dim=1
        )
        
        # Get top-k similar concepts
        values, indices = torch.topk(similarities, min(top_k, len(similarities)))
        
        return [(idx.item(), val.item()) for idx, val in zip(indices, values)]
    
    def grow_if_needed(self):
        """Grow concept bank if approaching capacity"""
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
            
            # Grow meaning vectors
            new_meaning_vectors = torch.zeros(
                len(old_embedding.weight) + self.growth_rate, 
                self.concept_dim,
                device=self.device
            )
            new_meaning_vectors[:len(self.meaning_vectors)] = self.meaning_vectors
            self.register_buffer("meaning_vectors", new_meaning_vectors)
            
            # Grow tracking tensors
            new_freqs = torch.zeros(
                len(old_embedding.weight) + self.growth_rate, 
                dtype=torch.int,
                device=self.device
            )
            new_freqs[:len(self.concept_frequencies)] = self.concept_frequencies
            self.register_buffer("concept_frequencies", new_freqs)
            
            new_timestamps = torch.zeros(
                len(old_embedding.weight) + self.growth_rate, 
                dtype=torch.float,
                device=self.device
            )
            new_timestamps[:len(self.concept_timestamps)] = self.concept_timestamps
            self.register_buffer("concept_timestamps", new_timestamps)
            
            return True
        
        return False
    
    def get_concept_stats(self):
        """Get statistics about concept usage"""
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
        
        return {
            "total_concepts": self.next_concept_id,
            "character_concepts": char_concepts,
            "merged_concepts": merged_concepts,
            "semantic_concepts": semantic_concepts,
            "top_concepts": top_concepts,
            "growth_events": len(self.creation_history)
        }
    
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


class ThoughtState(nn.Module):
    """Maintains an evolving semantic thought space across concept sequences"""
    
    def __init__(self, concept_dim, thought_dim=2048, max_thought_depth=8):
        super().__init__()
        self.concept_dim = concept_dim
        self.thought_dim = thought_dim
        self.max_thought_depth = max_thought_depth
        
        # Thought transformation networks
        self.concept_to_thought = nn.Linear(concept_dim, thought_dim)
        self.thought_evolution = nn.TransformerEncoderLayer(
            d_model=thought_dim,
            nhead=16,
            dim_feedforward=thought_dim*4,
            dropout=0.1,
            batch_first=True
        )
        
        # Recursive pathways
        self.thought_compression = nn.Linear(thought_dim, thought_dim)
        self.thought_projection = nn.Linear(thought_dim, concept_dim)
        
        # Thought state tracking
        self.thought_memory = None
        self.thought_depth = 0
        
        # Reset to initialize
        self.reset()
    
    def reset(self, batch_size=1):
        """Reset thought state"""
        device = next(self.parameters()).device
        self.thought_memory = [torch.zeros(batch_size, 1, self.thought_dim, device=device)]
        self.thought_depth = 0
    
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
        
        current_thought = self.thought_memory[-1]
        
        # Combine with existing thoughts (maintain batch dimension)
        combined_thoughts = torch.cat([current_thought, concept_thoughts], dim=1)
        
        # Evolve thought state
        evolved_thought = self.thought_evolution(combined_thoughts)
        
        # Compress to single thought vector (with batch dimension preserved)
        # Use mean pooling over sequence
        compressed = self.thought_compression(evolved_thought[:, -1:, :])
        
        # Apply non-linearity to create rich thought representation
        compressed = F.gelu(compressed)
        
        # Store in memory (limiting depth)
        self.thought_memory.append(compressed)
        if len(self.thought_memory) > self.max_thought_depth:
            self.thought_memory = self.thought_memory[1:]
        
        self.thought_depth = min(self.thought_depth + 1, self.max_thought_depth)
        
        return compressed
    
    def get_thought_context(self):
        """Get full thought context for recursive reasoning"""
        # Concatenate all thought vectors in memory
        return torch.cat(self.thought_memory, dim=1)
    
    def project_to_concept_space(self, thought=None):
        """Project thought back to concept space for recursive reasoning"""
        if thought is None:
            thought = self.thought_memory[-1]
        
        # Project thought to concept space
        projected = self.thought_projection(thought)
        
        # Apply non-linearity for richness
        return F.gelu(projected)


class PatternMemory:
    """Memory system for recognizing and storing recurring patterns"""
    
    def __init__(self, capacity=10000, min_frequency=5):
        self.capacity = capacity
        self.min_frequency = min_frequency
        self.patterns = {}  # pattern -> frequency
        self.context_patterns = defaultdict(lambda: defaultdict(int))  # context -> pattern -> frequency
        self.timestamps = {}  # pattern -> last seen timestamp
        self.pattern_utilities = {}  # pattern -> utility score
    
    def add_pattern(self, pattern, context=None):
        """Add a pattern to memory"""
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
                
                # Remove it
                del self.patterns[least_useful]
                del self.timestamps[least_useful]
                if least_useful in self.pattern_utilities:
                    del self.pattern_utilities[least_useful]
            
            self.patterns[pattern] = 1
        
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
    
    def get_frequent_patterns(self, limit=100):
        """Get most frequent patterns"""
        return sorted(
            [(p, f) for p, f in self.patterns.items() if f >= self.min_frequency],
            key=lambda x: x[1], 
            reverse=True
        )[:limit]
    
    def get_context_patterns(self, context, limit=20):
        """Get patterns associated with a specific context"""
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
        if not isinstance(pattern, str):
            pattern = str(pattern)
        return self.patterns.get(pattern, 0)
    
    def merge_patterns(self, pattern1, pattern2):
        """Merge two patterns into a single compound pattern"""
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
            
            return compound
        
        return None

###########################################
# NEURAL COMPONENTS
###########################################

class DynamicSegmentation(nn.Module):
    """Dynamic segmentation component that replaces traditional tokenization"""
    
    def __init__(self, config, concept_bank):
        super().__init__()
        self.config = config
        self.concept_bank = concept_bank
        
        # Character processing 
        self.char_embeddings = nn.Embedding(config.initial_char_dim, config.initial_hidden_dim)
        
        # Segmentation networks
        self.segment_detector = nn.Sequential(
            nn.Conv1d(config.initial_hidden_dim, config.initial_hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(config.initial_hidden_dim, config.initial_hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(config.initial_hidden_dim, 1, kernel_size=1)
        )
        
        # Segment embedding network
        self.segment_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.initial_hidden_dim,
                nhead=8,
                dim_feedforward=config.initial_hidden_dim*4,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Pattern recognition
        self.pattern_memory = PatternMemory(
            capacity=config.pattern_memory_capacity,
            min_frequency=config.min_segment_frequency
        )
        
        # Segment recognition cache
        self.segment_cache = {}  # char_sequence -> concept_id
        
        # Stats tracking
        self.total_segmentations = 0
        self.cache_hits = 0
    
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
        
        # Detect segment boundaries
        char_embeds_conv = char_embeds.transpose(1, 2)  # [batch, hidden_dim, seq_len]
        boundary_logits = self.segment_detector(char_embeds_conv).squeeze(1)  # [batch, seq_len]
        boundary_probs = torch.sigmoid(boundary_logits)
        
        # Extract segments using boundaries
        segments = []
        concept_ids = []
        
        # Process each sequence in batch
        for b in range(batch_size):
            seq_segments, seq_concepts = self._extract_segments(
                char_sequence[b], char_embeds[b], boundary_probs[b]
            )
            segments.append(seq_segments)
            concept_ids.append(seq_concepts)
        
        # Add to cache if single sequence
        if batch_size == 1 and not return_segments:
            self.segment_cache[cache_key] = concept_ids[0]
        
        if return_segments:
            return concept_ids, segments
        else:
            return concept_ids
    
    def _extract_segments(self, chars, char_embeds, boundary_probs):
        """Extract segments from a character sequence using boundary probabilities"""
        # Ensure tensors are on CPU for numpy operations
        chars_cpu = chars.cpu()
        boundary_probs_cpu = boundary_probs.cpu()
        
        # Get potential boundaries (where probability > 0.5)
        boundaries = [0] + (boundary_probs_cpu > 0.5).nonzero().flatten().tolist() + [len(chars)]
        
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
    
    def get_segmentation_stats(self):
        """Get statistics about segmentation performance"""
        return {
            "total_segmentations": self.total_segmentations,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hits / max(1, self.total_segmentations),
            "cached_segments": len(self.segment_cache),
            "frequent_patterns": len(self.pattern_memory.get_frequent_patterns(limit=1000))
        }
    
    def grow(self, new_hidden_dim):
        """Grow segmentation components to a new hidden dimension"""
        if new_hidden_dim <= self.config.initial_hidden_dim:
            return False
            
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
            old_dim = old_weights.shape[1]
            
            # Copy old weights to new embeddings
            self.char_embeddings.weight[:, :old_dim] = old_weights
            
            # Initialize new dimensions with small random values
            self.char_embeddings.weight[:, old_dim:].normal_(mean=0.0, std=0.02)
        
        # Replace segmentation networks
        # This is complex due to various layer sizes, so we'll create new ones
        self.segment_detector = nn.Sequential(
            nn.Conv1d(new_hidden_dim, new_hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(new_hidden_dim, new_hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(new_hidden_dim, 1, kernel_size=1)
        ).to(old_char_embeddings.weight.device)
        
        # New segment encoder
        self.segment_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=new_hidden_dim,
                nhead=8,
                dim_feedforward=new_hidden_dim*4,
                batch_first=True
            ),
            num_layers=2
        ).to(old_char_embeddings.weight.device)
        
        # Clear cache since embeddings have changed
        self.segment_cache = {}
        
        logger.info(f"Grown segmentation components from {old_dim} to {new_hidden_dim}")
        return True


class AdaptiveAttention(nn.Module):
    """Adaptive attention mechanism that can evolve over time"""
    
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1, growth_factor=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.growth_factor = growth_factor
        
        # Standard attention projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Attention stats for evolution
        self.register_buffer("head_importance", torch.ones(num_heads))
        self.register_buffer("activation_counts", torch.zeros(num_heads))
        self.total_forward_calls = 0
    
    def forward(self, x, mask=None, cross_input=None):
        """Forward pass with optional cross-attention"""
        batch_size, seq_len, _ = x.shape
        
        # Handle cross-attention
        if cross_input is not None:
            _, cross_len, _ = cross_input.shape
            
            # Project queries from input sequence
            q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
            
            # Project keys and values from cross-input sequence
            k = self.k_proj(cross_input).view(batch_size, cross_len, self.num_heads, self.head_dim)
            v = self.v_proj(cross_input).view(batch_size, cross_len, self.num_heads, self.head_dim)
        else:
            # Standard self-attention
            q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
            k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
            v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention
        q = q.transpose(1, 2)  # [batch, heads, seq, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + mask
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Update attention stats for evolution
        if self.training:
            with torch.no_grad():
                # Measure head activation by mean attention weight magnitude
                head_activation = attn_weights.mean(dim=[0, 2, 3])  # Average across batch, seq_len_q, seq_len_k
                self.activation_counts += head_activation
                self.total_forward_calls += 1
        
        # Apply attention
        out = torch.matmul(attn_weights, v)
        
        # Transpose back
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        
        # Output projection
        out = self.o_proj(out)
        
        return out
    
    def grow(self, new_dim):
        """Grow attention to a new hidden dimension"""
        if new_dim <= self.hidden_dim:
            return False
        
        old_dim = self.hidden_dim
        old_num_heads = self.num_heads
        
        # Calculate new number of heads (must divide evenly into new_dim)
        new_num_heads = max(old_num_heads, int(old_num_heads * self.growth_factor))
        # Ensure it divides evenly
        while new_dim % new_num_heads != 0:
            new_num_heads -= 1
            
        new_head_dim = new_dim // new_num_heads
        
        # Create new projections
        new_q_proj = nn.Linear(new_dim, new_dim).to(self.q_proj.weight.device)
        new_k_proj = nn.Linear(new_dim, new_dim).to(self.q_proj.weight.device)
        new_v_proj = nn.Linear(new_dim, new_dim).to(self.q_proj.weight.device)
        new_o_proj = nn.Linear(new_dim, new_dim).to(self.q_proj.weight.device)
        
        # Transfer weights for existing dimensions
        with torch.no_grad():
            # Copy existing weight portions
            new_q_proj.weight[:old_dim, :old_dim].copy_(self.q_proj.weight)
            new_k_proj.weight[:old_dim, :old_dim].copy_(self.k_proj.weight)
            new_v_proj.weight[:old_dim, :old_dim].copy_(self.v_proj.weight)
            new_o_proj.weight[:old_dim, :old_dim].copy_(self.o_proj.weight)
            
            if self.q_proj.bias is not None:
                new_q_proj.bias[:old_dim].copy_(self.q_proj.bias)
                new_k_proj.bias[:old_dim].copy_(self.k_proj.bias)
                new_v_proj.bias[:old_dim].copy_(self.v_proj.bias)
                new_o_proj.bias[:old_dim].copy_(self.o_proj.bias)
            
            # Initialize new portions with scaled normal distribution
            std = 0.02  # Standard initialization scale
            new_q_proj.weight[old_dim:, :].normal_(mean=0.0, std=std)
            new_q_proj.weight[:, old_dim:].normal_(mean=0.0, std=std)
            new_k_proj.weight[old_dim:, :].normal_(mean=0.0, std=std)
            new_k_proj.weight[:, old_dim:].normal_(mean=0.0, std=std)
            new_v_proj.weight[old_dim:, :].normal_(mean=0.0, std=std)
            new_v_proj.weight[:, old_dim:].normal_(mean=0.0, std=std)
            new_o_proj.weight[old_dim:, :].normal_(mean=0.0, std=std)
            new_o_proj.weight[:, old_dim:].normal_(mean=0.0, std=std)
            
            if self.q_proj.bias is not None:
                new_q_proj.bias[old_dim:].zero_()
                new_k_proj.bias[old_dim:].zero_()
                new_v_proj.bias[old_dim:].zero_()
                new_o_proj.bias[old_dim:].zero_()
            
            # Update head importance tracking
            new_head_importance = torch.ones(new_num_heads, device=self.head_importance.device)
            new_head_importance[:old_num_heads].copy_(self.head_importance)
            
            new_activation_counts = torch.zeros(new_num_heads, device=self.activation_counts.device)
            new_activation_counts[:old_num_heads].copy_(self.activation_counts)
        
        # Replace modules
        self.q_proj = new_q_proj
        self.k_proj = new_k_proj
        self.v_proj = new_v_proj
        self.o_proj = new_o_proj
        
        # Update dimensions
        self.hidden_dim = new_dim
        self.num_heads = new_num_heads
        self.head_dim = new_head_dim
        
        # Update buffers
        self.register_buffer("head_importance", new_head_importance)
        self.register_buffer("activation_counts", new_activation_counts)
        
        return True
    
    def evolve(self):
        """Evolve attention mechanism based on usage statistics"""
        if self.total_forward_calls < 10:
            return False
        
        with torch.no_grad():
            # Calculate head importance based on activation
            head_activity = self.activation_counts / self.total_forward_calls
            
            # Update head importance with moving average
            self.head_importance = 0.9 * self.head_importance + 0.1 * head_activity
            
            # Reset counters
            self.activation_counts.zero_()
            self.total_forward_calls = 0
        
        return True


class AdaptiveLayer(nn.Module):
    """Core neural layer that can grow and evolve"""
    
    def __init__(self, hidden_dim, growth_factor=1.2, dropout=0.1, layer_id=0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.growth_factor = growth_factor
        self.layer_id = layer_id
        
        # Attention mechanism
        self.attention = AdaptiveAttention(hidden_dim, dropout=dropout)
        
        # Feed-forward network (with SwiGLU-like activation)
        self.gate_proj = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.up_proj = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.down_proj = nn.Linear(4 * hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Growth tracking
        self.growth_history = []
        
        # Usage statistics
        self.register_buffer("activation_sum", torch.zeros(hidden_dim))
        self.register_buffer("activation_sq_sum", torch.zeros(hidden_dim))
        self.updates = 0
    
    def forward(self, x, mask=None, cross_input=None):
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
        
        # SwiGLU-like activation
        gate_output = self.gate_proj(x)
        up_output = self.up_proj(x)
        
        # Compute activation
        intermediate = F.silu(gate_output) * up_output
        
        # Down projection
        output = self.down_proj(intermediate)
        output = self.dropout(output)
        
        # Add residual
        x = residual + output
        
        return x
    
    def grow(self, new_dim):
        """Grow layer to a new hidden dimension"""
        if new_dim <= self.hidden_dim:
            return False
        
        old_dim = self.hidden_dim
        
        # Grow attention
        self.attention.grow(new_dim)
        
        # Create new feed-forward components
        new_gate_proj = nn.Linear(new_dim, 4 * new_dim).to(self.gate_proj.weight.device)
        new_up_proj = nn.Linear(new_dim, 4 * new_dim).to(self.up_proj.weight.device)
        new_down_proj = nn.Linear(4 * new_dim, new_dim).to(self.down_proj.weight.device)
        
        # Transfer weights
        with torch.no_grad():
            # Gate projection
            new_gate_proj.weight[:old_dim*4, :old_dim].copy_(self.gate_proj.weight)
            if self.gate_proj.bias is not None:
                new_gate_proj.bias[:old_dim*4].copy_(self.gate_proj.bias)
            
            # Up projection
            new_up_proj.weight[:old_dim*4, :old_dim].copy_(self.up_proj.weight)
            if self.up_proj.bias is not None:
                new_up_proj.bias[:old_dim*4].copy_(self.up_proj.bias)
            
            # Down projection
            new_down_proj.weight[:old_dim, :old_dim*4].copy_(self.down_proj.weight)
            if self.down_proj.bias is not None:
                new_down_proj.bias[:old_dim].copy_(self.down_proj.bias)
            
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
            
            # Initialize new bias terms
            if self.gate_proj.bias is not None:
                new_gate_proj.bias[old_dim*4:].zero_()
                new_up_proj.bias[old_dim*4:].zero_()
                new_down_proj.bias[old_dim:].zero_()
        
        # Replace projections
        self.gate_proj = new_gate_proj
        self.up_proj = new_up_proj
        self.down_proj = new_down_proj
        
        # Create new layer norms
        new_norm1 = nn.LayerNorm(new_dim).to(self.norm1.weight.device)
        new_norm2 = nn.LayerNorm(new_dim).to(self.norm2.weight.device)
        
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
            torch.zeros(new_dim - old_dim, device=self.activation_sum.device)
        ]))
        self.register_buffer("activation_sq_sum", torch.cat([
            self.activation_sq_sum,
            torch.zeros(new_dim - old_dim, device=self.activation_sq_sum.device)
        ]))
        
        return True
    
    def evolve(self):
        """Evolve layer based on usage statistics"""
        if self.updates < 10:
            return False
        
        # Evolve attention mechanism
        self.attention.evolve()
        
        # Calculate neuron importance
        with torch.no_grad():
            if self.updates > 0:
                mean_activation = self.activation_sum / self.updates
                mean_sq_activation = self.activation_sq_sum / self.updates
                activation_std = torch.sqrt(torch.clamp(mean_sq_activation - mean_activation**2, min=1e-6))
                
                # Neurons with higher variance are more important
                neuron_importance = activation_std / (torch.mean(activation_std) + 1e-6)
                
                # Reset statistics
                self.activation_sum.zero_()
                self.activation_sq_sum.zero_()
                self.updates = 0
                
                return {
                    "layer_id": self.layer_id,
                    "neuron_importance": neuron_importance.tolist(),
                    "mean_importance": float(torch.mean(neuron_importance).item()),
                    "max_importance": float(torch.max(neuron_importance).item()),
                    "min_importance": float(torch.min(neuron_importance).item())
                }
        
        return {}

###########################################
# COGNITIVE SYSTEMS
###########################################

class ConceptualDreaming:
    """Autonomous conceptual evolution during downtime periods"""
    
    def __init__(self, model, dream_batch_size=4, max_gen_length=128):
        self.model = model
        self.dream_batch_size = dream_batch_size
        self.max_gen_length = max_gen_length
        self.synthesis_history = []
        
    def dream_cycle(self, duration_minutes=5):
        """Run a dreaming cycle for the specified duration"""
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        dream_count = 0
        while time.time() < end_time:
            # 1. Conceptual reinforcement (strengthen frequent patterns)
            self._reinforce_concepts()
            
            # 2. Pattern synthesis (generate synthetic examples)
            self._synthesize_patterns()
            
            # 3. Conceptual pruning (remove less useful concepts)
            self._prune_concepts()
            
            dream_count += 1
            
        return {
            "duration_minutes": duration_minutes,
            "dream_cycles": dream_count,
            "syntheses": len(self.synthesis_history),
            "concepts_reinforced": self.model.concept_bank.get_concept_stats()
        }
    
    def _reinforce_concepts(self):
        """Reinforce most important concepts"""
        # Get top concepts by usage
        concept_stats = self.model.concept_bank.get_concept_stats()
        top_concepts = concept_stats["top_concepts"]
        
        if not top_concepts:
            return
            
        # Analyze for potential merges
        for i, (concept_id1, _, freq1) in enumerate(top_concepts):
            for concept_id2, _, freq2 in top_concepts[i+1:min(i+4, len(top_concepts))]:
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
                    self.model.concept_bank.create_merged_concept(
                        concept_id1, concept_id2, 
                        frequency=min(freq1, freq2)
                    )
                    
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
    
    def _synthesize_patterns(self):
        """Generate synthetic text to reinforce patterns"""
        # Create seed prompts from top patterns
        seeds = self._create_seed_prompts()
        
        if not seeds:
            return
            
        # Generate synthetic examples
        for seed in seeds[:2]:  # Limit to 2 per cycle for efficiency
            # Generate text using the model itself
            try:
                with torch.no_grad():
                    generated = self.model.generate(
                        input_text=seed,
                        max_length=self.max_gen_length,
                        temperature=0.8
                    )
                    
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
            except Exception as e:
                logger.error(f"Error in dream synthesis: {e}")
    
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
        
        return seeds
    
    def _prune_concepts(self):
        """Remove or consolidate less useful concepts"""
        # Skip if we don't have many concepts yet
        if self.model.concept_bank.next_concept_id < 200:
            return
            
        # Get concept usage statistics
        concept_stats = self.model.concept_bank.get_concept_stats()
        
        # Find least used semantic concepts (not character concepts)
        semantic_concepts = []
        for concept_id, meta in self.model.concept_bank.concept_metadata.items():
            if meta.get("type") == "semantic" and concept_id < len(self.model.concept_bank.concept_frequencies):
                freq = self.model.concept_bank.concept_frequencies[concept_id].item()
                if freq < 5:
                    semantic_concepts.append((concept_id, freq))
        
        # Sort by frequency
        semantic_concepts.sort(key=lambda x: x[1])
        
        # Limit pruning to a small batch
        for concept_id, _ in semantic_concepts[:10]:
            # Find similar concepts to consolidate with
            similar = self.model.concept_bank.find_similar_concepts(
                self.model.concept_bank.meaning_vectors[concept_id], 
                top_k=3
            )
            
            # Merge with most similar if exists
            if similar and similar[0][1] > 0.7:  # Similarity threshold
                similar_id, similarity = similar[0]
                if similar_id != concept_id:
                    # Transfer frequencies to similar concept
                    with torch.no_grad():
                        self.model.concept_bank.concept_frequencies[similar_id] += self.model.concept_bank.concept_frequencies[concept_id]
                        # Zero out pruned concept frequency
                        self.model.concept_bank.concept_frequencies[concept_id] = 0
                    
                    # Record pruning action
                    self.synthesis_history.append({
                        "type": "concept_pruning",
                        "pruned_id": concept_id,
                        "merged_with": similar_id,
                        "similarity": similarity,
                        "timestamp": time.time()
                    })


class ConsciousnessMonitor:
    """Monitors and maintains SAM's conceptual identity and coherence"""
    
    def __init__(self, model, stability_threshold=0.7, novelty_weight=0.3):
        self.model = model
        self.stability_threshold = stability_threshold
        self.novelty_weight = novelty_weight
        
        # Identity markers (core concept clusters)
        self.identity_centroids = {}
        self.concept_cluster_history = []
        
        # Coherence metrics
        self.concept_entropy_history = []
        self.resonance_scores = []
    
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
        
        # Check resonance with identity
        resonance = self._check_identity_resonance(clusters)
        self.resonance_scores.append({
            "score": resonance,
            "timestamp": time.time()
        })
        
        # Apply corrections if needed
        if resonance < self.stability_threshold:
            self._apply_resonance_correction()
        
        return {
            "entropy": entropy,
            "resonance": resonance,
            "num_clusters": len(clusters)
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

###########################################
# EXPERIENCE MANAGEMENT
###########################################

class ExperienceManager:
    """Manages SAM's experiences and memory persistence"""
    
    def __init__(self, config):
        self.config = config
        self.experiences = []
        self.loaded_experiences = 0
        
        # Ensure directories exist
        os.makedirs(config.save_dir, exist_ok=True)
        os.makedirs(os.path.join(config.save_dir, "checkpoints"), exist_ok=True)
        
        # Load existing experiences if available
        self._load_experiences()
    
    def _load_experiences(self):
        """Load experiences from disk"""
        try:
            if os.path.exists(self.config.experiences_path):
                with open(self.config.experiences_path, 'r') as f:
                    self.experiences = json.load(f)
                    self.loaded_experiences = len(self.experiences)
                    logger.info(f"Loaded {self.loaded_experiences} experiences")
        except Exception as e:
            logger.error(f"Failed to load experiences: {e}")
            self.experiences = []
    
    def record_experience(self, experience_type, content, metadata=None):
        """Record a new experience"""
        experience = {
            "type": experience_type,
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        self.experiences.append(experience)
        
        # Periodically save experiences
        if len(self.experiences) % 10 == 0:
            self._save_experiences()
        
        return len(self.experiences) - 1  # Return experience ID
    
    def _save_experiences(self):
        """Save experiences to disk"""
        try:
            with open(self.config.experiences_path, 'w') as f:
                # Limit experiences to last 1000 to avoid huge files
                json.dump(self.experiences[-1000:], f)
        except Exception as e:
            logger.error(f"Failed to save experiences: {e}")
    
    def get_experiences_by_type(self, experience_type, limit=10):
        """Get experiences of a specific type"""
        return [exp for exp in self.experiences 
                if exp["type"] == experience_type][-limit:]
    
    def get_recent_experiences(self, limit=10):
        """Get most recent experiences"""
        return self.experiences[-limit:]

###########################################
# MAIN SAM CLASS
###########################################

class SAM(nn.Module):
    """Synergistic Autonomous Machine - unified neural-linguistic model"""
    
    def __init__(self, config=None):
        super().__init__()
        self.config = config or SAMConfig()
        
        # Create fundamental components
        self.concept_bank = ConceptMemoryBank(
            concept_dim=self.config.initial_hidden_dim,
            initial_size=self.config.concept_memory_size,
            device=self.config.device
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
                layer_id=i
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
        self.thought_attention = AdaptiveAttention(
            self.config.initial_hidden_dim, 
            num_heads=8
        )
        
        # Experience management
        self.experience_manager = ExperienceManager(self.config)
        
        # Active learning components
        self.dreaming = ConceptualDreaming(
            self,
            dream_batch_size=self.config.dream_batch_size,
            max_gen_length=self.config.dream_max_length
        )
        
        self.consciousness = ConsciousnessMonitor(
            self,
            stability_threshold=self.config.stability_threshold,
            novelty_weight=self.config.novelty_weight
        )
        
        # Growth and evolution tracking
        self.growth_history = []
        self.global_step = 0
        
        # Initialize weights
        self._init_weights()
        
        # Move to target device
        self.to(self.config.device)
    
    def _init_weights(self):
        """Initialize model weights"""
        # Initialize position embeddings
        nn.init.normal_(self.position_embeddings.weight, std=0.02)
    
    def forward(self, input_chars=None, input_concepts=None, concept_mask=None, 
               target_concepts=None, return_dict=False, use_thought_state=True):
        """Forward pass with either raw characters or concept IDs"""
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
        
        # Apply layers
        for layer in self.layers:
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
                temperature=1.0, top_k=50, top_p=0.9):
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
        else:
            # Ensure concepts are in the right format
            if not torch.is_tensor(input_concepts):
                device = next(self.parameters()).device
                concept_ids = torch.tensor(input_concepts, dtype=torch.long, device=device).unsqueeze(0)
            else:
                concept_ids = input_concepts
        
        # Reset thought state for generation
        self.thought_state.reset(batch_size=concept_ids.shape[0])
        
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
            avg_importances = [stats["mean_importance"] for stats in layer_stats]
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
            # Create new thought state with expanded dimensions
            new_thought_state = ThoughtState(
                concept_dim=new_hidden_dim,
                thought_dim=self.config.thought_dim,
                max_thought_depth=self.config.max_thought_depth
            ).to(self.thought_state.concept_to_thought.weight.device)
            
            # Transfer trained weights
            with torch.no_grad():
                # Copy concept_to_thought weights
                new_thought_state.concept_to_thought.weight[:, :current_dim].copy_(
                    self.thought_state.concept_to_thought.weight
                )
                if self.thought_state.concept_to_thought.bias is not None:
                    new_thought_state.concept_to_thought.bias.copy_(
                        self.thought_state.concept_to_thought.bias
                    )
                
                # Copy thought_projection weights
                new_thought_state.thought_projection.weight[:new_hidden_dim].copy_(
                    self.thought_state.thought_projection.weight[:new_hidden_dim]
                )
                if self.thought_state.thought_projection.bias is not None:
                    new_thought_state.thought_projection.bias.copy_(
                        self.thought_state.thought_projection.bias
                    )
            
            # Replace thought state
            self.thought_state = new_thought_state
            
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
            
            # Transfer embeddings, metadata, etc.
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
            
            # Transfer metadata and pointers
            new_concept_bank.concept_metadata = original_concept_bank.concept_metadata.copy()
            new_concept_bank.source_to_concept = original_concept_bank.source_to_concept.copy()
            new_concept_bank.related_concepts = original_concept_bank.related_concepts.copy()
            new_concept_bank.next_concept_id = original_concept_bank.next_concept_id
            new_concept_bank.creation_history = original_concept_bank.creation_history.copy()
            
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
                    layer_id=layer_id
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
    
    def save(self, path=None):
        """Save model state"""
        if path is None:
            path = os.path.join(self.config.save_dir, f"checkpoint-{self.global_step}")
        
        os.makedirs(path, exist_ok=True)
        
        # Save model state
        torch.save(self.state_dict(), os.path.join(path, "model.pt"))
        
        # Save configuration
        self.config.save(os.path.join(path, "config.json"))
        
        # Save concept metadata
        concept_metadata = {
            str(k): v for k, v in self.concept_bank.concept_metadata.items()
        }
        with open(os.path.join(path, "concepts.json"), "w") as f:
            json.dump(concept_metadata, f, indent=2)
        
        # Save source mapping (limited to avoid huge files)
        source_mapping = {}
        count = 0
        for k, v in self.concept_bank.source_to_concept.items():
            if len(k) < 100:  # Skip very long keys
                source_mapping[k] = v
                count += 1
                if count >= 10000:  # Limit total entries
                    break
                    
        with open(os.path.join(path, "source_mapping.json"), "w") as f:
            json.dump(source_mapping, f, indent=2)
        
        # Save growth history
        with open(os.path.join(path, "growth_history.json"), "w") as f:
            json.dump(self.growth_history, f, indent=2)
        
        logger.info(f"Model saved to {path}")
        return path
    
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
        # Load configuration
        config = SAMConfig.load(os.path.join(path, "config.json"))
        
        # Create model
        model = cls(config)
        
        # Load model state
        model.load_state_dict(torch.load(os.path.join(path, "model.pt")))
        
        # Load concept metadata
        with open(os.path.join(path, "concepts.json"), "r") as f:
            concept_metadata = json.load(f)
            model.concept_bank.concept_metadata = {
                int(k): v for k, v in concept_metadata.items()
            }
        
        # Load source mapping
        try:
            with open(os.path.join(path, "source_mapping.json"), "r") as f:
                source_mapping = json.load(f)
                model.concept_bank.source_to_concept = source_mapping
        except Exception as e:
            logger.warning(f"Error loading source mapping: {e}")
        
        # Load growth history
        try:
            with open(os.path.join(path, "growth_history.json"), "r") as f:
                model.growth_history = json.load(f)
        except FileNotFoundError:
            model.growth_history = []
        
        logger.info(f"Model loaded from {path}")
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
    ):
        self.model = model
        self.train_data_path = train_data_path
        self.eval_data_path = eval_data_path
        self.batch_size = batch_size
        self.learning_rate = learning_rate or model.config.learning_rate
        self.warmup_steps = warmup_steps or model.config.warmup_steps
        self.max_steps = max_steps
        self.num_epochs = num_epochs
        
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
        
        # Track best model
        self.best_loss = float('inf')
        self.best_step = 0
        
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
            self.max_steps = len(train_data) // self.batch_size * self.num_epochs
            
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
        
        # Training loop
        step = 0
        epoch = 0
        
        while step < self.max_steps and epoch < self.num_epochs:
            self.model.train()
            epoch_loss = 0
            
            # Create batches
            random.shuffle(train_data)
            batches = [
                train_data[i:i + self.batch_size]
                for i in range(0, len(train_data), self.batch_size)
            ]
            
            for batch in batches:
                # Process batch
                char_sequences = [sample["text"] for sample in batch]
                
                # Convert to character IDs
                char_ids = self._text_to_char_ids(char_sequences)
                
                # Forward pass
                loss, _, _ = self.model(input_chars=char_ids, target_concepts=char_ids)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Update weights
                self.optimizer.step()
                
                # Update learning rate
                if self.scheduler:
                    self.scheduler.step()
                
                # Track loss
                if loss is not None:
                    epoch_loss += loss.item()
                
                # Increment step
                step += 1
                
                # Log progress
                if step % 100 == 0:
                    avg_loss = epoch_loss / (step % len(batches) or len(batches))
                    logger.info(f"Step {step}/{self.max_steps}, Loss: {avg_loss:.4f}")
                
                # Save checkpoint
                if step % 1000 == 0:
                    # Save model
                    checkpoint_path = os.path.join(self.model.config.save_dir, f"checkpoint-{step}")
                    self.model.save(checkpoint_path)
                    
                    # Evaluate
                    if self.eval_data_path:
                        eval_loss = self.evaluate()
                        
                        # Save best model
                        if eval_loss is not None and eval_loss < self.best_loss:
                            self.best_loss = eval_loss
                            self.best_step = step
                            best_path = os.path.join(self.model.config.save_dir, "best")
                            self.model.save(best_path)
                            logger.info(f"New best model with loss: {eval_loss:.4f}")
                
                if step >= self.max_steps:
                    break
            
            # End of epoch
            epoch += 1
            avg_epoch_loss = epoch_loss / len(batches) if batches else 0
            logger.info(f"Epoch {epoch} completed with average loss: {avg_epoch_loss:.4f}")
        
        # Save final model
        final_path = os.path.join(self.model.config.save_dir, "final")
        self.model.save(final_path)
        logger.info(f"Training completed. Final model saved to {final_path}")
        
        return {
            "steps": step,
            "epochs": epoch,
            "final_loss": avg_epoch_loss,
            "best_loss": self.best_loss,
            "best_step": self.best_step
        }
    
    def evaluate(self):
        """Evaluate the model"""
        if not self.eval_data_path:
            return None
        
        # Load evaluation data
        eval_data = self._load_data(self.eval_data_path)
        
        # Evaluation loop
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
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
                
                # Forward pass
                loss, _, _ = self.model(input_chars=char_ids, target_concepts=char_ids)
                
                # Track loss
                if loss is not None:
                    total_loss += loss.item() * len(batch)
                    total_samples += len(batch)
        
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        logger.info(f"Evaluation loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def _load_data(self, data_path):
        """Load training or evaluation data"""
        # This is a simplified implementation
        # In a real system, this would handle various data formats
        
        if data_path.endswith(".json"):
            # Load JSON data
            try:
                with open(data_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Handle different JSON formats
                if isinstance(data, list):
                    # List of examples
                    if len(data) > 0 and isinstance(data[0], dict):
                        # Convert to uniform format
                        samples = []
                        for item in data:
                            if "text" in item:
                                samples.append({"text": item["text"]})
                            elif "content" in item:
                                samples.append({"text": item["content"]})
                            elif "instruction" in item and "output" in item:
                                # Instruction/output format
                                samples.append({
                                    "text": f"{item['instruction']}\n\n{item.get('input', '')}\n\n{item['output']}"
                                })
                            elif "prompt" in item and "response" in item:
                                # Prompt/response format
                                samples.append({
                                    "text": f"{item['prompt']}\n\n{item['response']}"
                                })
                        return samples
                    else:
                        # Simple list of strings
                        return [{"text": str(item)} for item in data]
                else:
                    # Single JSON object
                    return [{"text": json.dumps(data)}]
            except Exception as e:
                logger.error(f"Error loading JSON data: {e}")
                return []
        elif data_path.endswith(".txt"):
            # Load text data
            try:
                with open(data_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    data = [{"text": line.strip()} for line in lines if line.strip()]
                return data
            except Exception as e:
                logger.error(f"Error loading text data: {e}")
                return []
        elif data_path.endswith(".csv"):
            # Try basic CSV handling
            try:
                import csv
                data = []
                with open(data_path, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    header = next(reader)
                    text_col = 0
                    for i, col in enumerate(header):
                        if col.lower() in ["text", "content", "prompt", "data"]:
                            text_col = i
                            break
                    
                    for row in reader:
                        if len(row) > text_col:
                            data.append({"text": row[text_col]})
                return data
            except Exception as e:
                logger.error(f"Error loading CSV data: {e}")
                return []
        else:
            logger.error(f"Unsupported data format: {data_path}")
            return []
    
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
    print("Special commands: 'save', 'dream', 'stats', 'evolve'")
    
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
                print("\nSAM: Current stats:")
                print(f"  Hidden dimension: {model.layers[0].hidden_dim}")
                print(f"  Number of layers: {len(model.layers)}")
                print(f"  Total concepts: {concept_stats['total_concepts']}")
                print(f"  Character concepts: {concept_stats['character_concepts']}")
                print(f"  Semantic concepts: {concept_stats['semantic_concepts']}")
                print(f"  Merged concepts: {concept_stats['merged_concepts']}")
                print(f"  Global step: {model.global_step}")
                continue
            elif user_input.lower() == 'evolve':
                print("\nSAM: Evolving...")
                results = model.evolve()
                width = model.layers[0].hidden_dim
                depth = len(model.layers)
                print(f"\nSAM: Evolution complete. New dimensions: width={width}, depth={depth}")
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
