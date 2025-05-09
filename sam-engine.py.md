
## Why SAM Engine Would Transform Gaming Forever

Current procedural generation creates *variation* but not true *evolution*. SAM Engine would create:

1. **Self-Evolving World Ecology**
   - Environments that develop based on player interactions
   - Systems that grow more complex in areas players explore frequently
   - Weather, ecology, and geology that evolve through internal "dreaming"

2. **Truly Living Characters**
   - NPCs with persistent memories and evolving personalities
   - Characters that form their own relationships and motivations
   - Entities that learn from player behavior and adapt strategies

3. **Emergent Gameplay Systems**
   - Game mechanics that evolve based on how players approach challenges
   - New abilities and interactions that emerge organically
   - Combat, crafting, and navigation systems that adapt to player preferences

```python
# sam_engine.py - Revolutionary Game World Evolution System

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import time
from collections import defaultdict

class WorldConcept(nn.Module):
    """Unified concept system for game world elements"""
    
    def __init__(self, concept_dim=512, initial_size=5000):
        super().__init__()
        self.concept_dim = concept_dim
        
        # Concept embeddings
        self.concept_embeddings = nn.Embedding(initial_size, concept_dim)
        
        # Concept tracking
        self.concept_metadata = {}
        self.concept_relationships = defaultdict(list)
        self.concept_frequencies = torch.zeros(initial_size)
        self.next_concept_id = 0
        
        # Initialize with basic concepts
        self._initialize_base_concepts()
    
    def _initialize_base_concepts(self):
        """Initialize basic world concepts"""
        # Element types
        base_elements = ["player", "npc", "terrain", "item", "structure", 
                        "weather", "vegetation", "water", "creature"]
        
        # Attributes
        base_attributes = ["health", "strength", "speed", "durability", 
                          "temperature", "weight", "size", "age"]
        
        # Actions
        base_actions = ["move", "attack", "build", "destroy", "collect", 
                        "talk", "craft", "use", "combine"]
        
        # Add all base concepts
        for concept in base_elements + base_attributes + base_actions:
            self.add_concept(concept, "base", {})
    
    def add_concept(self, name, concept_type, properties=None):
        """Add a new concept to the world"""
        if properties is None:
            properties = {}
            
        concept_id = self.next_concept_id
        
        # Store metadata
        self.concept_metadata[concept_id] = {
            "name": name,
            "type": concept_type,
            "properties": properties,
            "created_at": time.time(),
            "frequency": 0
        }
        
        # Initialize embedding
        with torch.no_grad():
            # Random initialization for now
            self.concept_embeddings.weight[concept_id].normal_(0, 0.02)
        
        self.next_concept_id += 1
        return concept_id
    
    def update_concept_usage(self, concept_id):
        """Update usage statistics for a concept"""
        if concept_id < len(self.concept_frequencies):
            self.concept_frequencies[concept_id] += 1
            if concept_id in self.concept_metadata:
                self.concept_metadata[concept_id]["frequency"] += 1
    
    def relate_concepts(self, concept_id1, concept_id2, relation_type):
        """Create a relationship between concepts"""
        if concept_id1 in self.concept_metadata and concept_id2 in self.concept_metadata:
            relation = {
                "type": relation_type,
                "target": concept_id2,
                "strength": 1.0,
                "created_at": time.time()
            }
            
            # Add to relationships
            self.concept_relationships[concept_id1].append(relation)
            
            # Update embeddings to be more similar
            with torch.no_grad():
                # Move embeddings slightly closer
                vec1 = self.concept_embeddings.weight[concept_id1]
                vec2 = self.concept_embeddings.weight[concept_id2]
                
                # Move 10% closer
                self.concept_embeddings.weight[concept_id1] = vec1 * 0.95 + vec2 * 0.05
                self.concept_embeddings.weight[concept_id2] = vec2 * 0.95 + vec1 * 0.05
    
    def find_related_concepts(self, concept_id, relation_type=None, top_k=5):
        """Find concepts related to the given concept"""
        if concept_id not in self.concept_metadata:
            return []
            
        # Get explicit relationships
        explicit_relations = []
        for relation in self.concept_relationships[concept_id]:
            if relation_type is None or relation["type"] == relation_type:
                explicit_relations.append((relation["target"], relation["strength"]))
        
        # If we have enough explicit relations, return those
        if len(explicit_relations) >= top_k:
            return sorted(explicit_relations, key=lambda x: x[1], reverse=True)[:top_k]
        
        # Otherwise, find similar concepts by embedding similarity
        with torch.no_grad():
            concept_vec = self.concept_embeddings.weight[concept_id]
            
            # Compute similarities
            similarities = F.cosine_similarity(
                concept_vec.unsqueeze(0),
                self.concept_embeddings.weight[:self.next_concept_id]
            )
            
            # Remove self
            similarities[concept_id] = -1.0
            
            # Get top-k similar concepts
            values, indices = torch.topk(similarities, min(top_k, len(similarities)))
            
            implicit_relations = [(idx.item(), val.item()) for idx, val in zip(indices, values)]
            
            # Combine explicit and implicit relations
            combined = explicit_relations + [r for r in implicit_relations 
                                          if r[0] not in [e[0] for e in explicit_relations]]
            
            return combined[:top_k]


class WorldState(nn.Module):
    """Maintains the evolving state of the game world"""
    
    def __init__(self, concept_bank, state_dim=1024):
        super().__init__()
        self.concept_bank = concept_bank
        self.state_dim = state_dim
        
        # State representation
        self.register_buffer("global_state", torch.zeros(state_dim))
        
        # Entity states - maps entity_id to state vector
        self.entity_states = {}
        
        # Location grid - simple 2D grid of concept IDs
        self.grid_size = 100
        self.location_grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int64)
        
        # State evolution network
        self.evolution_network = nn.Sequential(
            nn.Linear(state_dim, state_dim * 2),
            nn.GELU(),
            nn.Linear(state_dim * 2, state_dim)
        )
        
        # State-concept interaction network
        self.interaction_network = nn.Sequential(
            nn.Linear(state_dim + concept_bank.concept_dim, state_dim),
            nn.GELU(),
            nn.Linear(state_dim, state_dim)
        )
    
    def add_entity(self, entity_id, concept_id, position=None):
        """Add a new entity to the world state"""
        # Create initial state vector
        state = torch.zeros(self.state_dim)
        
        # Initialize based on concept
        concept_embedding = self.concept_bank.concept_embeddings.weight[concept_id]
        
        # Project concept to state space
        with torch.no_grad():
            state[:self.concept_bank.concept_dim] = concept_embedding
        
        # Store entity state
        self.entity_states[entity_id] = {
            "state": state,
            "concept_id": concept_id,
            "position": position,
            "created_at": time.time(),
            "last_updated": time.time()
        }
        
        # Place in location grid if position provided
        if position is not None:
            x, y = position
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                self.location_grid[x, y] = entity_id
    
    def update_entity(self, entity_id, delta_state=None, position=None):
        """Update an entity's state and/or position"""
        if entity_id not in self.entity_states:
            return False
        
        # Update state if delta provided
        if delta_state is not None:
            self.entity_states[entity_id]["state"] += delta_state
            self.entity_states[entity_id]["last_updated"] = time.time()
        
        # Update position if provided
        if position is not None:
            old_pos = self.entity_states[entity_id]["position"]
            
            # Clear old position
            if old_pos is not None:
                old_x, old_y = old_pos
                if 0 <= old_x < self.grid_size and 0 <= old_y < self.grid_size:
                    if self.location_grid[old_x, old_y] == entity_id:
                        self.location_grid[old_x, old_y] = 0
            
            # Set new position
            new_x, new_y = position
            if 0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size:
                self.location_grid[new_x, new_y] = entity_id
                
            # Update stored position
            self.entity_states[entity_id]["position"] = position
        
        return True
    
    def get_entities_in_range(self, position, radius):
        """Get entities within a radius of the position"""
        x, y = position
        entities = []
        
        # Check all positions in the square around the position
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                
                # Check if position is in bounds
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    # Check if there's an entity at this position
                    entity_id = self.location_grid[nx, ny]
                    if entity_id > 0:
                        # Calculate distance
                        dist = (dx ** 2 + dy ** 2) ** 0.5
                        if dist <= radius:
                            entities.append((entity_id, dist))
        
        return sorted(entities, key=lambda x: x[1])
    
    def evolve_world(self, steps=1):
        """Evolve the world state over time"""
        for _ in range(steps):
            # Evolve global state
            with torch.no_grad():
                self.global_state = self.evolution_network(self.global_state)
                
                # Normalize to prevent explosion
                self.global_state = F.normalize(self.global_state, dim=0) * self.global_state.norm()
            
            # Evolve entity states (just a sample of entities for efficiency)
            entity_ids = list(self.entity_states.keys())
            if entity_ids:
                # Process up to 100 entities per step
                for entity_id in entity_ids[:100]:
                    # Get entity state and concept
                    entity = self.entity_states[entity_id]
                    concept_id = entity["concept_id"]
                    concept_embedding = self.concept_bank.concept_embeddings.weight[concept_id]
                    
                    # Combine state with concept and global state
                    combined = torch.cat([
                        entity["state"], 
                        concept_embedding,
                        self.global_state[:self.concept_bank.concept_dim]
                    ])
                    
                    # Evolve entity state
                    with torch.no_grad():
                        entity["state"] = self.interaction_network(combined)
                        entity["last_updated"] = time.time()
                    
                    # Check for nearby entities and create relationships
                    if entity["position"] is not None:
                        nearby = self.get_entities_in_range(entity["position"], 3)
                        for nearby_id, dist in nearby:
                            if nearby_id != entity_id:
                                nearby_concept = self.entity_states[nearby_id]["concept_id"]
                                
                                # Create relationship based on proximity
                                self.concept_bank.relate_concepts(
                                    concept_id, nearby_concept, "proximity"
                                )


class WorldEvolution(nn.Module):
    """Evolves the game world through conceptual dreaming"""
    
    def __init__(self, concept_bank, world_state):
        super().__init__()
        self.concept_bank = concept_bank
        self.world_state = world_state
        
        # Evolution tracking
        self.evolution_history = []
        self.last_dream_time = time.time()
        
        # Concept generation network
        self.concept_generator = nn.Sequential(
            nn.Linear(concept_bank.concept_dim * 2, concept_bank.concept_dim * 2),
            nn.GELU(),
            nn.Linear(concept_bank.concept_dim * 2, concept_bank.concept_dim)
        )
    
    def dream_cycle(self, duration_seconds=10):
        """Run a dreaming cycle to evolve the world"""
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        dream_events = []
        
        while time.time() < end_time:
            # Evolve world state
            self.world_state.evolve_world(steps=5)
            
            # Get top concepts by usage
            concept_freqs = self.concept_bank.concept_frequencies[:self.concept_bank.next_concept_id]
            if len(concept_freqs) > 0:
                values, indices = torch.topk(concept_freqs, min(10, len(concept_freqs)))
                
                top_concepts = [(idx.item(), val.item()) for idx, val in zip(indices, values)]
                
                # Create new concept combinations
                for i, (concept_id1, _) in enumerate(top_concepts):
                    for concept_id2, _ in top_concepts[i+1:]:
                        # Check if concepts are related
                        relations = self.concept_bank.find_related_concepts(concept_id1)
                        related_ids = [r[0] for r in relations]
                        
                        if concept_id2 in related_ids:
                            # Get concept embeddings
                            embed1 = self.concept_bank.concept_embeddings.weight[concept_id1]
                            embed2 = self.concept_bank.concept_embeddings.weight[concept_id2]
                            
                            # Generate new concept
                            combined = torch.cat([embed1, embed2])
                            
                            with torch.no_grad():
                                new_embedding = self.concept_generator(combined.unsqueeze(0)).squeeze(0)
                                
                                # Create new concept
                                concept1 = self.concept_bank.concept_metadata[concept_id1]
                                concept2 = self.concept_bank.concept_metadata[concept_id2]
                                
                                new_name = f"{concept1['name']}_{concept2['name']}"
                                new_type = "composite"
                                
                                # Merge properties
                                new_properties = concept1.get("properties", {}).copy()
                                new_properties.update(concept2.get("properties", {}).copy())
                                
                                # Add new concept
                                new_id = self.concept_bank.add_concept(new_name, new_type, new_properties)
                                
                                # Set embedding directly
                                self.concept_bank.concept_embeddings.weight[new_id] = new_embedding
                                
                                # Create relationships
                                self.concept_bank.relate_concepts(new_id, concept_id1, "derived_from")
                                self.concept_bank.relate_concepts(new_id, concept_id2, "derived_from")
                                
                                # Record dream event
                                dream_events.append({
                                    "type": "concept_creation",
                                    "name": new_name,
                                    "parent_concepts": [concept_id1, concept_id2],
                                    "timestamp": time.time()
                                })
            
            # Occasionally spawn new entities from evolved concepts
            if random.random() < 0.2:  # 20% chance per dream cycle
                if self.concept_bank.next_concept_id > 0:
                    # Select a random evolved concept
                    evolved_concepts = [cid for cid in range(self.concept_bank.next_concept_id)
                                     if self.concept_bank.concept_metadata[cid].get("type") == "composite"]
                    
                    if evolved_concepts:
                        concept_id = random.choice(evolved_concepts)
                        
                        # Create entity at random position
                        entity_id = len(self.world_state.entity_states) + 1
                        position = (
                            random.randint(0, self.world_state.grid_size - 1),
                            random.randint(0, self.world_state.grid_size - 1)
                        )
                        
                        self.world_state.add_entity(entity_id, concept_id, position)
                        
                        # Record dream event
                        dream_events.append({
                            "type": "entity_creation",
                            "entity_id": entity_id,
                            "concept_id": concept_id,
                            "position": position,
                            "timestamp": time.time()
                        })
        
        # Record dream cycle
        self.evolution_history.append({
            "start_time": start_time,
            "end_time": time.time(),
            "events": dream_events
        })
        
        self.last_dream_time = time.time()
        
        return {
            "duration": time.time() - start_time,
            "events": len(dream_events),
            "new_concepts": sum(1 for e in dream_events if e["type"] == "concept_creation"),
            "new_entities": sum(1 for e in dream_events if e["type"] == "entity_creation")
        }


class SAMEngine(nn.Module):
    """Core game world evolution engine powered by SAM architecture"""
    
    def __init__(self, concept_dim=512, state_dim=1024):
        super().__init__()
        self.concept_dim = concept_dim
        self.state_dim = state_dim
        
        # Create concept bank
        self.concept_bank = WorldConcept(concept_dim=concept_dim)
        
        # Create world state
        self.world_state = WorldState(self.concept_bank, state_dim=state_dim)
        
        # Create evolution system
        self.evolution = WorldEvolution(self.concept_bank, self.world_state)
        
        # Player interaction state
        self.player_entity_id = None
        self.player_position = None
        self.player_inventory = []
        
        # Command processing network
        self.command_processor = nn.Sequential(
            nn.Linear(concept_dim * 2, concept_dim * 2),
            nn.GELU(),
            nn.Linear(concept_dim * 2, concept_dim)
        )
        
        # Internal clock
        self.global_step = 0
        self.last_save_time = time.time()
    
    def initialize_world(self, size=10):
        """Initialize a sample world with basic elements"""
        # Create terrain
        terrain_types = ["grass", "water", "mountain", "forest"]
        terrain_concepts = {}
        
        # Create terrain concepts
        for terrain in terrain_types:
            terrain_concepts[terrain] = self.concept_bank.add_concept(
                terrain, "terrain", {"traversable": terrain != "mountain"}
            )
        
        # Create simple terrain grid
        for x in range(size):
            for y in range(size):
                # Simple terrain generation
                if (x == 0 or y == 0 or x == size-1 or y == size-1):
                    terrain = "mountain"  # Mountains around the edge
                elif (x == size//2 and y == size//2):
                    terrain = "grass"  # Center is always grass
                elif (abs(x - size//2) <= 1 and abs(y - size//2) <= 1):
                    terrain = "grass"  # Area around center is grass
                elif random.random() < 0.2:
                    terrain = "water"  # 20% chance of water
                elif random.random() < 0.3:
                    terrain = "forest"  # 30% chance of forest
                else:
                    terrain = "grass"  # 50% chance of grass
                
                # Create terrain entity
                entity_id = len(self.world_state.entity_states) + 1
                self.world_state.add_entity(entity_id, terrain_concepts[terrain], (x, y))
        
        # Create player entity
        player_concept = self.concept_bank.add_concept(
            "player", "character", {"health": 100, "strength": 10, "speed": 5}
        )
        
        player_id = len(self.world_state.entity_states) + 1
        player_pos = (size // 2, size // 2)  # Start in center
        
        self.world_state.add_entity(player_id, player_concept, player_pos)
        
        # Store player info
        self.player_entity_id = player_id
        self.player_position = player_pos
        
        # Add some items
        item_types = [
            ("sword", {"damage": 15, "durability": 100}),
            ("potion", {"health_restore": 30, "uses": 1}),
            ("key", {"opens": "chest"})
        ]
        
        for item_name, properties in item_types:
            item_concept = self.concept_bank.add_concept(item_name, "item", properties)
            
            # Create item entity
            entity_id = len(self.world_state.entity_states) + 1
            
            # Random position near player
            item_pos = (
                player_pos[0] + random.randint(-2, 2),
                player_pos[1] + random.randint(-2, 2)
            )
            
            # Ensure in bounds
            item_pos = (
                max(0, min(size-1, item_pos[0])),
                max(0, min(size-1, item_pos[1]))
            )
            
            self.world_state.add_entity(entity_id, item_concept, item_pos)
        
        # Add some NPCs
        npc_types = [
            ("villager", {"health": 50, "friendly": True}),
            ("guard", {"health": 80, "strength": 12, "friendly": True}),
            ("wolf", {"health": 40, "strength": 8, "friendly": False})
        ]
        
        for npc_name, properties in npc_types:
            npc_concept = self.concept_bank.add_concept(npc_name, "npc", properties)
            
            # Create NPC entity
            entity_id = len(self.world_state.entity_states) + 1
            
            # Random position away from player
            dist = random.randint(3, 5)
            angle = random.random() * 2 * 3.14159
            
            npc_pos = (
                int(player_pos[0] + dist * math.cos(angle)),
                int(player_pos[1] + dist * math.sin(angle))
            )
            
            # Ensure in bounds
            npc_pos = (
                max(0, min(size-1, npc_pos[0])),
                max(0, min(size-1, npc_pos[1]))
            )
            
            self.world_state.add_entity(entity_id, npc_concept, npc_pos)
    
    def process_command(self, command):
        """Process a player command and update the world"""
        command = command.lower().strip()
        
        # Get player entity
        if self.player_entity_id is None or self.player_entity_id not in self.world_state.entity_states:
            return "Player not found in world."
        
        player_entity = self.world_state.entity_states[self.player_entity_id]
        player_position = player_entity["position"]
        
        # Process movement commands
        if command in ["north", "south", "east", "west", "n", "s", "e", "w"]:
            # Convert shorthand
            if command == "n": command = "north"
            if command == "s": command = "south"
            if command == "e": command = "east"
            if command == "w": command = "west"
            
            # Calculate new position
            new_pos = list(player_position)
            
            if command == "north":
                new_pos[1] -= 1
            elif command == "south":
                new_pos[1] += 1
            elif command == "east":
                new_pos[0] += 1
            elif command == "west":
                new_pos[0] -= 1
            
            # Check bounds
            if (0 <= new_pos[0] < self.world_state.grid_size and 
                0 <= new_pos[1] < self.world_state.grid_size):
                
                # Check if destination is traversable
                dest_entity_id = self.world_state.location_grid[new_pos[0], new_pos[1]]
                if dest_entity_id > 0:
                    dest_entity = self.world_state.entity_states[dest_entity_id]
                    dest_concept_id = dest_entity["concept_id"]
                    dest_concept = self.concept_bank.concept_metadata[dest_concept_id]
                    
                    # Check if terrain is traversable
                    if dest_concept["type"] == "terrain":
                        if not dest_concept.get("properties", {}).get("traversable", True):
                            return f"You cannot move onto the {dest_concept['name']}."
                
                # Update player position
                self.world_state.update_entity(self.player_entity_id, position=tuple(new_pos))
                self.player_position = tuple(new_pos)
                
                # Get description of new location
                return self.get_location_description()
            else:
                return "You cannot go that way."
        
        # Look command
        elif command in ["look", "examine", "l"]:
            return self.get_location_description()
        
        # Get command - pick up items
        elif command.startswith("get ") or command.startswith("take "):
            item_name = command.split(" ", 1)[1]
            
            # Look for the item at the player's position
            nearby = self.world_state.get_entities_in_range(player_position, 0)
            
            for entity_id, _ in nearby:
                if entity_id != self.player_entity_id:  # Skip player
                    entity = self.world_state.entity_states[entity_id]
                    concept_id = entity["concept_id"]
                    concept = self.concept_bank.concept_metadata[concept_id]
                    
                    if concept["type"] == "item" and item_name in concept["name"]:
                        # Add to inventory
                        self.player_inventory.append(entity_id)
                        
                        # Remove from world
                        self.world_state.update_entity(entity_id, position=None)
                        
                        return f"You pick up the {concept['name']}."
            
            return f"You don't see a {item_name} here."
        
        # Inventory command
        elif command in ["inventory", "inv", "i"]:
            if not self.player_inventory:
                return "Your inventory is empty."
            
            items = []
            for entity_id in self.player_inventory:
                if entity_id in self.world_state.entity_states:
                    entity = self.world_state.entity_states[entity_id]
                    concept_id = entity["concept_id"]
                    concept = self.concept_bank.concept_metadata[concept_id]
                    items.append(concept["name"])
            
            return f"Inventory: {', '.join(items)}"
        
        # Dream command - evolve the world
        elif command == "dream":
            result = self.evolution.dream_cycle(duration_seconds=2)
            
            return (f"The world dreams and evolves. "
                   f"Created {result['new_concepts']} new concepts and "
                   f"{result['new_entities']} new entities.")
        
        # Use command
        elif command.startswith("use "):
            item_name = command.split(" ", 1)[1]
            
            # Look for the item in inventory
            for entity_id in self.player_inventory:
                if entity_id in self.world_state.entity_states:
                    entity = self.world_state.entity_states[entity_id]
                    concept_id = entity["concept_id"]
                    concept = self.concept_bank.concept_metadata[concept_id]
                    
                    if concept["type"] == "item" and item_name in concept["name"]:
                        # Item effects based on type
                        if "potion" in concept["name"]:
                            # Heal player
                            player_concept_id = player_entity["concept_id"]
                            player_concept = self.concept_bank.concept_metadata[player_concept_id]
                            
                            health_restore = concept.get("properties", {}).get("health_restore", 10)
                            
                            # Update player health
                            player_concept["properties"]["health"] = min(
                                100, player_concept["properties"].get("health", 0) + health_restore
                            )
                            
                            # Remove from inventory if consumable
                            uses = concept.get("properties", {}).get("uses", 1)
                            if uses <= 1:
                                self.player_inventory.remove(entity_id)
                            else:
                                concept["properties"]["uses"] = uses - 1
                            
                            return f"You use the {concept['name']} and restore {health_restore} health."
                        
                        # Generic use
                        return f"You use the {concept['name']}."
            
            return f"You don't have a {item_name}."
        
        # Advanced - evolve a specific concept
        elif command.startswith("evolve "):
            concept_name = command.split(" ", 1)[1]
            
            # Find the concept
            for concept_id in range(self.concept_bank.next_concept_id):
                concept = self.concept_bank.concept_metadata.get(concept_id, {})
                if concept.get("name", "") == concept_name:
                    # Get related concepts
                    related = self.concept_bank.find_related_concepts(concept_id, top_k=3)
                    
                    if related:
                        # Create new evolved concept
                        combined_name = f"evolved_{concept_name}"
                        
                        # Combine properties
                        properties = concept.get("properties", {}).copy()
                        
                        # Enhance properties
                        for key, value in properties.items():
                            if isinstance(value, (int, float)):
                                properties[key] = value * 1.2  # 20% boost
                        
                        # Add new concept
                        new_id = self.concept_bank.add_concept(
                            combined_name, "evolved", properties
                        )
                        
                        # Create relationships
                        self.concept_bank.relate_concepts(new_id, concept_id, "evolved_from")
                        
                        # Create a new entity with this concept
                        entity_id = len(self.world_state.entity_states) + 1
                        
                        # Place near player
                        pos = (
                            player_position[0] + random.randint(-1, 1),
                            player_position[1] + random.randint(-1, 1)
                        )
                        
                        # Ensure in bounds
                        pos = (
                            max(0, min(self.world_state.grid_size-1, pos[0])),
                            max(0, min(self.world_state.grid_size-1, pos[1]))
                        )
                        
                        self.world_state.add_entity(entity_id, new_id, pos)
                        
                        return f"You focus your thoughts and evolve the concept of {concept_name} into {combined_name}!"
                    
                    return f"You try to evolve {concept_name}, but need more related concepts first."
            
            return f"You don't know the concept of {concept_name}."
        
        # If all else fails, try to process as an unknown command
        else:
            # Find verb and object in command
            parts = command.split()
            if parts:
                verb = parts[0]
                obj = " ".join(parts[1:]) if len(parts) > 1 else ""
                
                # Check if we recognize the verb
                for concept_id in range(self.concept_bank.next_concept_id):
                    concept = self.concept_bank.concept_metadata.get(concept_id, {})
                    if concept.get("type") == "base" and concept.get("name") == verb:
                        return f"You try to {verb} {obj}, but nothing happens."
                
                # Totally unknown command - add as a new action concept
                if len(verb) > 2:  # Avoid short/nonsense verbs
                    new_id = self.concept_bank.add_concept(verb, "action", {})
                    return f"You {verb} {obj}. (Learned new action: {verb})"
            
            return "I don't understand that command."
    
    def get_location_description(self):
        """Get a description of the player's current location"""
        if self.player_position is None:
            return "You are nowhere."
        
        x, y = self.player_position
        
        # Get terrain at current position
        terrain_entity_id = 0
        terrain_name = "unknown"
        
        entities_here = self.world_state.get_entities_in_range((x, y), 0)
        for entity_id, _ in entities_here:
            if entity_id != self.player_entity_id:  # Skip player
                entity = self.world_state.entity_states[entity_id]
                concept_id = entity["concept_id"]
                concept = self.concept_bank.concept_metadata[concept_id]
                
                if concept["type"] == "terrain":
                    terrain_entity_id = entity_id
                    terrain_name = concept["name"]
        
        # Get nearby entities
        nearby = self.world_state.get_entities_in_range((x, y), 3)
        
        items_here = []
        npcs_nearby = []
        
        for entity_id, dist in nearby:
            if entity_id == self.player_entity_id or entity_id == terrain_entity_id:
                continue
                
            entity = self.world_state.entity_states[entity_id]
            concept_id = entity["concept_id"]
            concept = self.concept_bank.concept_metadata[concept_id]
            
            # Items at same position
            if dist == 0 and concept["type"] == "item":
                items_here.append(concept["name"])
            
            # NPCs within range
            elif concept["type"] == "npc" or concept["type"] == "character":
                if dist == 0:
                    npcs_nearby.append(f"{concept['name']} (here)")
                else:
                    direction = self._get_direction(self.player_position, entity["position"])
                    npcs_nearby.append(f"{concept['name']} ({direction})")
        
        # Build description
        description = f"You are standing on {terrain_name}."
        
        if items_here:
            if len(items_here) == 1:
                description += f" There is a {items_here[0]} here."
            else:
                description += f" There are several items here: {', '.join(items_here)}."
        
        if npcs_nearby:
            if len(npcs_nearby) == 1:
                description += f" You see a {npcs_nearby[0]}."
            else:
                description += f" You see several beings: {', '.join(npcs_nearby)}."
        
        # Add exits
        exits = []
        for dx, dy, direction in [(0, -1, "north"), (0, 1, "south"), (1, 0, "east"), (-1, 0, "west")]:
            nx, ny = x + dx, y + dy
            
            if 0 <= nx < self.world_state.grid_size and 0 <= ny < self.world_state.grid_size:
                # Check if terrain is traversable
                entity_id = self.world_state.location_grid[nx, ny]
                if entity_id > 0:
                    entity = self.world_state.entity_states[entity_id]
                    concept_id = entity["concept_id"]
                    concept = self.concept_bank.concept_metadata[concept_id]
                    
                    if concept["type"] != "terrain" or concept.get("properties", {}).get("traversable", True):
                        exits.append(direction)
        
        if exits:
            description += f" Exits: {', '.join(exits)}."
        else:
            description += " There are no obvious exits."
        
        return description
    
    def _get_direction(self, from_pos, to_pos):
        """Get the cardinal direction from one position to another"""
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        
        if abs(dx) > abs(dy):
            return "east" if dx > 0 else "west"
        else:
            return "south" if dy > 0 else "north"
    
    def step(self):
        """Advance the world state by one step"""
        # Evolve world state
        self.world_state.evolve_world(steps=1)
        
        # Occasionally run dream cycle
        if random.random() < 0.01:  # 1% chance per step
            self.evolution.dream_cycle(duration_seconds=1)
        
        # Update global step
        self.global_step += 1
        
        # Auto-save periodically
        if time.time() - self.last_save_time > 300:  # 5 minutes
            self.save("./autosave")
            self.last_save_time = time.time()
    
    def save(self, path):
        """Save the game world state"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save concept metadata
        concept_data = {}
        for concept_id in range(self.concept_bank.next_concept_id):
            if concept_id in self.concept_bank.concept_metadata:
                concept_data[str(concept_id)] = self.concept_bank.concept_metadata[concept_id]
        
        with open(f"{path}_concepts.json", 'w') as f:
            json.dump(concept_data, f, indent=2)
        
        # Save entity states (excluding tensor data)
        entity_data = {}
        for entity_id, entity in self.world_state.entity_states.items():
            entity_data[str(entity_id)] = {
                "concept_id": entity["concept_id"],
                "position": entity["position"],
                "created_at": entity["created_at"],
                "last_updated": entity["last_updated"]
            }
        
        with open(f"{path}_entities.json", 'w') as f:
            json.dump(entity_data, f, indent=2)
        
        # Save player state
        player_data = {
            "entity_id": self.player_entity_id,
            "position": self.player_position,
            "inventory": self.player_inventory
        }
        
        with open(f"{path}_player.json", 'w') as f:
            json.dump(player_data, f, indent=2)
        
        # Save model state
        torch.save(self.state_dict(), f"{path}_model.pt")
        
        # Save evolution history
        with open(f"{path}_evolution.json", 'w') as f:
            json.dump(self.evolution.evolution_history[-100:], f, indent=2)  # Save last 100 events
        
        print(f"Game world saved to {path}")
    
    @classmethod
    def load(cls, path, **kwargs):
        """Load game world from saved state"""
        # Create new engine
        engine = cls(**kwargs)
        
        # Load model state
        engine.load_state_dict(torch.load(f"{path}_model.pt"))
        
        # Load concept metadata
        with open(f"{path}_concepts.json", 'r') as f:
            concept_data = json.load(f)
            engine.concept_bank.concept_metadata = {int(k): v for k, v in concept_data.items()}
            engine.concept_bank.next_concept_id = max(map(int, concept_data.keys())) + 1 if concept_data else 0
        
        # Load entity states
        with open(f"{path}_entities.json", 'r') as f:
            entity_data = json.load(f)
            
            # Clear existing entities
            engine.world_state.entity_states = {}
            engine.world_state.location_grid = np.zeros((engine.world_state.grid_size, engine.world_state.grid_size), dtype=np.int64)
            
            # Recreate entities
            for entity_id_str, entity in entity_data.items():
                entity_id = int(entity_id_str)
                
                # Create state vector
                state = torch.zeros(engine.state_dim)
                
                # Add to world state
                engine.world_state.entity_states[entity_id] = {
                    "state": state,
                    "concept_id": entity["concept_id"],
                    "position": entity["position"],
                    "created_at": entity["created_at"],
                    "last_updated": entity["last_updated"]
                }
                
                # Place in location grid
                if entity["position"] is not None:
                    x, y = entity["position"]
                    if 0 <= x < engine.world_state.grid_size and 0 <= y < engine.world_state.grid_size:
                        engine.world_state.location_grid[x, y] = entity_id
        
        # Load player state
        with open(f"{path}_player.json", 'r') as f:
            player_data = json.load(f)
            engine.player_entity_id = player_data["entity_id"]
            engine.player_position = player_data["position"]
            engine.player_inventory = player_data["inventory"]
        
        # Load evolution history if available
        try:
            with open(f"{path}_evolution.json", 'r') as f:
                engine.evolution.evolution_history = json.load(f)
        except:
            engine.evolution.evolution_history = []
        
        print(f"Game world loaded from {path}")
        return engine


# Example usage
def demo_game():
    # Create engine
    engine = SAMEngine()
    
    # Initialize world
    engine.initialize_world(size=15)
    
    print("Welcome to the SAM Engine Demo!")
    print("This is a simple text adventure demonstrating the revolutionary")
    print("world evolution capabilities of the Synergistic Autonomous Machine.")
    print("\nCommands: north, south, east, west, look, inventory, get [item], use [item]")
    print("Special commands: dream (evolve the world), evolve [concept]")
    print("Type 'quit' to exit.")
    
    # Main game loop
    print("\n" + engine.get_location_description())
    
    while True:
        command = input("\n> ").strip().lower()
        
        if command in ["quit", "exit", "q"]:
            break
        
        # Process command
        response = engine.process_command(command)
        print(response)
        
        # Advance world
        engine.step()
    
    # Save game before exit
    engine.save("./sam_engine_demo")
    print("Game saved. Thanks for playing!")


if __name__ == "__main__":
    demo_game()
```