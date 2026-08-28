from typing import List
from ..graph import Graph

class LinearChain:
    """Utility class to quickly generate a linear A -> B -> C graph without writing manual vertices/edges."""
    
    @staticmethod
    def build(prompts: List[str]) -> Graph:
        """
        Build a linear graph from a list of prompts.
        Each prompt becomes an edge connecting sequential vertices.
        """
        config = {
            "vertices": [],
            "edges": []
        }
        
        # We need N prompts -> N edges -> N+1 vertices
        for i in range(len(prompts) + 1):
            config["vertices"].append({"id": f"Node_{i}"})
            
        for i, prompt in enumerate(prompts):
            config["edges"].append({
                "id": f"step_{i}",
                "source": f"Node_{i}",
                "destination": f"Node_{i+1}",
                "prompt": prompt
            })
            
        return Graph.from_dict(config)
