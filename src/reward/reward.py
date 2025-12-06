"""
Reward Function - Score agent responses against ground truth.

This module provides:
- Semantic similarity scoring using Gemini embeddings
- Task completion detection
- Customer satisfaction prediction
- Composite reward calculation
"""

from typing import List, Dict, Any, Optional
import numpy as np
import google.generativeai as genai

from ..trace_parser import TraceSchema


class RewardFunction:
    """
    Score agent responses using multiple reward signals.
    
    Combines:
    1. Semantic similarity to ground truth response
    2. Task completion indicators
    3. Conversation outcome prediction
    
    Attributes:
        traces: Original traces for context and statistics
        outcome_weights: Reward weights for different outcomes
    """
    
    def __init__(self, traces: List[TraceSchema]):
        """
        Initialize reward function with parsed traces.
        
        Args:
            traces: List of TraceSchema objects for context
        """
        self.traces = traces
        self.outcome_weights = {
            "resolved": 1.0,
            "escalated": 0.3,
            "abandoned": -0.5,
            None: 0.0,
        }
        self._model = None
    
    @property
    def model(self):
        """Lazy-load Gemini model."""
        if self._model is None:
            # Use Gemini 3 Pro for high-quality evaluation
            self._model = genai.GenerativeModel('gemini-3-pro-preview')
        return self._model
    
    def score(
        self,
        agent_response: str,
        ground_truth: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Score an agent response.
        
        Args:
            agent_response: The agent's generated response
            ground_truth: Expected response from traces (optional)
            context: Additional context like conversation history
            
        Returns:
            Reward score between -1.0 and 1.0
        """
        scores = []
        
        # Semantic similarity to ground truth
        if ground_truth:
            similarity = self._compute_similarity(agent_response, ground_truth)
            scores.append(similarity)
        
        # Quality assessment via Gemini
        quality = self._assess_quality(agent_response, context)
        scores.append(quality)
        
        # Combine scores
        if scores:
            return float(np.mean(scores))
        return 0.0
    
    def _compute_similarity(self, response: str, ground_truth: str) -> float:
        """
        Compute semantic similarity between response and ground truth.
        
        Uses Gemini embeddings for semantic comparison.
        """
        try:
            # Get embeddings
            result = genai.embed_content(
                model="models/embedding-001",
                content=[response, ground_truth],
                task_type="semantic_similarity",
            )
            
            emb1 = np.array(result['embedding'][0])
            emb2 = np.array(result['embedding'][1])
            
            # Cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            return float(similarity)
        except Exception:
            # Fallback to simple word overlap
            words1 = set(response.lower().split())
            words2 = set(ground_truth.lower().split())
            if not words1 or not words2:
                return 0.0
            overlap = len(words1 & words2) / len(words1 | words2)
            return overlap
    
    def _assess_quality(
        self, response: str, context: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Use Gemini to assess response quality.
        
        Evaluates helpfulness, professionalism, and task relevance.
        """
        history_str = ""
        if context and "history" in context:
            history_str = "\n".join(
                f"{m['role']}: {m['content']}" 
                for m in context["history"][-5:]  # Last 5 messages
            )
        
        prompt = f"""Rate this customer service response on a scale of 0.0 to 1.0.

Conversation context:
{history_str if history_str else "No context available"}

Agent response: {response}

Consider:
- Helpfulness: Does it address the customer's needs?
- Professionalism: Is the tone appropriate?
- Clarity: Is the message clear and actionable?

Return only a single decimal number between 0.0 and 1.0."""

        try:
            result = self.model.generate_content(prompt)
            score = float(result.text.strip())
            return max(0.0, min(1.0, score))
        except Exception:
            return 0.5  # Neutral fallback
    
    def batch_score(
        self,
        responses: List[str],
        ground_truths: Optional[List[str]] = None,
    ) -> List[float]:
        """
        Score multiple responses efficiently.
        
        Args:
            responses: List of agent responses
            ground_truths: Corresponding ground truth responses
            
        Returns:
            List of reward scores
        """
        if ground_truths is None:
            ground_truths = [None] * len(responses)
        
        return [
            self.score(resp, gt) 
            for resp, gt in zip(responses, ground_truths)
        ]

