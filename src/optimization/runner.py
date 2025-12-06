"""
Optimization Runner - Manages prompt optimization with progress tracking.

This module provides:
- OptimizationRunner: Main class for running optimization
- Progress callbacks for frontend integration
- Structured events for real-time updates
"""

import os
import time
import logging
from typing import Optional, Callable, List, Dict, Any, Literal
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum

import dspy
from dspy.teleprompt import BootstrapFewShot

from ..data.sgd_loader import SGDLoader
from ..agent.sgd_agent import (
    SGDAgentModule,
    build_tool_catalog,
    extract_training_examples,
    create_metric,
    compute_response_similarity,
    compute_tool_accuracy,
)


# Configure module logger
logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Types of optimization events."""
    START = "start"
    PROGRESS = "progress"
    CANDIDATE = "candidate"
    EVALUATION = "evaluation"
    COMPLETE = "complete"
    ERROR = "error"
    LOG = "log"


@dataclass
class OptimizationEvent:
    """
    Event emitted during optimization for frontend tracking.
    
    Attributes:
        type: Event type (start, progress, candidate, complete, error, log)
        timestamp: When the event occurred
        data: Event-specific data
    """
    type: EventType
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Convert string type to EventType if needed."""
        if isinstance(self.type, str):
            self.type = EventType(self.type)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "type": self.type.value if isinstance(self.type, EventType) else self.type,
            "timestamp": self.timestamp.isoformat(),
            **self.data,
        }


@dataclass
class OptimizationResult:
    """
    Result of an optimization run.
    
    Attributes:
        success: Whether optimization completed successfully
        optimized_agent: The optimized agent module
        baseline_score: Score before optimization
        optimized_score: Score after optimization
        improvement: Absolute improvement
        improvement_pct: Percentage improvement
        training_time: Time taken in seconds
        num_examples: Number of training examples used
        strategy: Optimization strategy used
        events: List of events during optimization
        initial_prompt: The starting prompt before optimization
        optimized_prompt: The final prompt after optimization
        few_shot_demos: Few-shot examples added by optimizer
    """
    success: bool
    optimized_agent: Optional[SGDAgentModule] = None
    baseline_score: float = 0.0
    optimized_score: float = 0.0
    improvement: float = 0.0
    improvement_pct: float = 0.0
    training_time: float = 0.0
    num_examples: int = 0
    strategy: str = ""
    events: List[OptimizationEvent] = field(default_factory=list)
    error_message: Optional[str] = None
    initial_prompt: str = ""
    optimized_prompt: str = ""
    few_shot_demos: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self, include_full_data: bool = False) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.
        
        Args:
            include_full_data: If True, include full prompts and events (for saving).
                              If False, include only summaries (for frontend updates).
        """
        result = {
            "success": self.success,
            "baseline_score": self.baseline_score,
            "optimized_score": self.optimized_score,
            "improvement": self.improvement,
            "improvement_pct": self.improvement_pct,
            "training_time": self.training_time,
            "num_examples": self.num_examples,
            "strategy": self.strategy,
            "error_message": self.error_message,
            "initial_prompt_length": len(self.initial_prompt),
            "optimized_prompt_length": len(self.optimized_prompt),
            "num_few_shot_demos": len(self.few_shot_demos),
        }
        
        if include_full_data:
            result["initial_prompt"] = self.initial_prompt
            result["optimized_prompt"] = self.optimized_prompt
            result["few_shot_demos"] = self.few_shot_demos
            result["events"] = [e.to_dict() for e in self.events]
            result["prompt_changed"] = self.initial_prompt != self.optimized_prompt
        
        return result
    
    def save(self, path: str) -> None:
        """
        Save the full optimization result to a JSON file.
        
        Args:
            path: Path to save the result (e.g., 'checkpoints/result.json')
        """
        import json
        from pathlib import Path as P
        
        P(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(include_full_data=True), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "OptimizationResult":
        """
        Load an optimization result from a JSON file.
        
        Args:
            path: Path to the saved result
            
        Returns:
            OptimizationResult (without the optimized_agent - load that separately)
        """
        import json
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Reconstruct events
        events = []
        for e in data.get("events", []):
            # Extract type and timestamp, rest is data
            e_copy = dict(e)
            event_type = e_copy.pop("type", "log")
            timestamp = e_copy.pop("timestamp", "")
            events.append(OptimizationEvent(
                type=event_type,
                data=e_copy,  # Remaining fields are the data
                timestamp=timestamp,
            ))
        
        return cls(
            success=data["success"],
            baseline_score=data.get("baseline_score", 0.0),
            optimized_score=data.get("optimized_score", 0.0),
            improvement=data.get("improvement", 0.0),
            improvement_pct=data.get("improvement_pct", 0.0),
            training_time=data.get("training_time", 0.0),
            num_examples=data.get("num_examples", 0),
            strategy=data.get("strategy", ""),
            events=events,
            error_message=data.get("error_message"),
            initial_prompt=data.get("initial_prompt", ""),
            optimized_prompt=data.get("optimized_prompt", ""),
            few_shot_demos=data.get("few_shot_demos", []),
        )


class OptimizationRunner:
    """
    Manages prompt optimization with progress tracking.
    
    This class provides:
    - Easy setup for optimization
    - Progress callbacks for frontend integration
    - Structured events for real-time updates
    - Support for BootstrapFewShot and MIPRO strategies
    
    Example:
        def on_progress(event):
            print(f"{event.type}: {event.data}")
        
        runner = OptimizationRunner(
            data_path="data/raw/schema_guided_dialogue",
            strategy="bootstrap",
            callback=on_progress,
        )
        result = runner.run()
        print(f"Improvement: {result.improvement_pct:.1f}%")
    """
    
    def __init__(
        self,
        data_path: str,
        strategy: Literal["bootstrap", "mipro"] = "bootstrap",
        num_train: int = 100,
        num_eval: int = 20,
        max_demos: int = 4,
        num_candidates: int = 10,
        model: str = "gemini-3-pro-preview",
        callback: Optional[Callable[[OptimizationEvent], None]] = None,
        output_path: Optional[str] = None,
    ):
        """
        Initialize the optimization runner.
        
        Args:
            data_path: Path to SGD dataset
            strategy: "bootstrap" (fast) or "mipro" (Gemini rewrites prompts)
            num_train: Number of training dialogues
            num_eval: Number of evaluation examples
            max_demos: Max few-shot demos for bootstrap
            num_candidates: Number of candidates for MIPRO
            model: Gemini model to use
            callback: Function called with progress events
            output_path: Path to save optimized agent
        """
        self.data_path = Path(data_path)
        self.strategy = strategy
        self.num_train = num_train
        self.num_eval = num_eval
        self.max_demos = max_demos
        self.num_candidates = num_candidates
        self.model = model
        self.callback = callback
        self.output_path = output_path
        
        # State
        self.events: List[OptimizationEvent] = []
        self.current_step = 0
        self.total_steps = 0
        self.baseline_score = 0.0
        self.current_score = 0.0
        self.candidates: List[Dict[str, Any]] = []
        
        # Components (lazy loaded)
        self._loader: Optional[SGDLoader] = None
        self._tool_catalog: Optional[str] = None
        self._trainset: Optional[List] = None
        self._evalset: Optional[List] = None
        self._metric_fn = None
        self._baseline_agent: Optional[SGDAgentModule] = None
    
    def _emit(self, event_type: EventType, **data):
        """Emit an event to the callback."""
        event = OptimizationEvent(type=event_type, data=data)
        self.events.append(event)
        
        # Log
        logger.info(f"[{event_type.value}] {data}")
        
        # Call callback
        if self.callback:
            try:
                self.callback(event)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def _configure_dspy(self):
        """Configure DSPy with Gemini."""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set")
        
        lm = dspy.LM(f"gemini/{self.model}", api_key=api_key)
        dspy.configure(lm=lm)
        
        self._emit(EventType.LOG, message=f"Configured DSPy with {self.model}")
    
    def _load_data(self):
        """Load data and prepare training examples."""
        self._emit(EventType.LOG, message="Loading data...")
        
        self._loader = SGDLoader(self.data_path)
        self._tool_catalog = build_tool_catalog(self._loader, "train")
        
        # Load dialogues
        train_dialogues = self._loader.load_dialogues("train", limit=self.num_train)
        eval_dialogues = self._loader.load_dialogues("dev", limit=self.num_eval)
        
        # Extract examples
        self._trainset = extract_training_examples(train_dialogues)
        self._evalset = extract_training_examples(eval_dialogues)
        
        self._emit(
            EventType.LOG,
            message=f"Loaded {len(train_dialogues)} dialogues, {len(self._trainset)} examples",
            num_dialogues=len(train_dialogues),
            num_examples=len(self._trainset),
            num_services=self._tool_catalog.count("##"),
        )
    
    def _create_metric(self):
        """Create the metric function."""
        self._metric_fn = create_metric()
    
    def _evaluate_agent(
        self,
        agent: SGDAgentModule,
        num_samples: int = None,
    ) -> float:
        """Evaluate an agent on the eval set."""
        evalset = self._evalset[:num_samples] if num_samples else self._evalset
        
        scores = []
        for i, example in enumerate(evalset):
            try:
                pred = agent(
                    user_message=example.user_message,
                    conversation_history=example.conversation_history,
                )
                score = self._metric_fn(example, pred)
                scores.append(score)
                
                # Emit progress
                if i % 5 == 0:
                    self._emit(
                        EventType.EVALUATION,
                        step=i + 1,
                        total=len(evalset),
                        current_score=sum(scores) / len(scores),
                    )
                    
            except Exception as e:
                logger.warning(f"Evaluation error: {e}")
                scores.append(0.0)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _run_bootstrap(self) -> SGDAgentModule:
        """Run BootstrapFewShot optimization."""
        self._emit(
            EventType.LOG,
            message="Starting BootstrapFewShot optimization",
            strategy="bootstrap",
            max_demos=self.max_demos,
        )
        
        # Estimate total steps
        self.total_steps = min(50, len(self._trainset))
        self._emit(EventType.START, total_steps=self.total_steps, strategy="bootstrap")
        
        optimizer = BootstrapFewShot(
            metric=self._metric_fn,
            max_bootstrapped_demos=self.max_demos,
            max_labeled_demos=self.max_demos,
        )
        
        # Use subset for speed
        train_subset = self._trainset[:self.total_steps]
        
        # Track progress via custom metric wrapper
        original_metric = self._metric_fn
        step_count = [0]
        
        def tracking_metric(example, pred, trace=None):
            step_count[0] += 1
            score = original_metric(example, pred, trace)
            
            self.current_step = step_count[0]
            self._emit(
                EventType.PROGRESS,
                step=self.current_step,
                total=self.total_steps,
                metric=score,
            )
            
            return score
        
        optimizer.metric = tracking_metric
        
        optimized = optimizer.compile(self._baseline_agent, trainset=train_subset)
        
        return optimized
    
    def _get_optimized_prompt(self, agent: SGDAgentModule) -> str:
        """Extract the prompt from an optimized agent."""
        try:
            # Try to get from the respond module's signature
            if hasattr(agent, 'respond') and hasattr(agent.respond, 'signature'):
                return agent.respond.signature.__doc__ or ""
            # Fallback to agent's signature
            if hasattr(agent, 'signature'):
                return agent.signature.__doc__ or ""
        except Exception:
            pass
        return ""
    
    def _get_few_shot_demos(self, agent: SGDAgentModule) -> List[Dict[str, Any]]:
        """Extract few-shot demonstrations from an optimized agent."""
        demos = []
        try:
            # DSPy stores demos in the predict module
            if hasattr(agent, 'respond') and hasattr(agent.respond, 'demos'):
                for demo in agent.respond.demos or []:
                    demo_dict = {}
                    if hasattr(demo, 'user_message'):
                        demo_dict['user_message'] = demo.user_message
                    if hasattr(demo, 'response'):
                        demo_dict['response'] = demo.response
                    if hasattr(demo, 'tool_call'):
                        demo_dict['tool_call'] = demo.tool_call
                    if demo_dict:
                        demos.append(demo_dict)
        except Exception:
            pass
        return demos
    
    def _run_mipro(self) -> SGDAgentModule:
        """Run MIPRO optimization (Gemini rewrites prompts)."""
        try:
            from dspy.teleprompt import MIPROv2
        except ImportError:
            self._emit(
                EventType.LOG,
                message="MIPROv2 not available, falling back to BootstrapFewShot",
                level="warning",
            )
            return self._run_bootstrap()
        
        self._emit(
            EventType.LOG,
            message="Starting MIPRO optimization (Gemini will rewrite prompts)",
            strategy="mipro",
            num_candidates=self.num_candidates,
        )
        
        # MIPRO steps: candidates * evaluations
        self.total_steps = self.num_candidates * min(20, len(self._trainset))
        self._emit(EventType.START, total_steps=self.total_steps, strategy="mipro")
        
        # Use training subset
        train_subset = self._trainset[:min(30, len(self._trainset))]
        eval_subset = self._evalset[:min(10, len(self._evalset))] if self._evalset else None
        
        # MIPROv2 with explicit prompt optimization settings
        optimizer = MIPROv2(
            metric=self._metric_fn,
            num_candidates=self.num_candidates,
            init_temperature=1.0,
            verbose=True,
            auto=None,  # Disable auto mode to use explicit settings
        )
        
        self._emit(
            EventType.LOG, 
            message=f"Starting MIPROv2 with {self.num_candidates} trials on {len(train_subset)} examples"
        )
        
        try:
            optimized = optimizer.compile(
                self._baseline_agent,
                trainset=train_subset,
                valset=eval_subset,
                num_trials=self.num_candidates,
                minibatch=True,
                minibatch_size=min(10, len(train_subset)),
                program_aware_proposer=True,  # Enable instruction proposal
                tip_aware_proposer=True,      # Enable tips/hints
            )
        except Exception as e:
            self._emit(
                EventType.LOG,
                message=f"MIPRO error: {e}, falling back to BootstrapFewShot",
                level="warning",
            )
            return self._run_bootstrap()
        
        return optimized
    
    def run(self) -> OptimizationResult:
        """
        Run the optimization pipeline.
        
        Returns:
            OptimizationResult with scores and optimized agent
        """
        start_time = time.time()
        
        try:
            # Setup
            self._configure_dspy()
            self._load_data()
            self._create_metric()
            
            # Create baseline
            self._emit(EventType.LOG, message="Creating baseline agent...")
            self._baseline_agent = SGDAgentModule(self._tool_catalog)
            
            # Evaluate baseline
            self._emit(EventType.LOG, message="Evaluating baseline...")
            self.baseline_score = self._evaluate_agent(
                self._baseline_agent,
                num_samples=min(10, len(self._evalset)),
            )
            self._emit(
                EventType.LOG,
                message=f"Baseline score: {self.baseline_score:.3f}",
                baseline_score=self.baseline_score,
            )
            
            # Run optimization
            if self.strategy == "mipro":
                optimized_agent = self._run_mipro()
            else:
                optimized_agent = self._run_bootstrap()
            
            # Evaluate optimized
            self._emit(EventType.LOG, message="Evaluating optimized agent...")
            optimized_score = self._evaluate_agent(
                optimized_agent,
                num_samples=min(10, len(self._evalset)),
            )
            
            # Calculate improvement
            improvement = optimized_score - self.baseline_score
            improvement_pct = (improvement / self.baseline_score * 100) if self.baseline_score > 0 else 0
            
            training_time = time.time() - start_time
            
            # Capture prompts
            initial_prompt = self._baseline_agent.signature.__doc__ or ""
            optimized_prompt = self._get_optimized_prompt(optimized_agent)
            few_shot_demos = self._get_few_shot_demos(optimized_agent)
            
            # Emit completion
            self._emit(
                EventType.COMPLETE,
                baseline_score=self.baseline_score,
                optimized_score=optimized_score,
                improvement=improvement,
                improvement_pct=improvement_pct,
                training_time=training_time,
            )
            
            # Build result
            result = OptimizationResult(
                success=True,
                optimized_agent=optimized_agent,
                baseline_score=self.baseline_score,
                optimized_score=optimized_score,
                improvement=improvement,
                improvement_pct=improvement_pct,
                training_time=training_time,
                num_examples=len(self._trainset),
                strategy=self.strategy,
                events=self.events,
                initial_prompt=initial_prompt,
                optimized_prompt=optimized_prompt,
                few_shot_demos=few_shot_demos,
            )
            
            # Save if path provided
            if self.output_path:
                try:
                    Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
                    
                    # Save the agent checkpoint
                    optimized_agent.save(self.output_path)
                    self._emit(EventType.LOG, message=f"Saved agent to {self.output_path}")
                    
                    # Save the full result (metrics, prompts, events)
                    result_path = self.output_path.replace('.json', '_result.json')
                    result.save(result_path)
                    self._emit(EventType.LOG, message=f"Saved result to {result_path}")
                    
                except Exception as e:
                    self._emit(EventType.LOG, message=f"Save error: {e}", level="warning")
            
            return result
            
        except Exception as e:
            self._emit(EventType.ERROR, message=str(e))
            logger.exception("Optimization failed")
            
            return OptimizationResult(
                success=False,
                error_message=str(e),
                training_time=time.time() - start_time,
                strategy=self.strategy,
                events=self.events,
            )


def run_optimization(
    data_path: str,
    strategy: str = "bootstrap",
    callback: Optional[Callable] = None,
    **kwargs,
) -> OptimizationResult:
    """
    Convenience function to run optimization.
    
    Args:
        data_path: Path to SGD dataset
        strategy: "bootstrap" or "mipro"
        callback: Progress callback function
        **kwargs: Additional arguments to OptimizationRunner
        
    Returns:
        OptimizationResult
    """
    runner = OptimizationRunner(
        data_path=data_path,
        strategy=strategy,
        callback=callback,
        **kwargs,
    )
    return runner.run()

