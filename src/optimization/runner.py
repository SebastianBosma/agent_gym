"""
Environment Runner - Creates optimized RL environments for training dialogue agents.

This module provides:
- EnvironmentRunner: Optimizes SimulatedUserAgent and creates full environment
- Progress callbacks for frontend integration
- Structured events for real-time updates
- Saves environment components for later use
"""

import os
import time
import json
import logging
from typing import Optional, Callable, List, Dict, Any, Literal, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum

import dspy
from dspy.teleprompt import BootstrapFewShot

from ..data.sgd_loader import SGDLoader
from ..agent.simulated_user import (
    SimulatedUserAgent,
    SimulatedUserModule,
    extract_user_training_examples,
    extract_all_user_goals,
    create_user_simulation_metric,
    compute_user_simulation_similarity,
)
from ..tool_mocker.sgd_mocker import LLMToolMocker, SGDToolMocker
from ..reward.llm_judge import CachedLLMJudge


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
            "timestamp": self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            **self.data,
        }


@dataclass
class EnvironmentResult:
    """
    Result of creating an optimized RL environment.
    
    Contains all components needed for training:
    - Optimized SimulatedUserAgent
    - Tool mocker configuration
    - Reward function configuration
    """
    success: bool
    
    # Components
    simulated_user: Optional[SimulatedUserAgent] = None
    tool_mocker: Optional[SGDToolMocker] = None
    reward_fn: Optional[CachedLLMJudge] = None
    
    # Optimization metrics
    baseline_score: float = 0.0
    optimized_score: float = 0.0
    improvement: float = 0.0
    improvement_pct: float = 0.0
    training_time: float = 0.0
    
    # Details
    num_user_goals: int = 0
    num_training_examples: int = 0
    num_tool_methods: int = 0
    strategy: str = ""
    events: List[OptimizationEvent] = field(default_factory=list)
    error_message: Optional[str] = None
    
    # Prompts (for inspection)
    simulated_user_prompt: str = ""
    few_shot_demos: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self, include_full_data: bool = False) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "success": self.success,
            "baseline_score": self.baseline_score,
            "optimized_score": self.optimized_score,
            "improvement": self.improvement,
            "improvement_pct": self.improvement_pct,
            "training_time": self.training_time,
            "num_user_goals": self.num_user_goals,
            "num_training_examples": self.num_training_examples,
            "num_tool_methods": self.num_tool_methods,
            "strategy": self.strategy,
            "error_message": self.error_message,
        }
        
        if include_full_data:
            result["simulated_user_prompt"] = self.simulated_user_prompt
            result["few_shot_demos"] = self.few_shot_demos
            result["events"] = [e.to_dict() for e in self.events]
        
        return result
    
    def save(self, output_dir: str) -> Dict[str, str]:
        """
        Save all environment components to a directory.
        
        Args:
            output_dir: Directory to save components
            
        Returns:
            Dict mapping component names to file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Save simulated user module
        if self.simulated_user:
            user_path = output_path / "simulated_user.json"
            try:
                self.simulated_user.get_module().save(str(user_path))
                saved_files["simulated_user"] = str(user_path)
            except Exception as e:
                logger.warning(f"Could not save simulated user: {e}")
        
        # Save result metadata
        result_path = output_path / "environment_result.json"
        with open(result_path, "w") as f:
            json.dump(self.to_dict(include_full_data=True), f, indent=2)
        saved_files["result"] = str(result_path)
        
        return saved_files
    
    @classmethod
    def load(cls, output_dir: str) -> "EnvironmentResult":
        """Load result from a directory."""
        result_path = Path(output_dir) / "environment_result.json"
        
        with open(result_path, "r") as f:
            data = json.load(f)
        
        # Reconstruct events
        events = []
        for e in data.get("events", []):
            e_copy = dict(e)
            event_type = e_copy.pop("type", "log")
            timestamp = e_copy.pop("timestamp", "")
            events.append(OptimizationEvent(
                type=event_type,
                data=e_copy,
                timestamp=timestamp,
            ))
        
        return cls(
            success=data["success"],
            baseline_score=data.get("baseline_score", 0.0),
            optimized_score=data.get("optimized_score", 0.0),
            improvement=data.get("improvement", 0.0),
            improvement_pct=data.get("improvement_pct", 0.0),
            training_time=data.get("training_time", 0.0),
            num_user_goals=data.get("num_user_goals", 0),
            num_training_examples=data.get("num_training_examples", 0),
            num_tool_methods=data.get("num_tool_methods", 0),
            strategy=data.get("strategy", ""),
            events=events,
            error_message=data.get("error_message"),
            simulated_user_prompt=data.get("simulated_user_prompt", ""),
            few_shot_demos=data.get("few_shot_demos", []),
        )


# Keep old names for backward compatibility
OptimizationResult = EnvironmentResult
OptimizationEvent = OptimizationEvent


class EnvironmentRunner:
    """
    Creates optimized RL environments for training dialogue agents.
    
    This runner:
    1. Loads SGD data and extracts user goals
    2. Optimizes the SimulatedUserAgent using DSPy
    3. Creates LLM-powered tool mocker
    4. Creates LLM judge reward function
    5. Saves all components
    
    Example:
        def on_progress(event):
            print(f"{event.type}: {event.data}")
        
        runner = EnvironmentRunner(
            data_path="data/raw/schema_guided_dialogue",
            strategy="bootstrap",
            callback=on_progress,
        )
        result = runner.run()
        
        # Use the environment
        simulated_user = result.simulated_user
        tool_mocker = result.tool_mocker
        reward_fn = result.reward_fn
    """
    
    def __init__(
        self,
        data_path: str,
        strategy: Literal["bootstrap", "mipro"] = "bootstrap",
        num_train: int = 100,
        num_eval: int = 20,
        max_demos: int = 4,
        num_candidates: int = 10,
        model: str = "gemini-2.0-flash",
        use_llm_tool_mocker: bool = True,
        callback: Optional[Callable[[OptimizationEvent], None]] = None,
        output_path: Optional[str] = None,
    ):
        """
        Initialize the environment runner.
        
        Args:
            data_path: Path to SGD dataset
            strategy: "bootstrap" (fast) or "mipro" (Gemini rewrites prompts)
            num_train: Number of training dialogues for user simulation
            num_eval: Number of evaluation examples
            max_demos: Max few-shot demos for bootstrap
            num_candidates: Number of candidates for MIPRO
            model: Gemini model to use
            use_llm_tool_mocker: Use LLM for generating plausible tool results
            callback: Function called with progress events
            output_path: Directory to save environment components
        """
        self.data_path = Path(data_path)
        self.strategy = strategy
        self.num_train = num_train
        self.num_eval = num_eval
        self.max_demos = max_demos
        self.num_candidates = num_candidates
        self.model = model
        self.use_llm_tool_mocker = use_llm_tool_mocker
        self.callback = callback
        self.output_path = output_path
        
        # State
        self.events: List[OptimizationEvent] = []
        self.current_step = 0
        self.total_steps = 0
        self.baseline_score = 0.0
        
        # Components (lazy loaded)
        self._loader: Optional[SGDLoader] = None
        self._trainset: Optional[List] = None
        self._evalset: Optional[List] = None
        self._metric_fn = None
        self._baseline_module: Optional[SimulatedUserModule] = None
    
    def _emit(self, event_type: EventType, **data):
        """Emit an event to the callback."""
        event = OptimizationEvent(type=event_type, data=data)
        self.events.append(event)
        
        logger.info(f"[{event_type.value}] {data}")
        
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
        self._emit(EventType.LOG, message="Loading SGD data...")
        
        self._loader = SGDLoader(self.data_path)
        
        # Load dialogues
        train_dialogues = self._loader.load_dialogues("train", limit=self.num_train)
        eval_dialogues = self._loader.load_dialogues("dev", limit=self.num_eval)
        
        # Extract user simulation training examples
        self._trainset = extract_user_training_examples(train_dialogues)
        self._evalset = extract_user_training_examples(eval_dialogues)
        
        # Count initial vs response examples
        initial_count = sum(1 for ex in self._trainset 
                          if not hasattr(ex, 'assistant_response') or not ex.assistant_response)
        
        self._emit(
            EventType.LOG,
            message=f"Loaded {len(train_dialogues)} dialogues",
            num_dialogues=len(train_dialogues),
            num_examples=len(self._trainset),
            initial_messages=initial_count,
            response_messages=len(self._trainset) - initial_count,
        )
    
    def _evaluate_simulated_user(
        self,
        module: SimulatedUserModule,
        num_samples: int = None,
    ) -> float:
        """Evaluate simulated user on the eval set."""
        evalset = self._evalset[:num_samples] if num_samples else self._evalset
        
        scores = []
        for i, example in enumerate(evalset):
            try:
                has_assistant = hasattr(example, 'assistant_response') and example.assistant_response
                
                if has_assistant:
                    pred = module(
                        user_goal=example.user_goal,
                        conversation_history=example.conversation_history,
                        assistant_response=example.assistant_response,
                    )
                else:
                    pred = module(user_goal=example.user_goal)
                
                score = compute_user_simulation_similarity(
                    pred.user_message,
                    example.user_message,
                )
                scores.append(score)
                
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
    
    def _run_bootstrap(self) -> SimulatedUserModule:
        """Run BootstrapFewShot optimization on SimulatedUserModule."""
        self._emit(
            EventType.LOG,
            message="Starting BootstrapFewShot optimization for SimulatedUser",
            strategy="bootstrap",
            max_demos=self.max_demos,
        )
        
        self.total_steps = min(50, len(self._trainset))
        self._emit(EventType.START, total_steps=self.total_steps, strategy="bootstrap")
        
        optimizer = BootstrapFewShot(
            metric=self._metric_fn,
            max_bootstrapped_demos=self.max_demos,
            max_labeled_demos=self.max_demos,
        )
        
        train_subset = self._trainset[:self.total_steps]
        
        # Track progress
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
        
        optimized = optimizer.compile(self._baseline_module, trainset=train_subset)
        
        return optimized
    
    def _run_mipro(self) -> SimulatedUserModule:
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
            message="Starting MIPRO optimization for SimulatedUser",
            strategy="mipro",
            num_candidates=self.num_candidates,
        )
        
        self.total_steps = self.num_candidates * min(20, len(self._trainset))
        self._emit(EventType.START, total_steps=self.total_steps, strategy="mipro")
        
        train_subset = self._trainset[:min(30, len(self._trainset))]
        eval_subset = self._evalset[:min(10, len(self._evalset))] if self._evalset else None
        
        optimizer = MIPROv2(
            metric=self._metric_fn,
            num_candidates=self.num_candidates,
            init_temperature=1.0,
            verbose=True,
        )
        
        try:
            optimized = optimizer.compile(
                self._baseline_module,
                trainset=train_subset,
                valset=eval_subset,
                num_trials=self.num_candidates,
                minibatch=True,
                minibatch_size=min(10, len(train_subset)),
            )
        except Exception as e:
            self._emit(
                EventType.LOG,
                message=f"MIPRO error: {e}, falling back to BootstrapFewShot",
                level="warning",
            )
            return self._run_bootstrap()
        
        return optimized
    
    def _get_simulated_user_prompt(self, module: SimulatedUserModule) -> str:
        """Extract prompt from simulated user module."""
        try:
            if hasattr(module, 'respond') and hasattr(module.respond, 'signature'):
                return module.respond.signature.__doc__ or ""
        except Exception:
            pass
        return ""
    
    def _get_few_shot_demos(self, module: SimulatedUserModule) -> List[Dict[str, Any]]:
        """Extract few-shot demos from module."""
        demos = []
        try:
            for predictor_name in ['respond', 'start']:
                predictor = getattr(module, predictor_name, None)
                if predictor and hasattr(predictor, 'demos'):
                    for demo in predictor.demos or []:
                        demo_dict = {}
                        for attr in ['user_goal', 'conversation_history', 'assistant_response', 'user_message']:
                            if hasattr(demo, attr):
                                demo_dict[attr] = getattr(demo, attr)
                        if demo_dict:
                            demos.append(demo_dict)
        except Exception:
            pass
        return demos
    
    def run(self) -> EnvironmentResult:
        """
        Run the environment creation pipeline.
        
        Returns:
            EnvironmentResult with optimized components
        """
        start_time = time.time()
        
        try:
            # Setup
            self._configure_dspy()
            self._load_data()
            self._metric_fn = create_user_simulation_metric()
            
            # Create baseline simulated user
            self._emit(EventType.LOG, message="Creating baseline SimulatedUser...")
            self._baseline_module = SimulatedUserModule()
            
            # Evaluate baseline
            self._emit(EventType.LOG, message="Evaluating baseline SimulatedUser...")
            self.baseline_score = self._evaluate_simulated_user(
                self._baseline_module,
                num_samples=min(15, len(self._evalset)),
            )
            self._emit(
                EventType.LOG,
                message=f"Baseline score: {self.baseline_score:.3f}",
                baseline_score=self.baseline_score,
            )
            
            # Optimize simulated user
            if self.strategy == "mipro":
                optimized_module = self._run_mipro()
            else:
                optimized_module = self._run_bootstrap()
            
            # Evaluate optimized
            self._emit(EventType.LOG, message="Evaluating optimized SimulatedUser...")
            optimized_score = self._evaluate_simulated_user(
                optimized_module,
                num_samples=min(15, len(self._evalset)),
            )
            
            # Calculate improvement
            improvement = optimized_score - self.baseline_score
            improvement_pct = (improvement / self.baseline_score * 100) if self.baseline_score > 0 else 0
            
            self._emit(
                EventType.LOG,
                message=f"Optimized score: {optimized_score:.3f} ({improvement_pct:+.1f}%)",
                optimized_score=optimized_score,
                improvement_pct=improvement_pct,
            )
            
            # Create full environment components
            self._emit(EventType.LOG, message="Creating environment components...")
            
            # Create SimulatedUserAgent with optimized module
            simulated_user = SimulatedUserAgent(
                data_path=str(self.data_path),
                split="train",
                model=self.model,
            )
            simulated_user.set_module(optimized_module)
            
            # Create tool mocker
            if self.use_llm_tool_mocker:
                tool_mocker = LLMToolMocker.from_loader(
                    self._loader, "train", use_llm_fallback=True
                )
                self._emit(EventType.LOG, message="Created LLMToolMocker with fallback")
            else:
                tool_mocker = SGDToolMocker.from_loader(self._loader, "train")
                self._emit(EventType.LOG, message="Created SGDToolMocker")
            
            # Create reward function
            reward_fn = CachedLLMJudge()
            self._emit(EventType.LOG, message="Created CachedLLMJudge reward function")
            
            training_time = time.time() - start_time
            
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
            result = EnvironmentResult(
                success=True,
                simulated_user=simulated_user,
                tool_mocker=tool_mocker,
                reward_fn=reward_fn,
                baseline_score=self.baseline_score,
                optimized_score=optimized_score,
                improvement=improvement,
                improvement_pct=improvement_pct,
                training_time=training_time,
                num_user_goals=simulated_user.get_num_goals(),
                num_training_examples=len(self._trainset),
                num_tool_methods=len(tool_mocker.get_available_methods()),
                strategy=self.strategy,
                events=self.events,
                simulated_user_prompt=self._get_simulated_user_prompt(optimized_module),
                few_shot_demos=self._get_few_shot_demos(optimized_module),
            )
            
            # Save if path provided
            if self.output_path:
                saved = result.save(self.output_path)
                self._emit(EventType.LOG, message=f"Saved to {self.output_path}", files=saved)
            
            return result
            
        except Exception as e:
            self._emit(EventType.ERROR, message=str(e))
            logger.exception("Environment creation failed")
            
            return EnvironmentResult(
                success=False,
                error_message=str(e),
                training_time=time.time() - start_time,
                strategy=self.strategy,
                events=self.events,
            )


# Backward compatibility alias
OptimizationRunner = EnvironmentRunner


def run_optimization(
    data_path: str,
    strategy: str = "bootstrap",
    callback: Optional[Callable] = None,
    **kwargs,
) -> EnvironmentResult:
    """
    Convenience function to create an optimized environment.
    
    Args:
        data_path: Path to SGD dataset
        strategy: "bootstrap" or "mipro"
        callback: Progress callback function
        **kwargs: Additional arguments to EnvironmentRunner
        
    Returns:
        EnvironmentResult with optimized components
    """
    runner = EnvironmentRunner(
        data_path=data_path,
        strategy=strategy,
        callback=callback,
        **kwargs,
    )
    return runner.run()
