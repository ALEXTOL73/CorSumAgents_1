# agents/__init__.py
from .base_agent import BaseAgent
from .correction_prompt_generator import CorrectionPromptGenerator
from .corrector_ensemble import CorrectorEnsemble
from .corrector_aggregator import CorrectorAggregator
from .correction_judge import CorrectionJudge
from .summarization_prompt_generator import SummarizationPromptGenerator
from .summarizer_ensemble import SummarizerEnsemble
from .summarizer_aggregator import SummarizerAggregator
from .summarizer import Summarizer
from .summarization_judge import SummarizationJudge
from .hyperparameter_optimizer import HyperparameterOptimizer   # ✅ новый агент
from .task_generator import TaskGenerator                       # ✅ новый агент
from .reflection_agent import ReflectionAgent                   # ✅ новый агент

__all__ = [
    'BaseAgent',
    'CorrectionPromptGenerator',
    'CorrectorEnsemble',
    'CorrectorAggregator',
    'CorrectionJudge',
    'SummarizationPromptGenerator',
    'SummarizerEnsemble',
    'SummarizerAggregator',
    'Summarizer',
    'SummarizationJudge',
    'HyperparameterOptimizer',
    'TaskGenerator',
]