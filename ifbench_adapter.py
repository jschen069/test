import logging
import os
from typing import Any, Dict, List

from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
from evalscope.api.dataset import Sample
from evalscope.api.evaluator import TaskState
from evalscope.api.messages import ChatMessageUser
from evalscope.api.metric import Score
from evalscope.api.registry import register_benchmark
from evalscope.constants import Tags
from evalscope.utils.import_utils import check_import
from evalscope.utils.logger import get_logger

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name='ifbench',
        pretty_name='IFBench',
        description="""
## Overview

IFBench is a benchmark designed to evaluate how reliably AI models follow novel, challenging, and diverse verifiable instructions, with a strong focus on out-of-domain generalization. Developed by AllenAI, it addresses overfitting and data contamination issues in existing benchmarks.

## Task Description

- **Task Type**: Instruction Following Evaluation
- **Input**: Prompts with verifiable constraints
- **Output**: Responses that must satisfy specific constraints
- **Focus**: Precise instruction-following capabilities

## Key Features

- 58 manually curated verifiable constraints
- Categories: counting, formatting, word usage, etc.
- Focus on out-of-domain generalization
- Programmatic verification of constraint satisfaction
- Addresses data contamination concerns

## Evaluation Notes

- Default configuration uses **0-shot** evaluation
- Metrics: prompt_level_strict, inst_level_strict, prompt_level_loose, inst_level_loose
- Requires emoji, syllapy packages
- Evaluates both strict and loose constraint satisfaction
""",  # noqa: E501
        tags=[Tags.INSTRUCTION_FOLLOWING],
        dataset_id='allenai/IFBench_test',
        subset_list=['default'],
        metric_list=[
            'prompt_level_strict',
            'inst_level_strict',
            'prompt_level_loose',
            'inst_level_loose',
        ],
        few_shot_num=0,
        train_split=None,
        eval_split='train',
    )
)
class IFBenchAdapter(DefaultDataAdapter):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        check_import(
            module_name=['emoji', 'syllapy', 'nltk'], extra='ifbench', raise_error=True, feature_name=self.pretty_name
        )

    def _get_debug_logger(self) -> logging.Logger:
        """Lazy-init a dedicated file logger for prompt / eval debug output.

        Writes to ``outputs/<timestamp>/logs/eval_prompt_log.log`` so verbose
        per-sample output does not flood the main console / log file.
        """
        if hasattr(self, '_debug_logger') and self._debug_logger is not None:
            return self._debug_logger

        logs_dir = os.path.join(self.output_dir, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        log_file = os.path.join(logs_dir, 'eval_prompt_log.log')

        debug_logger = logging.getLogger('evalscope.ifbench_debug')
        debug_logger.propagate = False
        debug_logger.setLevel(logging.DEBUG)
        debug_logger.handlers.clear()

        handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        debug_logger.addHandler(handler)

        logger.info(f'[IFBench DEBUG] Prompt debug log: {log_file}')
        self._debug_logger = debug_logger
        return debug_logger

    def record_to_sample(self, record: Dict[str, Any]) -> Sample:
        prompt = record.get('prompt', '')
        message_list = [ChatMessageUser(content=prompt)]

        # ====== log constructed prompt ======
        dbg = self._get_debug_logger()
        dbg.info('========== CONSTRUCTED PROMPT (key=%s) ==========', record.get('key', '?'))
        dbg.info('PROMPT: %s', prompt)
        dbg.info('INSTRUCTION_IDS: %s', record.get('instruction_id_list', []))
        dbg.info('KWARGS: %s', record.get('kwargs', []))
        dbg.info('========== CONSTRUCTED PROMPT END ==========')
        # =====================================

        return Sample(input=message_list, target='', metadata=record)

    def match_score(
        self, original_prediction: str, filtered_prediction: str, reference: Dict, task_state: TaskState
    ) -> Score:
        """
        Calculate evaluation scores by comparing prediction with reference.
        """
        from evalscope.benchmarks.ifbench.evaluation_lib import process_results

        # Initialize the score object with prediction details
        score = Score(
            extracted_prediction=filtered_prediction,
            prediction=original_prediction,
        )

        doc = task_state.metadata

        # ====== log eval input ======
        dbg = self._get_debug_logger()
        dbg.info('========== EVAL INPUT (sample_id=%s) ==========', task_state.sample_id)
        dbg.info('PROMPT:     %s', doc.get('prompt', ''))
        dbg.info('PREDICTION: %s', filtered_prediction)
        dbg.info('INSTRUCTION_IDS: %s', doc.get('instruction_id_list', []))
        dbg.info('KWARGS:     %s', doc.get('kwargs', []))
        # ============================

        try:
            # Process results using the existing ifeval utility
            results = process_results(doc, [filtered_prediction])
            score.value.update(results)

            # Set main score name
            score.main_score_name = 'prompt_level_strict'

            # ====== log eval result ======
            dbg.info('========== EVAL RESULT (sample_id=%s) ==========', task_state.sample_id)
            dbg.info('SCORE: %s', results)
            dbg.info('========== EVAL RESULT END ==========')
            # =============================

        except Exception as e:
            logger.error(f'Error calculating ifbench metrics: {e}')
            score.value = {}

        return score
