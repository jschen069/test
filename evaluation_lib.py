# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Binary of evaluating instruction following. See README.md."""

import collections
import dataclasses
from typing import Dict, Optional, Sequence, Union

from evalscope.utils.function_utils import thread_safe
from evalscope.utils.logger import get_logger
from . import instructions_registry

logger = get_logger()


@dataclasses.dataclass
class InputExample:
    key: int
    instruction_id_list: list[str]
    prompt: str
    kwargs: list[Dict[str, Optional[Union[str, int]]]]


@dataclasses.dataclass
class OutputExample:
    instruction_id_list: list[str]
    prompt: str
    response: str
    follow_all_instructions: bool
    follow_instruction_list: list[bool]


def test_instruction_following_strict(
    inp: 'InputExample',
    response: str,
):
    """Tests response to see if instrutions are followed."""
    logger.debug('[ifbench][strict] ========== strict check start ==========')
    logger.debug('[ifbench][strict] inp.key: %s', inp.key)
    logger.debug('[ifbench][strict] inp.instruction_id_list: %s', inp.instruction_id_list)
    logger.debug('[ifbench][strict] inp.prompt: %s', inp.prompt)
    logger.debug('[ifbench][strict] inp.kwargs: %s', inp.kwargs)
    logger.debug('[ifbench][strict] response: %s', response)

    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        logger.debug('[ifbench][strict] --- instruction[%d] %s ---', index, instruction_id)
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        logger.debug('[ifbench][strict] instruction_cls: %s', instruction_cls)
        instruction = instruction_cls(instruction_id)
        logger.debug('[ifbench][strict] instruction: %s', instruction)

        inp.kwargs[index] = {key: value for key, value in inp.kwargs[index].items() if value is not None}
        logger.debug('[ifbench][strict] kwargs[%d] cleaned: %s', index, inp.kwargs[index])

        instruction.build_description(**inp.kwargs[index])
        args = instruction.get_instruction_args()
        logger.debug('[ifbench][strict] args: %s', args)

        if args and 'prompt' in args:
            logger.debug('[ifbench][strict] building description with prompt: %s', inp.prompt)
            instruction.build_description(prompt=inp.prompt)

        check_result = instruction.check_following(response)
        logger.debug('[ifbench][strict] response.strip(): %r', response.strip())
        logger.debug('[ifbench][strict] check_following result: %s', check_result)

        if response.strip() and check_result:
            is_following_list.append(True)
        else:
            is_following_list.append(False)
        logger.debug('[ifbench][strict] instruction[%d] passed: %s', index, is_following_list[-1])

    result = OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )
    logger.debug('[ifbench][strict] result.follow_all_instructions: %s', result.follow_all_instructions)
    logger.debug('[ifbench][strict] result.follow_instruction_list: %s', result.follow_instruction_list)
    logger.debug('[ifbench][strict] ========== strict check end ==========')
    return result


def test_instruction_following_loose(
    inp: 'InputExample',
    response: str,
):
    """Tests response for an upper bound for following instructions."""
    logger.debug('[ifbench][loose] ========== loose check start ==========')
    logger.debug('[ifbench][loose] inp.key: %s', inp.key)
    logger.debug('[ifbench][loose] inp.instruction_id_list: %s', inp.instruction_id_list)
    logger.debug('[ifbench][loose] inp.prompt: %s', inp.prompt)
    logger.debug('[ifbench][loose] inp.kwargs: %s', inp.kwargs)
    logger.debug('[ifbench][loose] response: %s', response)

    r = response.split('\n')
    response_remove_first = '\n'.join(r[1:]).strip()
    response_remove_last = '\n'.join(r[:-1]).strip()
    response_remove_both = '\n'.join(r[1:-1]).strip()
    revised_response = response.replace('*', '')
    revised_response_remove_first = response_remove_first.replace('*', '')
    revised_response_remove_last = response_remove_last.replace('*', '')
    revised_response_remove_both = response_remove_both.replace('*', '')

    logger.debug('[ifbench][loose] r (split lines): %s', r)
    logger.debug('[ifbench][loose] response_remove_first: %r', response_remove_first)
    logger.debug('[ifbench][loose] response_remove_last: %r', response_remove_last)
    logger.debug('[ifbench][loose] response_remove_both: %r', response_remove_both)
    logger.debug('[ifbench][loose] revised_response: %r', revised_response)
    logger.debug('[ifbench][loose] revised_response_remove_first: %r', revised_response_remove_first)
    logger.debug('[ifbench][loose] revised_response_remove_last: %r', revised_response_remove_last)
    logger.debug('[ifbench][loose] revised_response_remove_both: %r', revised_response_remove_both)

    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]
    logger.debug('[ifbench][loose] all_responses (8 variants):')
    for i, ar in enumerate(all_responses):
        logger.debug('[ifbench][loose]   all_responses[%d]: %r', i, ar)

    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        logger.debug('[ifbench][loose] --- instruction[%d] %s ---', index, instruction_id)
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        logger.debug('[ifbench][loose] instruction_cls: %s', instruction_cls)
        instruction = instruction_cls(instruction_id)
        logger.debug('[ifbench][loose] instruction: %s', instruction)

        instruction.build_description(**inp.kwargs[index])
        args = instruction.get_instruction_args()
        logger.debug('[ifbench][loose] kwargs[%d]: %s', index, inp.kwargs[index])
        logger.debug('[ifbench][loose] args: %s', args)

        if args and 'prompt' in args:
            logger.debug('[ifbench][loose] building description with prompt: %s', inp.prompt)
            instruction.build_description(prompt=inp.prompt)

        is_following = False
        for vi, ar in enumerate(all_responses):
            if ar.strip() and instruction.check_following(ar):
                logger.debug('[ifbench][loose] instruction[%d] matched at all_responses[%d]: %r', index, vi, ar)
                is_following = True
                break

        is_following_list.append(is_following)
        logger.debug('[ifbench][loose] instruction[%d] passed: %s', index, is_following)

    result = OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )
    logger.debug('[ifbench][loose] result.follow_all_instructions: %s', result.follow_all_instructions)
    logger.debug('[ifbench][loose] result.follow_instruction_list: %s', result.follow_instruction_list)
    logger.debug('[ifbench][loose] ========== loose check end ==========')
    return result


def print_report(outputs):
    """Prints a report on accuracy scores."""

    prompt_total = 0
    prompt_correct = 0
    instruction_total = 0
    instruction_correct = 0

    tier0_total = collections.defaultdict(int)
    tier0_correct = collections.defaultdict(int)

    tier1_total = collections.defaultdict(int)
    tier1_correct = collections.defaultdict(int)

    for example in outputs:
        follow_instruction_list = example.follow_instruction_list
        instruction_id_list = example.instruction_id_list

        prompt_total += 1
        if all(follow_instruction_list):
            prompt_correct += 1

        instruction_total += len(instruction_id_list)
        instruction_correct += sum(follow_instruction_list)

        for instruction_id, followed_or_not in zip(instruction_id_list, follow_instruction_list):
            instruction_id = instruction_id.split(':')[0]
            tier0_total[instruction_id] += 1
            if followed_or_not:
                tier0_correct[instruction_id] += 1

        for instruction_id, followed_or_not in zip(instruction_id_list, follow_instruction_list):
            tier1_total[instruction_id] += 1
            if followed_or_not:
                tier1_correct[instruction_id] += 1

    print(f'prompt-level: {prompt_correct / prompt_total}')
    print(f'instruction-level: {instruction_correct / instruction_total}')
    print()
    for instruction_id in sorted(tier0_total.keys()):
        accuracy = tier0_correct[instruction_id] / tier0_total[instruction_id]
        print(f'{instruction_id} {accuracy}')
    print()
    for instruction_id in sorted(tier1_total.keys()):
        accuracy = tier1_correct[instruction_id] / tier1_total[instruction_id]
        print(f'{instruction_id} {accuracy}')


@thread_safe
def process_results(doc, results):
    logger.debug('[ifbench][process] ========== process_results start ==========')
    logger.debug('[ifbench][process] doc key: %s', doc.get('key'))
    logger.debug('[ifbench][process] doc instruction_id_list: %s', doc.get('instruction_id_list'))
    logger.debug('[ifbench][process] doc prompt: %s', doc.get('prompt'))
    logger.debug('[ifbench][process] doc kwargs: %s', doc.get('kwargs'))
    logger.debug('[ifbench][process] results: %s', results)

    inp = InputExample(
        key=doc['key'],
        instruction_id_list=doc['instruction_id_list'],
        prompt=doc['prompt'],
        kwargs=doc['kwargs'],
    )
    response = results[0]
    logger.debug('[ifbench][process] InputExample created: key=%d, num_instructions=%d',
                 inp.key, len(inp.instruction_id_list))

    out_strict = test_instruction_following_strict(inp, response)
    logger.debug('[ifbench][process] strict: follow_all=%s, follow_list=%s',
                 out_strict.follow_all_instructions, out_strict.follow_instruction_list)

    out_loose = test_instruction_following_loose(inp, response)
    logger.debug('[ifbench][process] loose: follow_all=%s, follow_list=%s',
                 out_loose.follow_all_instructions, out_loose.follow_instruction_list)

    result = {
        'prompt_level_strict': float(out_strict.follow_all_instructions),
        'inst_level_strict': agg_inst_level_acc(out_strict.follow_instruction_list),
        'prompt_level_loose': float(out_loose.follow_all_instructions),
        'inst_level_loose': agg_inst_level_acc(out_loose.follow_instruction_list),
    }
    logger.debug('[ifbench][process] result: %s', result)
    logger.debug('[ifbench][process] ========== process_results end ==========')
    return result


def agg_inst_level_acc(items):
    inst_level_acc = sum(items) / len(items) if items else 0
    return inst_level_acc
