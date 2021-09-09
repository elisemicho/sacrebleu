# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import pytest
import sacrebleu
import argparse
from collections import namedtuple

EPSILON = 1e-8

# test for README example with empty hypothesis strings check
_refs = [
    ['The dog <eol> bit the man. <eob>', 'The dog <eol> bit the man. <eob>']
]

_hyps = [
    'The dog <eol> bit the man. <eob>', 'The dog bit <eol> the man. <eob>'
]

test_corpus_bleu_cases = [
    (_hyps, _refs, {}, 100.0),   # test for default BLEU settings
    (('', ''), _refs, {}, 0.0),  # ensure that empty hypotheses are not removed
    # (_hyps, _refs, {'tokenize': 'none'}, 49.1919566),
    # (_hyps, _refs, {'tokenize': '13a'}, 48.530827),
    # (_hyps, _refs, {'tokenize': 'intl'}, 43.91623493),
    # (_hyps, _refs, {'smooth_method': 'none'}, 48.530827),
]

@pytest.mark.parametrize("hypotheses, references, kwargs, expected_bleu", test_corpus_bleu_cases)
def test_corpus_bleu(hypotheses, references, kwargs, expected_bleu):
    # args = argparse.Namespace(tokenize=sacrebleu.DEFAULT_TOKENIZER)
    # metric = sacrebleu.metrics.BR_BLEU(args)
    # bleu = metric.corpus_break_centered_score(hypotheses, references).score

    bleu = sacrebleu.corpus_break_centered_bleu(hypotheses, references, **kwargs).score
    print("BR_BLEU=%f" %bleu)
    assert abs(bleu - expected_bleu) < EPSILON
