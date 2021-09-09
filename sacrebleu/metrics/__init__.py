# -*- coding: utf-8 -*-

from .bleu import BLEU, BLEUScore
from .br_bleu import BR_BLEU, BR_BLEUScore
from .chrf import CHRF, CHRFScore
from .ter import TER, TERScore

METRICS = {
    'br_bleu': BR_BLEU,
    'bleu': BLEU,
    'chrf': CHRF,
    'ter': TER,
}
