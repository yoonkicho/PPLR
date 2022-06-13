from __future__ import absolute_import

from .triplet import TripletLoss, SoftTripletLoss
from .crossentropy import CrossEntropyLabelSmooth, SoftEntropy
from .loss import AALS, PGLR, InterCamProxy

__all__ = [
    'TripletLoss',
    'CrossEntropyLabelSmooth',
    'SoftTripletLoss',
    'SoftEntropy',
    'AALS',
    'PGLR',
    'InterCamProxy'
]