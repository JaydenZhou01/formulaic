from enum import IntEnum
from typing import Type

class SeparatorType(IntEnum):
    BEG = 0
    SEP = 1

class FeatureType(IntEnum):
    OPEN = 0
    HIGH = 1
    LOW = 2
    CLOSE = 3
    VWAP = 4
    VOLUME = 5

class Token:
    def __repr__(self):
        return str(self)
    
class ConstantToken(Token):
    def __init__(self, constant: float):
        self.constant = constant
        
    def __str__(self):
        return str(self.constant)
    
class DeltaTimeToken(Token):
    def __init__(self, delta: int):
        self.delta = delta
        
    def __str__(self):
        return str(self.delta)

class FeatureToken(Token):
    def __init__(self, feature: FeatureType):
        self.feature = feature
        
    def __str__(self):
        return self.feature.name.lower()

class OperatorToken(Token):
    def __init__(self, operator):
        self.operator = operator
        
    def __str__(self):
        return self.operator.__name__
    
class SeparatorToken(Token):
    def __init__(self, sep: SeparatorType):
        self.indicator = sep
        
    def __str__(self):
        return self.indicator.name

class ExpressionToken(Token):
    def __init__(self, expr):
        self.expr = expr
    
    def __str__(self):
        return str(self.expr)
    
BEG_TOKEN = SeparatorToken(SeparatorType.BEG)
SEP_TOKEN = SeparatorToken(SeparatorType.SEP)