from abc import ABC, abstractmethod
from ast import arg
from typing import List, Tuple, Type, Union
from webbrowser import Opera
from agent.tokens import FeatureType
from torch import Tensor
import torch

class Expression(ABC):
    """
    Base class for all expressions.
    """

    @abstractmethod
    def evaluate(self, data) -> Tensor:
        pass

    def __repr__(self):
        return str(self)
    
    def __add__(self, other):
        return Add(self, other)
    def __radd__(self, other):
        return Add(other, self)
    def __sub__(self, other):
        return Sub(self, other)
    def __rsub__(self, other):
        return Sub(other, self)
    def __mul__(self, other):
        return Mul(self, other)
    def __rmul__(self, other):
        return Mul(other, self)
    def __truediv__(self, other):
        return Div(self, other)
    def __rtruediv__(self, other):
        return Div(other, self)
    def __pow__(self, other):
        return Pow(self, other)
    def __rpow__(self, other):
        return Pow(other, self)
    def __pos__(self):
        return self
    def __abs__(self):
        return Abs(self)
    def __neg__(self):
        return Sub(0., self)

    @property
    @abstractmethod
    def is_featured(self) -> bool:
        pass

class Feature(Expression):
    """
    Represents a feature in the expression.
    """

    def __init__(self, feature: FeatureType):
        self._feature = feature

    def evaluate(self, data) -> Tensor:
        return data[self._feature.name.lower()]

    @property
    def is_featured(self) -> bool:
        return True

    def __str__(self) -> str:
        return "$" + self._feature.name.lower()
    
class Constant(Expression):
    """
    Represents a constant in the expression.
    """

    def __init__(self, constant: float):
        self._constant = constant

    def evaluate(self, data) -> Tensor:
        return self._constant * data['1.0']

    @property
    def is_featured(self) -> bool:
        return False

    def __str__(self) -> str:
        return str(self._constant)
    
class DeltaTime(Expression):
    """
    Represents the delta time in the expression.
    """

    def __init__(self, delta: int):
        self._delta = delta

    def evaluate(self, data) -> Tensor:
        assert False, "DeltaTime cannot be evaluated directly."

    @property
    def is_featured(self) -> bool:
        return False

    def __str__(self) -> str:
        return f"{self._delta}d"
    
    
class Operator(Expression):
    @classmethod
    @abstractmethod
    def n_args(cls) -> int:
        """
        Returns the number of arguments this operator takes.
        """
        pass
    
    @classmethod
    @abstractmethod
    def category(cls) -> Type["Operator"]:
        pass

    @classmethod
    def _arity_check(cls, *args):
        arity = cls.n_args()
        if len(args) == arity:
            return True
        else:
            raise ValueError(f"Operator {cls.__name__} takes {arity} arguments, but got {len(args)}.")

    @classmethod    
    def _check_exprs_featured(cls, args:list):
        arg_is_featured = False
        for i, args in enumerate(args):
            if not isinstance(arg, (Expression, float)):
                raise TypeError(f"Argument {i} of operator {cls.__name__} must be an Expression or a float.")
            arg_is_featured = arg_is_featured or (isinstance(arg, Expression) and arg.is_featured)

        if not arg_is_featured:
            raise ValueError(f"Operator {cls.__name__} must have at least one featured argument.")
        
        return True
    
    @property
    @abstractmethod
    def operands(self) -> Tuple[Expression, ...]:
        """
        Returns the operands of this operator.
        """
        pass
    
    def __str__(self) -> str:
        return f"{type(self).__name__}({', '.join(map(str, self.operands))})"
    
    
def _into_expr(value: Union["Expression", float]) -> "Expression":
    return value if isinstance(value, Expression) else Constant(value)
    
class UnaryOperator(Operator):
    """
    Base class for unary operators.
    """

    def __init__(self, operand: Union["Expression", float]):
        self._operand = _into_expr(operand)

    @classmethod
    def n_args(cls) -> int:
        return 1
    
    @abstractmethod
    def _apply(self, operand: Tensor) -> Tensor:
        """
        Applies the operator to the operand.
        """
        pass
    
    def evaluate(self, data) -> Tensor:
        return self._apply(self._operand.evaluate(data))
    
    @property
    def operands(self):
        return self._operand,

    @property
    def is_featured(self) -> bool:
        return self._operand.is_featured
    
    @classmethod
    def category(cls):
        return UnaryOperator
    
    
class BinaryOperator(Operator):
    """
    Base class for binary operators.
    """

    def __init__(self, left: Union["Expression", float], right: Union["Expression", float]):
        self._left = _into_expr(left)
        self._right = _into_expr(right)

    @classmethod
    def n_args(cls) -> int:
        return 2
    
    @abstractmethod
    def _apply(self, left: Tensor, right: Tensor) -> Tensor:
        """
        Applies the operator to the left and right operands.
        """
        pass

    def evaluate(self, data) -> Tensor:
        return self._apply(self._left.evaluate(data), self._right.evaluate(data))
    
    def __str__(self) -> str:
        return f"{type(self).__name__}({self._left}, {self._right})"
    
    @property
    def operands(self):
        return self._left, self._right
    
    @property
    def is_featured(self) -> bool:
        return self._left.is_featured or self._right.is_featured
    
    @classmethod
    def category(cls):
        return BinaryOperator
    

def _into_delta(value):
    return value if isinstance(value, DeltaTime) else DeltaTime(value)

class RollingOperator(Operator):
    """
    Base class for rolling operators.
    """

    def __init__(self, operand: Union["Expression", float], window: int):
        self._operand = _into_expr(operand)
        self._window = _into_delta(window)._delta

    @classmethod
    def n_args(cls) -> int:
        return 2
    
    @abstractmethod
    def _apply(self, operand: Tensor) -> Tensor:
        """
        Applies the operator to the operand.
        """
        pass
    
    def evaluate(self, data):
        values = self._operand.evaluate(data)
        values = values.unfold(0, self._window, 1)
        pad = torch.full((self._window - 1, values.shape[1], self._window), float('nan'), device=values.device)
        values = torch.cat([pad, values], dim=0)
        return self._apply(values)
    
    @property
    def operands(self):
        return self._operand, DeltaTime(self._window)
    
    @property
    def is_featured(self) -> bool:
        return self._operand.is_featured
    
    @classmethod
    def category(cls):
        return RollingOperator
    

# Operator Implementations
class Abs(UnaryOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.abs()
    
class Log(UnaryOperator):
    EPS = 1e-6

    def _apply(self, operand: Tensor) -> Tensor:
        # force all values to be at least EPS, avoiding log(0) or log(neg)
        safe_operand = torch.clamp(operand, min=self.EPS)
        return safe_operand.log()
    
class Add(BinaryOperator):
    def _apply(self, left: Tensor, right: Tensor) -> Tensor:
        return left + right
    
class Sub(BinaryOperator):
    def _apply(self, left: Tensor, right: Tensor) -> Tensor:
        return left - right
    
class Mul(BinaryOperator):
    def _apply(self, left: Tensor, right: Tensor) -> Tensor:
        return left * right
    
class Div(BinaryOperator):
    def _apply(self, left: Tensor, right: Tensor) -> Tensor:
        result = left / right
        result = torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        return result
    
class Pow(BinaryOperator):
    def _apply(self, left: Tensor, right: Tensor) -> Tensor:
        return left.pow(right)

class Greater(BinaryOperator):
    def _apply(self, left: Tensor, right: Tensor) -> Tensor:
        return left.max(right)

class Less(BinaryOperator):
    def _apply(self, left: Tensor, right: Tensor) -> Tensor:
        return left.min(right)
    
class Mean(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.mean(dim=-1)
    
class Median(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.median(dim=-1)[0]    

class Sum(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.sum(dim=-1)
    
class Std(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.std(dim=-1)
    
class Var(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.var(dim=-1)
    
class Max(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.max(dim=-1)[0]
    
class Min(RollingOperator):
    def _apply(self, operand: Tensor) -> Tensor:
        return operand.min(dim=-1)[0]


    

