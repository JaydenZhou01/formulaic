from typing import List, Type

from agent.expression import Abs, Add, Greater, Less, Log, Max, Median, Min, Operator, Sub, Mul, Div, Pow, Mean, Sum, Std, Var

MAX_EXPR_LENGTH = 20

OPERATORS = [
    # Unary
    Abs, Log,
    # Binary
    Add, Sub, Mul, Div, Greater, Less, Pow,
    # Rolling
    Mean, Median, Sum, Std, Var, Max, Min
]

DELTA_TIMES = [1, 5, 10, 20, 40]

CONSTANTS = [-30., -10., -5., -2., -1., -0.5, -0.01, 0.01, 0.5, 1., 2., 5., 10., 30.]

COST_PER_STEP = 0.001
TERMINATION_BONUS = 0.01
REWARD_THRESHOLD = 0.02

MIN_EXPR_LENGTH = 3
