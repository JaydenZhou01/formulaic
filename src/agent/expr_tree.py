from typing import List
from agent.expression import BinaryOperator, Constant, DeltaTime, Expression, Feature, Operator, Pow, RollingOperator, UnaryOperator
from agent.tokens import ConstantToken, DeltaTimeToken, ExpressionToken, FeatureToken, OperatorToken, Token
from agent.utils import parse_token


class ExprTree:
    def __init__(self):
        self.stack: List[Expression] = []

    def get_stack(self):
        if len(self.stack) == 1:
            return self.stack[0]
        else:
            raise ValueError("Stack is not valid. Expected a single expression.")
        
    def push_tokens(self, tokens: List[str]):
        for token in tokens:
            token_obj = parse_token(token)
            self.push(token_obj)
    
    def push(self, token: Token):
        if not self.validate(token):
            raise ValueError(f"Invalid token: {token}")
        
        if isinstance(token, OperatorToken):
            n_args = token.operator.n_args()
            children = []
            for _ in range(n_args):
                children.append(self.stack.pop())
            self.stack.append(token.operator(*reversed(children)))
        elif isinstance(token, ConstantToken):
            self.stack.append(Constant(token.constant))
        elif isinstance(token, DeltaTimeToken):
            self.stack.append(DeltaTime(token.delta))
        elif isinstance(token, FeatureToken):
            self.stack.append(Feature(token.feature))
        elif isinstance(token, ExpressionToken):
            self.stack.append(token.expr)
        else:
            assert False, f"Unknown token type: {type(token)}"
        
    def is_valid(self):
        return len(self.stack) == 1 and self.stack[0].is_featured
    
    def validate(self, token: Token):
        if isinstance(token, OperatorToken):
            return self.validate_op(token.operator)
        elif isinstance(token, ConstantToken):
            return self.validate_const()
        elif isinstance(token, DeltaTimeToken):
            return self.validate_delta()
        elif isinstance(token, (FeatureToken, ExpressionToken)):
            return self.validate_feature_expr()
        else:
            assert False, f"Unknown token type: {type(token)}"

    def validate_op(self, op: type[Operator]):
        if len(self.stack) < op.n_args():
                return False
        if issubclass(op, UnaryOperator):
            if not self.stack[-1].is_featured:
                return False
        elif issubclass(op, BinaryOperator):
            if issubclass(op, Pow):
                if not isinstance(self.stack[-1], Constant):
                    return False
            if not self.stack[-1].is_featured and not self.stack[-2].is_featured:
                return False
            if (isinstance(self.stack[-1], DeltaTime) or
                    isinstance(self.stack[-2], DeltaTime)):
                return False
        elif issubclass(op, RollingOperator):
            if not isinstance(self.stack[-1], DeltaTime):
                return False
            if not self.stack[-2].is_featured:
                return False
        else:
            assert False, f"Unknown operator type: {op}"
        return True
    
    def validate_const(self):
        return len(self.stack) == 0 or self.stack[-1].is_featured
    
    def validate_delta(self):
        return len(self.stack) > 0 and self.stack[-1].is_featured
    
    def validate_feature_expr(self):
        return not (len(self.stack) >= 1 and isinstance(self.stack[-1], DeltaTime))
            
        
