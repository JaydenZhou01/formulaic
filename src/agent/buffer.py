from collections import deque
from typing import Any, Dict, List, NamedTuple, Optional, Sequence
import numpy as np
import random

class Transition(NamedTuple):
    obs: Dict[str, Any]
    pi: Optional[np.ndarray]
    value: Optional[float]
    mask: Optional[np.ndarray]

class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0):
        self.capacity = capacity
        self.random = random.Random(seed)
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def add_game(self, game_seq: Sequence[Transition]):
        for transition in game_seq:
            self.buffer.append(transition)

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size: int) -> List[Transition]:
        if len(self.buffer) < batch_size:
            raise ValueError("Not enough samples to sample from.")
        indices = self.random.sample(range(len(self.buffer)), batch_size)
        batch = [self.buffer[i] for i in indices]
        return batch
    
    def clear(self):
        self.buffer.clear()