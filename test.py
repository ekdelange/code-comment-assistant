import math
from typing import List, Optional

class Calculator:
    def __init__(self):
        self.history = []

    def add(self, a: float, b: float) -> float:
        result = a + b
        self.history.append(f"Added {a} + {b} = {result}")
        return result

    def divide(self, a: float, b: float) -> Optional[float]:
        if b == 0:
            return None
        result = a / b
        self.history.append(f"Divided {a} / {b} = {result}")
        return result

def fibonacci(n: int) -> List[int]:
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    
    sequence = [0, 1]
    for i in range(2, n):
        sequence.append(sequence[i-1] + sequence[i-2])
    return sequence

def is_prime(number: int) -> bool:
    if number < 2:
        return False
    for i in range(2, int(math.sqrt(number)) + 1):
        if number % i == 0:
            return False
    return True
