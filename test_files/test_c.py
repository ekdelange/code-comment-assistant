"""
This module provides a simple calculator class and several utility functions.

Classes:
- Calculator: A basic calculator that supports addition and division operations, and maintains a history of operations.

Functions:
- fibonacci(n: int) -> List[int]: Generates a list containing the Fibonacci sequence up to the nth number.
- is_prime(number: int) -> bool: Determines whether a given number is prime.

The Calculator class includes methods for adding and dividing two numbers, with division handling the case of division by zero gracefully.
The fibonacci function generates the Fibonacci sequence, and the is_prime function checks for prime numbers using a straightforward algorithm.
"""
import math
from typing import List, Optional

class Calculator:
def __init__(self):
    """
    Initializes a new instance of the class, setting up an empty history list.
    """
    self.history = []

def add(self, a: float, b: float) -> float:
    """
    Adds two floating-point numbers and records the operation in the history.

    Args:
        a (float): The first number to add.
        b (float): The second number to add.

    Returns:
        float: The result of adding a and b.
    """
    result = a + b
    self.history.append(f"Added {a} + {b} = {result}")
    return result

def divide(self, a: float, b: float) -> Optional[float]:
    """
    Divides one floating-point number by another and records the operation in the history.
    If the divisor is zero, returns None.

    Args:
        a (float): The dividend.
        b (float): The divisor.

    Returns:
        Optional[float]: The result of dividing a by b, or None if b is zero.
    """
    if b == 0:
        return None
    result = a / b
    self.history.append(f"Divided {a} / {b} = {result}")
    return resultdef fibonacci(n: int) -> List[int]:
    """
    Generate a list containing the Fibonacci sequence up to the n-th term.

    Parameters:
    n (int): The number of terms in the Fibonacci sequence to generate.

    Returns:
    List[int]: A list containing the Fibonacci sequence up to the n-th term.
               If n <= 0, returns an empty list. If n == 1, returns a list with
               a single element [0].
    """
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    
    sequence = [0, 1]
    for i in range(2, n):
        sequence.append(sequence[i-1] + sequence[i-2])
    return sequencedef is_prime(number: int) -> bool:
    """
    Determine if a given number is a prime number.

    A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself.

    Args:
        number (int): The number to check for primality.

    Returns:
        bool: True if the number is prime, False otherwise.
    """
    if number < 2:
        return False
    for i in range(2, int(math.sqrt(number)) + 1):
        if number % i == 0:
            return False
    return True