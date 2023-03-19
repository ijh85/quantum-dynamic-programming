"""Utility functions."""

from sympy import Symbol

class Binary(Symbol):
    """A binary variable."""	
    def __pow__(self, other):
        return self

def keys_to_integers(x):
    """Converts dictionary keys to integers."""
    return {int(k): v for k, v in x.items()}