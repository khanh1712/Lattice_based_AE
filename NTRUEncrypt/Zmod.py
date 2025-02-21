class Zmod:
    def __init__(self, n):
        if n <= 0:
            raise ValueError("Modulo n must be a positive integer.")
        self.n = n
        

    def __call__(self, value):
        return self.element(value)

    def __repr__(self):
        return f"Ring of integers modulo {self.n}"
    def element(self, value):
        if isinstance(value, int):
            mod_value = value % self.n  
        else: mod_value = value 
        return ZmodElement(mod_value, self)
    
    def zero(self):
        return self.cache[0]
    
    @classmethod
    def base(cls):
        return ZmodElement
    
class ZmodElement:
    __slots__ = ["value", "ring"]  # Reduce memory usage by avoiding dynamic attributes

    def __init__(self, value, ring):
        self.ring = ring
        self.value = value

    def __repr__(self):
        return f"{self.value}"

    def __add__(self, other):
        if isinstance(other, ZmodElement) and self.ring is other.ring:
            return self.ring(self.value + other.value)  
        raise ValueError("Elements must be from the same Zmod ring.")

    def __sub__(self, other):
        if isinstance(other, ZmodElement) and self.ring is other.ring:
            return self.ring(self.value - other.value)  
        raise ValueError("Elements must be from the same Zmod ring.")

    def __mul__(self, other):
        if isinstance(other, ZmodElement) and self.ring is other.ring:
            return self.ring(self.value * other.value) 
        raise ValueError("Elements must be from the same Zmod ring.")

    def __eq__(self, other):
        return isinstance(other, ZmodElement) and self.ring is other.ring and self.value == other.value

    def __hash__(self):
        """Make the object hashable using its value and ring's modulus."""
        return hash((self.value, self.ring.n))
    
    def inv(self):
        return self.ring(pow(self.value, -1, self.ring.n))
    
    def negate_inplace(self):
        self.value = (self.ring.n - self.value) % self.ring.n