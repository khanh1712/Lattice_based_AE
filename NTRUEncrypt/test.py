import numpy as np

class Poly:
    def __init__(self, n, coeffs):
        self.n = n
        self.coeffs = np.array(coeffs) % n
    
    def trim(self):
        idx = len(self.coeffs) - 1
        while idx > 0 and self.coeffs[idx] == 0:
            idx -= 1
        self.coeffs = self.coeffs[:idx + 1]
        return self

    def naive_multiply(self, other):
        """Naive polynomial multiplication implementation"""
        if not isinstance(other, Poly):
            return NotImplemented
        
        len_result = len(self.coeffs) + len(other.coeffs) - 1
        result = np.zeros(len_result, dtype=int)
        
        for i in range(len(self.coeffs)):
            for j in range(len(other.coeffs)):
                result[i + j] = (result[i + j] + (self.coeffs[i] * other.coeffs[j])) % self.n
                
        return Poly(self.n, result)

    def karatsuba_multiply(self, other):
        """Karatsuba multiplication implementation"""
        if not isinstance(other, Poly):
            return NotImplemented
        
        def karatsuba(a, b, cut_off=8):
            max_len = max(len(a), len(b))
            if max_len <= cut_off:
                c = np.zeros(2 * max_len - 1, dtype=int)
                for i in range(len(a)):
                    for j in range(len(b)):
                        c[i + j] = (c[i + j] + (a[i] * b[j])) % self.n
                return c

            n = 1 << (max_len - 1).bit_length()
            a = np.pad(a, (0, n - len(a)), constant_values=0)
            b = np.pad(b, (0, n - len(b)), constant_values=0)

            mid = n // 2
            a_low, a_high = a[:mid], a[mid:]
            b_low, b_high = b[:mid], b[mid:]
            
            p1 = karatsuba(a_low, b_low, cut_off)
            p3 = karatsuba(a_high, b_high, cut_off)
            
            sum_a = (a_low + a_high) % self.n
            sum_b = (b_low + b_high) % self.n
            p2 = karatsuba(sum_a, sum_b, cut_off)
            p2 = (p2 - p1 - p3) % self.n
            
            result = np.zeros(2 * n - 1, dtype=int)
            result[:len(p1)] = p1
            result[mid:mid + len(p2)] = (result[mid:mid + len(p2)] + p2) % self.n
            result[2 * mid:2 * mid + len(p3)] = (result[2 * mid:2 * mid + len(p3)] + p3) % self.n
            
            return result

        result = karatsuba(self.coeffs, other.coeffs)
        while len(result) > 1 and result[-1] == 0:
            result = result[:-1]
            
        return Poly(self.n, result)

def compare_multiplications(a_coeffs, b_coeffs, n):
    """Compare results of naive and Karatsuba multiplication"""
    a = Poly(n, a_coeffs)
    b = Poly(n, b_coeffs)
    
    naive_result = a.naive_multiply(b)
    karatsuba_result = a.karatsuba_multiply(b)
    
    print(f"\nTesting polynomials:")
    print(f"a = {a_coeffs}")
    print(f"b = {b_coeffs}")
    print(f"Modulus = {n}")
    print(f"Naive result: {naive_result.coeffs}")
    print(f"Karatsuba result: {karatsuba_result.coeffs}")
    print(f"Results match: {np.array_equal(naive_result.coeffs, karatsuba_result.coeffs)}")
    
# Test cases
test_cases = [
    # Simple cases
    ([1, 1], [1, 1], 2),  # (x+1)(x+1) mod 2
    
    # Medium cases
    ([1, 0, 1], [1, 1, 0], 2),  # (x²+1)(x²+x) mod 2
    ([1, 1, 1], [1, 0, 1], 2),  # (x²+x+1)(x²+1) mod 2
    
    # Larger cases
    ([1, 0, 1, 1], [1, 1, 0, 1], 2),  # (x³+x+1)(x³+x+1) mod 2
    ([1, 1, 1, 1], [1, 1, 1, 1], 2),  # (x³+x²+x+1)(x³+x²+x+1) mod 2
    
    # Different sizes
    ([1, 1], [1, 1, 1], 2),  # (x+1)(x²+x+1) mod 2
    ([1, 0, 1, 1], [1, 1], 2),  # (x³+x+1)(x+1) mod 2
]

print("Running comparison tests...")
for a_coeffs, b_coeffs, n in test_cases:
    compare_multiplications(a_coeffs, b_coeffs, n)