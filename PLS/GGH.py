import numpy as np
from numpy.linalg import solve

class GGHScheme:
    def __init__(self, n: int, sigma: float = 3.0, l: int = 4):
        self.n = n
        self.sigma = sigma
        self.l = l if l is not None else int(np.sqrt(n) + 1)
        self.k = self.l * int(np.sqrt(n) + 1)

        self.R = None  # Private basis
        self.T = None  # Transformation matrix
        self.B = None  # Public basis

    def generate_perturbation_matrix(self) -> np.ndarray:
        max_val = min(self.l, 4)
        return np.random.randint(-max_val, max_val + 1, size=(self.n, self.n), dtype=np.int32)

    def generate_unimodular_matrix(self) -> np.ndarray:
        """Generate a random unimodular matrix."""
        T = np.eye(self.n, dtype=np.int32)
        for _ in range(10 * self.n):
            i, j = np.random.randint(0, self.n, 2)
            if i != j:
                T[j] += np.random.choice([-1, 1]) * T[i]
        return T

    def generate_keys(self):
        """Generate private basis R and public basis B."""
        P = self.generate_perturbation_matrix()
        I = np.eye(self.n, dtype=np.int32)
        self.R = (self.k * I + P).astype(np.int32) 

        self.T = self.generate_unimodular_matrix()
        self.B = np.dot(self.R, self.T).astype(np.int32)

        return self.B, (self.R, self.T)

    def encrypt(self, message: np.ndarray) -> np.ndarray:
        """Encrypt a complex message using the public basis B."""
        if message.ndim == 1:
            message = message.reshape(-1, 1)

        real_part = message.real
        imag_part = message.imag  # Fixed typo

        # max_val = min(self.l, 5)
        max_val = 1
        r_real = np.random.randint(-max_val, max_val + 1, size=(self.n, 1), dtype=np.int32)
        r_imag = np.random.randint(-max_val, max_val + 1, size=(self.n, 1), dtype=np.int32)

        e_real = np.dot(self.B, real_part) + r_real  
        e_imag = np.dot(self.B, imag_part) + r_imag
        # print("Error real\n", r_real)
        # print("Error imag\n", r_imag)
        return e_real + 1j * e_imag  # Ensure result is a complex matrix

    def decrypt(self, e: np.ndarray) -> np.ndarray:
        """Decrypt the ciphertext e using the private basis R."""
        real_enc = e.real.round().astype(np.int32)
        imag_enc = e.imag.round().astype(np.int32)
        # print(real_enc)
        # print(imag_enc)
        approx_real = solve(self.R, real_enc).round().astype(np.int32)
        approx_imag = solve(self.R, imag_enc).round().astype(np.int32)

        m_real = solve(self.T, approx_real).round().astype(np.int32)
        m_imag = solve(self.T, approx_imag).round().astype(np.int32)

        return m_real + 1j * m_imag  # Reconstruct original complex message

# ----------------- TESTING THE SCHEME -----------------
def main():
    np.random.seed(42)  # Fix seed for reproducibility

    n = 350  # Reduced dimension for testing
    sigma = 3.0
    l = 4  

    try:
        ggh = GGHScheme(n=n, sigma=sigma, l=l)

        # Generate keys
        public_key, private_key = ggh.generate_keys()

        print("\n🔑 Public Key B:")
        print(public_key)

        # ---- Encrypting and Decrypting a Complex Message ----
        message = np.random.randint(-100, 100 + 1, size=(n, 1)) + 1j * np.random.randint(-100, 100 + 1, size=(n, 1))
        print("\n✉️  Original Complex Message:")
        print(message)

        encrypted_message = ggh.encrypt(message)
        print("\n🔒 Encrypted Message:")
        print(encrypted_message)

        decrypted_message = ggh.decrypt(encrypted_message)
        print("\n🔓 Decrypted Message:")
        print(decrypted_message)

        # Verification
        print("\n✅ Decryption Successful?", np.all(message == decrypted_message))

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()