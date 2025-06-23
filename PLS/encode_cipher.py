import numpy as np

def encode_cipher(arrays):
    # arrays: list/tuple of two 1D integer arrays, different lengths allowed
    lens = [len(arr) for arr in arrays]
    bits = []
    for arr in arrays:
        for num in arr:
            bits.extend([int(b) for b in format(num, '012b')])
    # Prepend lengths for decoding
    header = []
    for l in lens:
        header.extend([int(b) for b in format(l, '032b')])  # 32 bits for length (supports up to 2^32-1)
    return np.array(header + bits, dtype=np.uint8)

def decode_cipher(bits):
    bits = np.array(bits, dtype=np.uint8)
    # Extract lengths from the first 64 bits (2 x 32 bits)
    len1 = int(''.join(bits[:32].astype(str)), 2)
    len2 = int(''.join(bits[32:64].astype(str)), 2)
    data_bits = bits[64:]
    # Split into two arrays
    arrs = []
    pos = 0
    for l in (len1, len2):
        arr = []
        for _ in range(l):
            num = int(''.join(data_bits[pos:pos+12].astype(str)), 2)
            arr.append(num)
            pos += 12
        arrs.append(np.array(arr, dtype=np.int32))
    return arrs

# Example usage:
if __name__ == "__main__":
    a = np.random.randint(0, 4096, size=5)
    b = np.random.randint(0, 4096, size=20)
    arrays = [a, b]
    encoded = encode_cipher(arrays)
    decoded = decode_cipher(encoded)
    print("Original:", [list(x) for x in arrays])
    print(encoded)
    print("Decoded :", [list(x) for x in decoded])
    print("Match?  :", all(np.array_equal(x, y) for x, y in zip(arrays, decoded)))
