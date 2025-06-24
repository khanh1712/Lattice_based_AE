import numpy as np

def encode_cipher(arrays):
    """
    arrays: list/tuple of two 1D integer arrays/lists, each value in [0,4095]
    returns: np.ndarray of bits (uint8)
    """
    arrs = [np.asarray(arr, dtype=np.int32).flatten() for arr in arrays]
    bits = []
    for arr in arrs:
        for num in arr:
            bits.extend([int(b) for b in format(int(num), '012b')])
    return np.array(bits, dtype=np.uint8)

def decode_cipher(bits, len1, len2):
    """
    bits: np.ndarray/list of bits (uint8/0-1), length = 12*(len1+len2)
    len1: length of first array
    len2: length of second array
    returns: [array1, array2] as np.int32 arrays
    """
    bits = np.array(bits, dtype=np.uint8)
    arrs = []
    pos = 0
    for l in (len1, len2):
        arr = []
        for _ in range(l):
            num = int(''.join(bits[pos:pos+12].astype(str)), 2)
            arr.append(num)
            pos += 12
        arrs.append(np.array(arr, dtype=np.int32))
    return arrs

# Example usage
if __name__ == "__main__":
    a = np.random.randint(0, 4096, size=5)
    b = np.random.randint(0, 4096, size=8)
    arrays = [a, b]
    encoded = encode_cipher(arrays)
    decoded = decode_cipher(encoded, len(a), len(b))
    print("Original:", [list(x) for x in arrays])
    print("Decoded :", [list(x) for x in decoded])
    print("Match?  :", all(np.array_equal(x, y) for x, y in zip(arrays, decoded)))
