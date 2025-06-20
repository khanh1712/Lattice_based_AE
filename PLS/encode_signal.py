import numpy as np

def int_to_bits(x, width):
    return np.array([int(b) for b in np.binary_repr(x, width=width)], dtype=np.uint8)

def bits_to_int(bits):
    return int(''.join(bits.astype(str)), 2)

def qam_modulate(bits, M):
    k = int(np.log2(M))
    num_symbols = len(bits) // k
    bits = bits[:num_symbols * k]
    symbols = bits.reshape((num_symbols, k))
    m = int(np.sqrt(M))
    I_bits = symbols[:, :k//2]
    Q_bits = symbols[:, k//2:]
    I = np.array([bits_to_int(x) for x in I_bits])
    Q = np.array([bits_to_int(x) for x in Q_bits])
    norm = (m - 1)
    I = 2 * I - norm
    Q = 2 * Q - norm
    return I + 1j * Q

def qam_demodulate(symbols, M):
    m = int(np.sqrt(M))
    k = int(np.log2(M))
    norm = (m - 1)
    I = np.round((np.real(symbols) + norm) / 2).astype(int)
    Q = np.round((np.imag(symbols) + norm) / 2).astype(int)
    I = np.clip(I, 0, m-1)
    Q = np.clip(Q, 0, m-1)
    bits = []
    for i, q in zip(I, Q):
        bits.extend(int_to_bits(i, k//2))
        bits.extend(int_to_bits(q, k//2))
    return np.array(bits, dtype=np.uint8)

def encode_signal(input_array, M):
    N = len(input_array)
    bits = []
    for z in input_array:
        a = int(np.round(np.real(z)))
        b = int(np.round(np.imag(z)))
        bits.extend(int_to_bits(a, 12))
        bits.extend(int_to_bits(b, 12))
    bits = np.array(bits, dtype=np.uint8)
    k = int(np.log2(M))
    pad_len = (-len(bits)) % k
    if pad_len > 0:
        bits = np.concatenate([bits, np.zeros(pad_len, dtype=np.uint8)])
    # return both the encoded signal and the number of padding bits
    return qam_modulate(bits, M), pad_len

def decode_signal(encoded_array, N, M, pad_len):
    k = int(np.log2(M))
    total_bits = N * 24
    bits = qam_demodulate(encoded_array, M)
    # Remove padding at the end, if any
    if pad_len > 0:
        bits = bits[:-pad_len]
    if len(bits) < total_bits:
        bits = np.concatenate([bits, np.zeros(total_bits - len(bits), dtype=np.uint8)])
    bits = bits[:total_bits]
    output = []
    for i in range(N):
        a_bits = bits[i*24 : i*24+12]
        b_bits = bits[i*24+12 : i*24+24]
        a = bits_to_int(a_bits)
        b = bits_to_int(b_bits)
        output.append(complex(a, b))
    return np.array(output)

# M = 16
# input_array = np.random.randint(0, 4096, size=12) + 1j*np.random.randint(0, 4096, size=12)
# encoded, pad_len = encode_signal(input_array, M)
# decoded = decode_signal(encoded, len(input_array), M, pad_len)
# print("Original: ", input_array)
# print("Encoded: ", encoded)
# print("Decoded:  ", decoded)
