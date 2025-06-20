import numpy as np
import encode_signal
from GGH import GGHScheme
from scipy.fft import fft, ifft

M = 16
N = 4

log2M = int(np.log2(M))
qam_symbols = np.array([(2 * int(i % np.sqrt(M)) - np.sqrt(M) + 1) + 
                             1j * int(2 * (i // np.sqrt(M)) - np.sqrt(M) + 1) 
                             for i in range(M)])
    # qam_symbols /= np.sqrt((2/3) * (M - 1)) #!!!!!!!!!!!!!!!!!!!
qam_map = {format(i, f'0{log2M}b'): qam_symbols[i] for i in range(M)}
qam_demod_map = {v: k for k, v in qam_map.items()}

def bits_to_qam(bits):
    bit_strings = ["".join(map(str, b)) for b in bits.reshape(-1, log2M)]
    symbols = np.array([qam_map[b] for b in bit_strings])
    return symbols

def qam_to_bits(symbols):
    demapped_bits = np.array([list(qam_demod_map[min(qam_symbols, key=lambda x: abs(x - s))]) for s in symbols])
    return demapped_bits.astype(int).flatten()


bits = np.random.randint(0, 2, (N * log2M))  # Step 1: Bit Generation

qam_symbols = bits_to_qam(bits)
print(qam_symbols)

ggh = GGHScheme(N)
public_key, private_key = ggh.generate_keys()

encrypted_symbols = ggh.encrypt(qam_symbols)
print(encrypted_symbols)

encoded_symbols, pad_len = encode_signal.encode_signal(encrypted_symbols, M)
print(encoded_symbols)

ifft_symbols = ifft(encoded_symbols)
# print(ifft_symbols)
# print(len(ifft_symbols))

fft_symbols = fft(ifft_symbols)
print(fft_symbols)
# print(len(fft_symbols))

decoded_symbols = encode_signal.decode_signal(fft_symbols, N, M, pad_len)
print(decoded_symbols)