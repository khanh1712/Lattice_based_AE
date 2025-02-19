import numpy as np
from scipy.fft import fft, ifft
import GGH
from time import time

start = time()

N = 16
M = 16
log2M = int(np.log2(M))
snr = 60

ggh = GGH.GGHScheme(N)
public_key, private_key = ggh.generate_keys()

noise_var = 10**(-snr / 10) / (N*log2M)

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


errors = 0
m = 50
for _ in range(m):
    bits = np.random.randint(0, 2, (N * log2M))  # Step 1: Bit Generation
    qam_symbols = bits_to_qam(bits)  # Step 3: QAM Mapping
    print("Original symbols\n", qam_symbols)

    encrypted_symbols = ggh.encrypt(qam_symbols)
    encrypted_symbols = np.reshape(encrypted_symbols, newshape=(1,N))
    print("Encrypted symbols\n", encrypted_symbols)

    ofdm_symbols = ifft(encrypted_symbols)  # Step 4: IFFT
    print("After IFFT\n", ofdm_symbols)
    noise = np.sqrt(noise_var / 2) * (np.random.randn(N) + 1j * np.random.randn(N))
    # noise = np.sqrt(noise_var / 2) * np.ndarray(np.random.randn(N) + 1j * np.random.randn(N))

    received = ofdm_symbols + noise  # Step 5: AWGN Channel
    # received = ofdm_symbols
    print("AWGN channel\n", received)

    received_qam = fft(received)  # Step 6: FFT
    received_qam = np.reshape(received_qam, newshape=(N,1))
    print("After FFT\n", received_qam)

    decrypted_symbols = ggh.decrypt(received_qam)
    # decrypted_symbols = ggh.decrypt(np.reshape(encrypted_symbols, newshape=(N,1)))
    print("Decrypted symbols\n", decrypted_symbols)

    received_bits = qam_to_bits(decrypted_symbols)  # Step 7: QAM Demapping
    print("Sent bits: ", bits)
    print("Received bits: ", received_bits)
    print("Number of wrong bits: ", np.sum(bits != received_bits))
    errors += np.sum(bits != received_bits)

end = time()
print("Duration: ", end - start)
print("BER: ", errors/(N*log2M*m))