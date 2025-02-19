# multiple value of N and M

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
from time import time
import GGH
# Simulation Parameters
N_values = [16]  # Different values of N
M_values = [16]  # QAM Modulation orders (64-QAM, 256-QAM)
num_bits = 10**5  # Reduced to speed up simulation
SNR_dB = np.arange(0, 61, 5)  # SNR range in dB


plt.figure(figsize=(8, 6))

start = time()

for M in M_values:
    log2M = int(np.log2(M))
    qam_symbols = np.array([(2 * (i % np.sqrt(M)) - np.sqrt(M) + 1) + 
                             1j * (2 * (i // np.sqrt(M)) - np.sqrt(M) + 1) 
                             for i in range(M)])
    qam_symbols /= np.sqrt((2/3) * (M - 1)) #!!!!!!!!!!!!!!!!!!!
    qam_map = {format(i, f'0{log2M}b'): qam_symbols[i] for i in range(M)}
    qam_demod_map = {v: k for k, v in qam_map.items()}

    def bits_to_qam(bits):
        bit_strings = ["".join(map(str, b)) for b in bits.reshape(-1, log2M)]
        symbols = np.array([qam_map[b] for b in bit_strings])
        return symbols

    def qam_to_bits(symbols):
        demapped_bits = np.array([list(qam_demod_map[min(qam_symbols, key=lambda x: abs(x - s))]) for s in symbols])
        return demapped_bits.astype(int).flatten()

    for N in N_values:
        num_blocks = num_bits // (N * log2M)

        ggh = GGH.GGHScheme(N)
        public_key, private_key = ggh.generate_keys()

        ber = []
        
        for snr in SNR_dB:
            errors = 0
            noise_var = 10**(-snr / 10) / (N*log2M) #!!!!!!!!!!!!!!!!!!!!!!
            
            for _ in range(num_blocks):
                # bits = np.random.randint(0, 2, (N * log2M))  # Step 1: Bit Generation
                
                # qam_symbols = bits_to_qam(bits)  # Step 3: QAM Mapping
                
                # encrypted_symbols = ggh.encrypt(qam_symbols)
                # encrypted_symbols = np.reshape(encrypted_symbols, newshape=(1,N))
                
                # ofdm_symbols = ifft(encrypted_symbols)  # Step 4: IFFT
                
                # noise = np.sqrt(noise_var / 2) * (np.random.randn(N) + 1j * np.random.randn(N))
                
                # received = ofdm_symbols + noise  # Step 5: AWGN Channel
                
                # received_qam = fft(received)  # Step 6: FFT
                # received_qam = np.reshape(received_qam, newshape=(N,1))

                # decrypted_symbols = ggh.decrypt(received_qam)

                # received_bits = qam_to_bits(decrypted_symbols)  # Step 7: QAM Demapping
                bits = np.random.randint(0, 2, (N * log2M))  # Step 1: Bit Generation
                qam_symbols = bits_to_qam(bits)  # Step 3: QAM Mapping
                # print("Original symbols\n", qam_symbols)

                encrypted_symbols = ggh.encrypt(qam_symbols)
                encrypted_symbols = np.reshape(encrypted_symbols, newshape=(1,N))
                # print("Encrypted symbols\n", encrypted_symbols)

                ofdm_symbols = ifft(encrypted_symbols)  # Step 4: IFFT
                # print("After IFFT\n", ofdm_symbols)
                noise = np.sqrt(noise_var / 2) * (np.random.randn(N) + 1j * np.random.randn(N))
                # noise = np.sqrt(noise_var / 2) * np.ndarray(np.random.randn(N) + 1j * np.random.randn(N))

                received = ofdm_symbols + noise  # Step 5: AWGN Channel
                # print("AWGN channel\n", received)

                received_qam = fft(received)  # Step 6: FFT
                received_qam = np.reshape(received_qam, newshape=(N,1))
                # print("After FFT\n", received_qam)

                decrypted_symbols = ggh.decrypt(received_qam)
                # print("Decrypted symbols\n", decrypted_symbols)

                received_bits = qam_to_bits(decrypted_symbols)  # Step 7: QAM Demapping

                errors += np.sum(bits != received_bits)
            
            ber.append(errors / num_bits)
        
        plt.semilogy(SNR_dB, ber, 'o-', label=f'M={M}, N={N}')

end = time()
print("Duration: ", end - start)
plt.xlabel('SNR (dB)')
plt.ylabel('Bit Error Rate (BER)')
plt.title('SNR vs BER for QAM-OFDM System with Different M and N')
plt.grid(True, which='both', linestyle='--')
plt.legend()
plt.show()
