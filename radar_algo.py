import numpy as np
import matplotlib.pyplot as plt

# class Radar:

def generate_lfm(fs = 100e6, T = 1e-4, bw = 5e5, f0 = 0):
    mu = bw / T
    N = int(T * fs)      #sec * samples / sec = samples
    tt = np.arange(N) / fs
    return np.exp(1j * 2 * np.pi * (f0 * tt + (mu/2) * (tt ** 2)))

def apply_delay(lfm, delay_samples):
    rx = np.zeros_like(lfm)

    rx[delay_samples:] = lfm[:-delay_samples]

    return rx

def matched_filter(tx: np.ndarray, rx: np.ndarray):
    h = np.conj(tx[::-1])       #impulse response is the conjugate of the output waveform reversed
    return np.convolve(rx, h, mode = 'same')

def apply_doppler(tgt, f_d, fs = 100e6, T = 1e-4):
    N = int(T * fs)      #sec * samples / sec = samples
    tt = np.arange(N) / fs
    return tgt * np.exp(1j * 2 * np.pi * f_d * tt)

def simulate_pulse(tx, targets, k, PRI, fs):
    """
    k = pulse index
    PRI = pulse repetition interval
    """

    N = len(tx)
    rx = np.zeros(N, dtype=complex)

    t_fast = np.arange(N) / fs

    for tgt in targets:
        delay = tgt["delay"]
        amp = tgt["amp"]
        fd = tgt["fd"]

        if delay < N:
            # Doppler evolves across pulses (slow time)
            phase = np.exp(1j * 2 * np.pi * fd * k * PRI)

            rx[delay:] += amp * tx[:N-delay] * phase

    # noise
    noise = 0.05 * (np.random.randn(N) + 1j*np.random.randn(N))
    rx += noise

    return rx


#test
fs = 100e6
c = 299792458
tx = generate_lfm()
tx_delayed = apply_delay(tx, 200)

fig, axs = plt.subplots(3, 1)
noise = np.random.normal(0, 0.05, len(tx))

rx = tx_delayed + noise
correlation = matched_filter(tx, rx)

range_axis = np.arange(len(correlation)) * c / (2 * fs)

axs[0].plot(np.real(tx))
axs[0].set_title('TX chirp')
axs[1].plot(np.real(rx))
axs[1].set_title('Echo (TX delayed by 200 samples)')
axs[2].plot(range_axis, np.real(correlation))
axs[2].set_title('Correlation of the TX chirp and its echo')
plt.show()

#multiple targets
tx = generate_lfm()
noise = np.random.normal(0, 0.05, len(tx))


tgt1 = apply_delay(tx, 200)
tgt2 = apply_delay(tx, 400)
tgt3 = apply_delay(tx, 900)

rx = tgt1 + tgt2 + tgt3 + noise

correlation = matched_filter(tx, rx)

range_axis = np.arange(len(correlation)) * c / (2 * fs)

fig, ax = plt.subplots(3, 1)

ax[0].plot(np.real(tx))
ax[0].set_title('TX chirp')
ax[1].plot(np.real(rx))
ax[1].set_title('Echo (3 targets, 200, 400, and 900 sample delay)')
ax[2].plot(range_axis, 20 * np.log10(np.abs(correlation) + 1e-12))
ax[2].set_title('Correlation of the TX chirp and its echo')
plt.show()

#doppler
fs = 100e6
PRI = 1e-3      # pulse repetition interval (1 kHz PRF)
num_pulses = 64

tx = generate_lfm(fs=fs)

targets = [
    {"delay": 200, "amp": 1.0, "fd": 50},   # slow target
    {"delay": 400, "amp": 0.7, "fd": 150},  # medium velocity
    {"delay": 900, "amp": 0.5, "fd": 300},  # fast target
]
range_profiles = []

for k in range(num_pulses):
    rx = simulate_pulse(tx, targets, k, PRI, fs)
    mf = matched_filter(tx, rx)

    range_profiles.append(np.abs(mf))

range_profiles = np.array(range_profiles)

rd_map = np.fft.fftshift(np.fft.fft(range_profiles, axis=0), axes=0)
rd_map = 20 * np.log10(np.abs(rd_map) + 1e-12)

plt.figure(figsize=(8, 6))
plt.imshow(
    rd_map,
    aspect="auto",
    cmap="jet",
    origin="lower"
)

plt.title("Range-Doppler Map")
plt.xlabel("Range bin")
plt.ylabel("Doppler bin")
plt.colorbar(label="dB")
plt.show()