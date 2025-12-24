import argparse
import numpy as np
import matplotlib.pyplot as plt
import wfdb


def build_sliding_matrix(x: np.ndarray, L: int) -> np.ndarray:
    N = len(x)
    if not (1 <= L <= N):
        raise ValueError("L must satisfy 1 <= L <= len(x).")
    K = N - L + 1
    return np.column_stack([x[j:j + L] for j in range(K)])


def diagonal_averaging(A: np.ndarray) -> np.ndarray:
    L, K = A.shape
    N = L + K - 1
    out = np.zeros(N, dtype=float)
    cnt = np.zeros(N, dtype=int)

    for i in range(L):
        for j in range(K):
            out[i + j] += A[i, j]
            cnt[i + j] += 1

    out /= cnt
    return out


def choose_r_by_energy(s: np.ndarray, energy_keep: float = 0.95) -> int:
    e = s**2
    cum = np.cumsum(e) / np.sum(e)
    r = int(np.searchsorted(cum, energy_keep) + 1)
    return max(1, min(r, len(s)))


def load_noise_nstdb(record: str, seconds: float | None = 10.0, channel: int = 0):
    sig, fields = wfdb.rdsamp(record, pn_dir="nstdb")
    fs = float(fields["fs"])
    if channel < 0 or channel >= sig.shape[1]:
        raise ValueError(f"channel must be between 0 and {sig.shape[1]-1}")

    x = sig[:, channel].astype(float)

    if seconds is not None:
        N = int(seconds * fs)
        N = min(N, len(x))
        x = x[:N]

    x = x - np.mean(x)
    std = np.std(x)
    if std > 1e-12:
        x = x / std

    t = np.arange(len(x)) / fs
    return x, fs, t


def load_clean_ecg_mitdb(record: str, seconds: float | None = 10.0, channel: int = 0):
    sig, fields = wfdb.rdsamp(record, pn_dir="mitdb")
    fs = float(fields["fs"])
    if channel < 0 or channel >= sig.shape[1]:
        raise ValueError(f"channel must be between 0 and {sig.shape[1]-1}")

    x = sig[:, channel].astype(float)

    if seconds is not None:
        N = int(seconds * fs)
        N = min(N, len(x))
        x = x[:N]

    x = x - np.mean(x)
    std = np.std(x)
    if std > 1e-12:
        x = x / std

    t = np.arange(len(x)) / fs
    return x, fs, t


def mix_clean_with_noise(clean: np.ndarray, noise: np.ndarray, alpha: float):
    N = min(len(clean), len(noise))
    return clean[:N] + alpha * noise[:N], N


def svd_denoise_ecg(x: np.ndarray, L: int, r: int | None = None, energy_keep: float = 0.95):
    A = build_sliding_matrix(x, L)

    U, s, VT = np.linalg.svd(A, full_matrices=False)

    if r is None:
        r = choose_r_by_energy(s, energy_keep=energy_keep)

    A_r = U[:, :r] @ np.diag(s[:r]) @ VT[:r, :]

    x_hat = diagonal_averaging(A_r)

    return x_hat, s, r


def main():
    parser = argparse.ArgumentParser(description="SVD ECG denoising: Clean ECG (mitdb) + BW noise (nstdb).")
    parser.add_argument("--ecg_record", type=str, default="118", help="Clean ECG record from mitdb (e.g., 118, 119)")
    parser.add_argument("--noise_record", type=str, default="bw", help="Noise record from nstdb: bw/em/ma")
    parser.add_argument("--seconds", type=float, default=10.0, help="How many seconds to load")
    parser.add_argument("--channel", type=int, default=0, help="Channel index (default 0)")
    parser.add_argument("--alpha", type=float, default=0.5, help="Noise mixing scale alpha (0.1–1.0 typical)")
    parser.add_argument("--L", type=int, default=100, help="Window length (samples) for sliding matrix")
    parser.add_argument("--r", type=int, default=None, help="Rank to keep (optional)")
    parser.add_argument("--energy_keep", type=float, default=0.95, help="Energy fraction when auto-choosing r")
    args = parser.parse_args()

    ecg_clean, fs_ecg, t_ecg = load_clean_ecg_mitdb(args.ecg_record, seconds=args.seconds, channel=args.channel)

    noise, fs_noise, t_noise = load_noise_nstdb(args.noise_record, seconds=args.seconds, channel=args.channel)

    if abs(fs_ecg - fs_noise) > 1e-6:
        raise ValueError(f"Sampling rate mismatch: ECG fs={fs_ecg}, noise fs={fs_noise}")

    fs = fs_ecg

    x_noisy, N = mix_clean_with_noise(ecg_clean, noise, alpha=args.alpha)
    t = np.arange(N) / fs
    ecg_clean = ecg_clean[:N]
    noise = noise[:N]

    if args.L > len(x_noisy):
        raise ValueError(f"L={args.L} is larger than signal length N={len(x_noisy)}. Reduce L or load more seconds.")

    x_denoised, s, r_used = svd_denoise_ecg(x_noisy, L=args.L, r=args.r, energy_keep=args.energy_keep)

    t_hat = np.arange(len(x_denoised)) / fs

    plt.figure(figsize=(10, 4))
    plt.plot(t, x_noisy, label=f"Original ECG")
    plt.title(f"Noisy ECG | noise={args.noise_record} | alpha={args.alpha} | fs={fs} Hz")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (normalized)")
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(10, 4))
    plt.plot(t, x_noisy, label="Original", alpha=0.6)
    plt.plot(t_hat, x_denoised, label=f"Denoised (SVD, L={args.L}, r={r_used})", alpha=0.95)
    plt.title(f"Denoising Result | L={args.L}, r={r_used}, energy_keep={args.energy_keep}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (normalized)")
    plt.legend()
    plt.tight_layout()

    print("=== Info ===")
    print("Clean ECG record (mitdb):", args.ecg_record)
    print("Noise record (nstdb):", args.noise_record)
    print("fs:", fs)
    print("Duration:", N / fs, "seconds")
    print("alpha:", args.alpha)
    print("Sliding window L:", args.L, "samples (", args.L / fs, "seconds )")
    print("Chosen rank r:", r_used)
    print("First 10 singular values:", np.round(s[:10], 6))

    plt.show()


if __name__ == "__main__":
    main()
