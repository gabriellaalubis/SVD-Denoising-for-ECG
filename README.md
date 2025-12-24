# ECG Denoising Using Singular Value Decomposition (SVD)

This repository implements **ECG signal denoising using Singular Value Decomposition (SVD)**.
The experiment uses **clean ECG signals from MIT-BIH Arrhythmia Database (mitdb)** and
**baseline wander / muscle / electrode motion noise from the Noise Stress Test Database (nstdb)**.

---

## Key Features

- Uses **real ECG and noise data** from PhysioNet
- Sliding window (trajectory matrix) construction
- Automatic rank selection based on **energy preservation**
- Reconstruction via **diagonal averaging**
- Fully reproducible and configurable via command line

---

## Data Sources (PhysioNet)

- **Clean ECG**: MIT-BIH Arrhythmia Database  
  https://physionet.org/content/mitdb/

- **Noise**: Noise Stress Test Database (NSTDB)  
  https://physionet.org/content/nstdb/1.0.0/

> No manual download required — data is fetched automatically using `wfdb`.

---

## Experimental Setup

| Parameter            | Value                          |
|---------------------|--------------------------------|
| Sampling rate (fs)  | 360 Hz                         |
| Channels            | 2 (only one used)              |
| Signal duration     | First 10 seconds               |
| Noise type          | BW / EM / MA                   |
| Amplitude unit      | Normalized (zero mean, unit std) |

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/gabriellaalubis/SVD-Denoising-for-ECG.git
cd src
pip install -r requirements.txt
python denoise.py
