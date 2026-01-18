import numpy as np
import torch

#seed: 42 for training set and 101 for test set

class PaperDatasetGenerator:
    def __init__(self, num_samples=1000, sequence_length=30, seed=1):
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.w1 = 0.045
        self.w2 = 0.38
        self.w3 = 0.07
        self.rho = 0.65

    def generate_one_sample(self):
        t = np.arange(self.sequence_length)
        x2 = np.linspace(0, 10, self.sequence_length)
        x3 = np.linspace(0, 5, self.sequence_length)
        trend1 = 5 + 5.71 * (x2 - 1) + 0.05 * x2 - 5 * np.exp(x3) - np.exp(-4) * np.sin(x2)
        periodicity1 = 1.2 * np.sin(2 * np.pi * t / 30)
        noise1 = 0.7 * np.random.normal(0, 1, self.sequence_length)
        series1 = trend1 + periodicity1 + noise1
        series1_safe = np.maximum(series1, 0.01)
        t_safe = t + 1
        trend2 = 0.3 * (t_safe ** 0.71) + np.log(t_safe)
        periodicity2 = 3 * np.cos(np.pi * t / 30)
        noise2 = np.random.normal(0, 2, self.sequence_length)
        series2 = (self.rho * 0.5 * series1 + 0.5 * np.log(series1_safe + 1)
                   + trend2 + periodicity2 + noise2)
        factor1 = np.random.randint(1, 6)
        factor2 = np.random.uniform(15, 30)
        factor3 = np.random.randint(1, 5)
        factor4 = np.random.randint(1, 5)
        factor1_ts = np.full(self.sequence_length, factor1)
        factor2_ts = np.full(self.sequence_length, factor2)
        factor3_ts = np.full(self.sequence_length, factor3)
        factor4_ts = np.full(self.sequence_length, factor4)
        X = np.stack([
            series1,
            series2,
            factor1_ts,
            factor2_ts,
            factor3_ts,
            factor4_ts
        ], axis=1)
        y = (self.w1 * series1 * factor1
             + self.w2 * (35 - factor2) / (factor2 - 3)
             + self.w3 * series2 * factor4
             + np.random.normal(0, 0.5, self.sequence_length))
        return X, y, (factor1, factor2, factor3, factor4)

    def generate_dataset(self):
        X_list = []
        Y_list = []
        for i in range(self.num_samples):
            if i % 100 == 0:
                print(f"Generating sample {i}/{self.num_samples}")
            X, y, _ = self.generate_one_sample()
            X_list.append(X)
            Y_list.append(y)
        X = np.array(X_list)
        Y = np.array(Y_list)
        return X, Y

    def to_torch(self, X, Y):
        X_torch = torch.from_numpy(X).float()
        Y_torch = torch.from_numpy(Y).float()
        return X_torch, Y_torch


if __name__ == "__main__":
    print("Generating dataset...")
    gen = PaperDatasetGenerator(num_samples=1000, sequence_length=30, seed=42)
    X, Y = gen.generate_dataset()
    X_torch, Y_torch = gen.to_torch(X, Y)
    print(f"\n✓ Dataset generated!")
    print(f"X shape: {X_torch.shape}")
    print(f"Y shape: {Y_torch.shape}")
    print(f"\nSample #0:")
    print(f"Shape: {X_torch.shape}")
    print(f"First timestep: {X_torch[0, 0]}")
    print(f"Features: [series1, series2, factor1, factor2, factor3, factor4]")
    torch.save({'X': X_torch, 'Y': Y_torch}, 'paper_dataset.pt')
    print(f"\nSaved")
