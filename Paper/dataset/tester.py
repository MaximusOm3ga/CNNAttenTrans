import torch

data = torch.load("paper_dataset.pt", map_location="cpu")
data_test = torch.load("paper_new_test.pt", map_location="cpu")

print(type(data))

print(data.keys())
print(data["X"].shape)
print(data["X"][0], data["Y"][0])

print(data_test.keys())
print(data_test["X"].shape)
print(data_test["X"][0], data["Y"][0])