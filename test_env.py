import torch
import torch_geometric
import scipy
import sklearn
import numpy as np

print("=" * 50)
print("Environment Verification")
print("=" * 50)

print(f"\nPyTorch version: {torch.__version__}")
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
print(f"PyTorch Geometric version: {torch_geometric.__version__}")
print(f"SciPy version: {scipy.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
print(f"NumPy version: {np.__version__}")

print("\n" + "=" * 50)
print("Testing PyTorch tensor operations...")
print("=" * 50)

x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])
z = x + y
print(f"Tensor addition: {z}")

print("\n" + "=" * 50)
print("Testing PyTorch Geometric...")
print("=" * 50)

from torch_geometric.data import Data
edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
data = Data(x=x, edge_index=edge_index)
print(f"Graph data created: {data}")
print(f"Number of nodes: {data.num_nodes}")
print(f"Number of edges: {data.num_edges}")

print("\n" + "=" * 50)
print("All tests passed! Environment is ready.")
print("=" * 50)