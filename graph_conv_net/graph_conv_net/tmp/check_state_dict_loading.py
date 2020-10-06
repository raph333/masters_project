
# CHECK LOGGING OF MODEL PARAMETERS
import torch
from graph_conv_net import tencent_mpnn

file_path = '../../experiments/mlruns/1/00d56bb0f57f4391a1015fa628e72cc1/artifacts/state_dict.pt'

model = tencent_mpnn.MPNN(node_input_dim=15,
                          edge_input_dim=5)

print([p for p in model.parameters()][0][0:1])
model.load_state_dict(torch.load(file_path))
print([p for p in model.parameters()][0][0:1])
