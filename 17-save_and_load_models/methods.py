import torch
import torch.nn as nn

# there are 3 methods
torch.save(arg, PATH) # can use tensors, models or any dictionaries as parameter for saving

torch.load(PATH)

model.load_state_dict(arg)