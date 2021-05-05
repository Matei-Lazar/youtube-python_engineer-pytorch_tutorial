import torch
import torch.nn as nn

### COMPLETE MODEL ###
torch.save(model, PATH)

# model class must be defined somewhere
model = torch.load(PATH)
model.eval()

### STATE DICT ###
torch.save(model.state_dict(), PATH) # state dict holds the parameters

# model must be created again with parameters; this is the recommended way
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.evel() # sets the model to evaluation mode or something like that
