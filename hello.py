import torch



# TODO: perhaps let the user decide the 'device' in the future
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# the user needs to provide a "torch.utils.data.Dataset" object

# load the dataset
