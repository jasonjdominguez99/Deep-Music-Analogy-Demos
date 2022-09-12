# generate.py
#
# source code for generating music, using a trained
# EC2-VAE model

# imports
import json
import torch
from ec_squared_vae import ECSquaredVAE

# function definitions
def load_ec_squared_vae(config_file_path):
    with open(config_file_path) as f:
        args = json.load(f)
    
    load_path = "ec_squared_vae/params/{}.pt".format(args["name"])

    model = ECSquaredVAE(
        args["roll_dim"], args["hidden_dim"], args["rhythm_dim"], 
        args["condition_dims"], args["z1_dim"],
        args["z2_dim"], args["time_step"]
    )
    # remove module. from start of state dict keys
    from collections import OrderedDict
    state_dict = torch.load(load_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove module.
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)

    return model

def main():
    config_file_path = "ec_squared_vae/code/ec_squared_vae_model_config.json"
    model = load_ec_squared_vae(config_file_path)
    print("Loaded!")

if __name__ == "__main__":
    main()