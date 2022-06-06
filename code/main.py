# main.py
#
# main source code for training and saving the
# EC^2 VAE model


# imports
import json
import os

from ec_squared_vae import ECSquaredVAE
from utils import (
    MinExponentialLR, std_normal, loss_function
)
from data_loader import MusicArrayLoader

import numpy as np

import torch
from torch import optim
from torch.distributions import Normal
from tensorboardX import SummaryWriter


# function definitions and implementations
def configure_model(config_file_path):
    with open(config_file_path) as f:
        args = json.load(f)

    if not os.path.isdir("log"):
        os.mkdir("log")

    if not os.path.isdir("params"):
        os.mkdir("params")

    save_path = "../params/{}.pt".format(args["name"])
    writer = SummaryWriter("../log/{}".format(args["name"]))

    model = ECSquaredVAE(
        args["roll_dim"], args["hidden_dim"], args["rhythm_dim"], 
        args["condition_dim"], args["pitch_dim"],
        args["rhythm_dim"], args["time_step"]
    )

    if args["if_parallel"]:
        model = torch.nn.DataParallel(model, device_ids=[0, 1])
    optimizer = optim.Adam(model.parameters(), lr=args["lr"])

    if args["decay"] > 0:
        scheduler = MinExponentialLR(
            optimizer, gamma=args["decay"], minimum=1e-5
        )

    if torch.cuda.is_available():
        print(
            "Using: ",
            torch.cuda.get_device_name(torch.cuda.current_device())
        )
        model.cuda()
    else:
        print("CPU mode")

    step, pre_epoch = 0, 0
    model.train()

    dl = MusicArrayLoader(args["data_path"], args["time_step"], 16)
    dl.chunking()

    return (model, args, save_path, writer, 
            scheduler, step, pre_epoch, dl, optimizer)


def train(model, args, writer, scheduler, step, dl, optimizer):
    batch, c = dl.get_batch(args["batch_size"])
    print(batch.shape, c.shape)
    encode_tensor = torch.from_numpy(batch).float()
    c = torch.from_numpy(c).float()

    rhythm_target = np.expand_dims(batch[:, :, :-2].sum(-1), -1)
    rhythm_target = np.concatenate((rhythm_target, batch[:, :, -2:]), -1)
    rhythm_target = torch.from_numpy(rhythm_target).float()
    rhythm_target = rhythm_target.view(
        -1, rhythm_target.size(-1)
    ).max(-1)[1]
    target_tensor = encode_tensor.view(
        -1, encode_tensor.size(-1)
    ).max(-1)[1]

    if torch.cuda.is_available():
        encode_tensor = encode_tensor.cuda()
        target_tensor = target_tensor.cuda()
        rhythm_target = rhythm_target.cuda()
        c = c.cuda()

    optimizer.zero_grad()
    recon, recon_rhythm, dis1m, dis1s, dis2m, dis2s = model(encode_tensor, c)
    distribution_1 = Normal(dis1m, dis1s)
    distribution_2 = Normal(dis2m, dis2s)

    loss = loss_function(
        recon,
        recon_rhythm,
        target_tensor,
        rhythm_target,
        distribution_1,
        distribution_2,
        step,
        beta=args["beta"]
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    optimizer.step()
    step += 1

    print("batch loss: {:.5f}".format(loss.item()))
    writer.add_scalar("batch_loss", loss.item(), step)
    if args["decay"] > 0:
        scheduler.step()
    dl.shuffle_samples()

    return step


def main():
    config_fname = "ec_squared_vae_model_config.json"

    (model, args, save_path, writer, scheduler,
     step, pre_epoch, dl, optimizer) = configure_model(config_fname)

    while dl.get_n_epoch() < args["n_epochs"]:
        step = train(model, args, writer, scheduler, step, dl, optimizer)
        if dl.get_n_epoch() != pre_epoch:
            pre_epoch = dl.get_n_epoch()
            torch.save(model.cpu().state_dict(), save_path)

            if torch.cuda.is_available():
                model.cuda()
            print("Model saved!")

if __name__ == "__main__":
    main()
