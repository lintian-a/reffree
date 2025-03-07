
import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from torch.nn import L1Loss
from torch.utils.data import DataLoader
import footsteps
from datetime import datetime
import os
import argparse

from tqdm import tqdm
import shutil

# from networks import UNet2
import sys
# sys.path.append("/autofs/space/tayswift_001/users/lt456/projects/pytorch-3dunet")
from pytorch3dunet.unet3d.model import UNet2D

from reffree.datasets import PhotoSynthReconTaskGenerator
import reffree.utils as utils


def train_kernel(data, task_generator, net, optimizer, writer, ite):
    # Generate the task
    with torch.no_grad():
        data = [d.to('cuda') for d in data]
        image, mask, mni, slice_id = task_generator.get_task_data(*data)

    try:
        optimizer.zero_grad()

        image_shape = image.shape[2:]
        input = torch.cat([image,
                        slice_id[:, None, None, None].expand(-1, -1, *image_shape)], dim=1)
        pred = net(input)

        assert torch.sum(torch.isnan(pred)) == 0, "NaN in prediction"
        assert torch.sum(torch.isnan(mni)) == 0, "NaN in target"
        assert torch.sum(mask) > 0, "Mask is empty"
        
        # MSE loss
        # mask = mask.float()
        # loss = torch.sum(((pred - mni)**2)*mask)/torch.sum(mask)

        # L1 loss
        mask = mask.bool().expand(-1, 3, -1, -1)
        loss = L1Loss()(pred[mask], mni[mask])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()

    except Exception as e:
        # Figure out the reason causing the Inf norm of the gradient.
        print(f"Error at iteration {ite}: {e}")
        utils.make_dir(footsteps.output_dir + "/debug/")
        debug_data = {
            "image": image,
            "mask": mask,
            "mni": mni,
            "slice_id": slice_id
        }
        utils.slice_training(net, optimizer, debug_data, ite, footsteps.output_dir + "/debug/")
        raise e

    writer.add_scalar("train/disp_loss_sq", loss.item(), ite)

def val_kernel(data, task_generator, net, writer, ite):
    with torch.no_grad():
        data = [d.to('cuda') for d in data]
        image, mask, mni, slice_id = task_generator.get_task_data(*data)

        image_shape = image.shape[2:]
        input = torch.cat([image,
                        slice_id[:, None, None, None].expand(-1, -1, *image_shape)], dim=1)
        pred = net(input)

        # MSE loss
        # mask = mask.float()
        # loss = torch.sum(((pred - mni)**2)*mask)/torch.sum(mask)

        # L1 loss
        mask = mask.bool().expand(-1, 3, -1, -1)
        loss = L1Loss()(pred[mask], mni[mask])

        pred = (pred-torch.min(pred))/(torch.max(pred)-torch.min(pred))
        pred = pred*mask.float()
        writer.add_images("debug/pred_disp_x", pred[:,0:1,:,:], ite)
        writer.add_images("debug/pred_disp_y", pred[:,1:2,:,:], ite)
        writer.add_images("debug/pred_disp_z", pred[:,2:,:,:], ite)
        writer.add_scalar("debug/disp_loss_sq", loss.item(), ite)

###################################################
# Train
###################################################

parser = argparse.ArgumentParser(
                    prog='refree',
                    description='Reference free registration')
parser.add_argument("-e", "--exp_name", type=str, default="test", help="Experiment name.")
parser.add_argument("--exp_folder", type=str, 
                    default="/autofs/space/tayswift_001/users/lt456/projects/photoRecon/results", 
                    help="Directory to save the experiment.")
parser.add_argument("--batch_size", type=int, 
                    default=2, 
                    help="Batch size.")
parser.add_argument("--slice_num", type=int, 
                    default=64, 
                    help="Number of slice used per volume.")
parser.add_argument("--epochs", type=int, 
                    default=8, 
                    help="Number of epochs to train.")
parser.add_argument("--train_config", type=str, 
                    required=True,
                    default="", 
                    help="The path to the config file. It should be a python file.")
parser.add_argument("--resume", type=str, default=None, help="Path to the checkpoint to resume from.")
args = parser.parse_args()

utils.set_seed_for_demo()
torch.backends.cudnn.deterministic = True
torch.autograd.set_detect_anomaly(True)

footsteps.initialize(run_name=args.exp_name, output_root=args.exp_folder)

# Load training config
assert args.train_config is not None, "Need specify train config python file."
train_config = utils.path_import(args.train_config)

# Save training config
shutil.copy(args.train_config, f"{footsteps.output_dir}/train_config.py")


net = UNet2D(2, 3, final_sigmoid=False, f_maps=128, layer_order='gcl',  num_groups=8, num_levels=5, is_segmentation=False).cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

if args.resume is not None:
    print(f"Resuming from {args.resume}")
    net.load_state_dict(torch.load(args.resume))
    epoch = int(args.resume.split("/")[-1].split("_")[-1])
    opt_dict = torch.load("/".join(args.resume.split("/")[:-1])+f"/optimizer_weight_{epoch}")
    optimizer.load_state_dict(opt_dict["optimizer_state_dict"])
    EPOCH_FROM = opt_dict["epoch"] + 1
    ite = opt_dict["ite"] + 1
else:
    ite = 0
    epoch = 0
    EPOCH_FROM = 0

train_dataset, debug_dataset = train_config.get_train_dataset(), train_config.get_debug_dataset()

train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_dataset.sampler, num_workers=2, pin_memory=True, drop_last=True, persistent_workers=True)
debug_data_loader = DataLoader(debug_dataset, batch_size=2, shuffle=False, num_workers=2)

task_generator = PhotoSynthReconTaskGenerator(in_shape=train_config.in_shape, slice_shape=train_config.slice_shape, slices_per_volume=args.slice_num, synth_config=train_config.synth_config, device='cuda')


# Train settings
EPOCHS =  EPOCH_FROM + args.epochs
MODEL_SAVE_FRE = 5
MODEL_DEBUG_FRE = 1
TENSORBOARD_FRE = 1

writer = SummaryWriter(
        footsteps.output_dir + "/logs/" + datetime.now().strftime("%Y%m%d-%H%M%S"),
        flush_secs=300,
    )

os.mkdir(footsteps.output_dir + "checkpoints/")

for epoch in range(EPOCH_FROM, EPOCHS):
    print(f"### Epoch {epoch}")
    # training
    for data in tqdm(train_data_loader):
        train_kernel(data, task_generator, net, optimizer, writer, ite)
        ite += 1

    # validation
    if epoch % MODEL_DEBUG_FRE == 0:
        data = next(iter(debug_data_loader))
        net.eval()
        val_kernel(data, task_generator, net, writer, epoch)
        net.train()
    
    # save model
    if epoch % MODEL_SAVE_FRE == 0:
        torch.save({
            "optimizer_state_dict": optimizer.state_dict(),
            "ite": ite,
            "epoch": epoch
            },
            footsteps.output_dir + "checkpoints/optimizer_weight_" + str(epoch)
        )

        torch.save(
            net.state_dict(),
            footsteps.output_dir + "checkpoints/net_weight_" + str(epoch)
        )

# save model
torch.save({
    "optimizer_state_dict": optimizer.state_dict(),
    "ite": ite,
    "epoch": epoch
    },
    footsteps.output_dir + "checkpoints/optimizer_weight_" + str(epoch)
)

torch.save(
    net.state_dict(),
    footsteps.output_dir + "checkpoints/net_weight_" + str(epoch)
)

writer.close()
