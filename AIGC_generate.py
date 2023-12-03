# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse
import os
import cv2
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:                 
        os.makedirs(path)

def main(args):
    # Setup PyTorch:
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    data_dir = args.save_dir
    if not data_dir:
        data_dir = './data/step'+str(args.num_sampling_steps)+'_s'+str(args.cfg_scale).replace('.','_')+'_seed_'+str(args.start_seed)+'-'+str(args.end_seed)+'/'
    mkdir(data_dir)
    for class_idx in range(args.start_id, args.end_id):  # 0-99类，100-199类，类别
        print('generating class: ' + str(class_idx) + ' at ' +data_dir)
        for seed_idx in range(args.start_seed, args.end_seed):

            path = data_dir + "class_"+str(class_idx)+"_seed_"+str(int(seed_idx))+".png"
            if os.path.exists(path):
                continue
            if os.path.exists(path):
                print('skip path:'+str(path))
            else:
                # print('generating:'+str(path))
                torch.manual_seed(seed_idx)
                
                class_labels=[class_idx]*10
                # print(class_idx)

                # Create sampling noise:
                n = len(class_labels)
                z = torch.randn(n, 4, latent_size, latent_size, device=device)
                y = torch.tensor(class_labels, device=device)

                # Setup classifier-free guidance:
                z = torch.cat([z, z], 0)
                y_null = torch.tensor([1000] * n, device=device)
                y = torch.cat([y, y_null], 0)
                model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

                # Sample images:
                samples = diffusion.p_sample_loop(
                    model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
                )
                samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
                samples = vae.decode(samples / 0.18215).sample

                for i in range(samples.shape[0]):
                    temp_path = os.path.join(data_dir,'class_'+str(class_idx)+'_seed_'+str(seed_idx)+'_num_'+str(i)+'.png')
                    save_image(samples[i], temp_path, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=2)
    parser.add_argument("--num-sampling-steps", type=int, default=100)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")

    parser.add_argument("--start_seed", type=int, default=0)  
    parser.add_argument("--end_seed", type=int, default=20) 
    parser.add_argument("--start_id", type=int, default=0)  
    parser.add_argument("--end_id", type=int, default=0)  
    parser.add_argument("--save_dir", type=str, default=None)

    args = parser.parse_args()
    main(args)

