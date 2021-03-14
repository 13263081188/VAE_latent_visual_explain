#!usr/bin/python

"""
CAM visualization
"""

import math
import argparse
from io import BytesIO

import matplotlib.pyplot as plt
import requests
from PIL import Image
import torch
from torchvision import models
from torchvision.transforms.functional import normalize, resize, to_tensor, to_pil_image

from torchcam.cams import CAM, GradCAM, GradCAMpp, SmoothGradCAMpp, ScoreCAM, SSCAM, ISCAM
from torchcam.utils import overlay_mask


def main(args):

    if args.device is None:
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    device = torch.device(args.device)

    # Pretrained imagenet model
    model = models.__dict__[args.model](pretrained=True).to(device=device)

    # Image
    if args.img.startswith('http'):
        img_path = BytesIO(requests.get(args.img).content)
    else:
        img_path = args.img
    pil_img = Image.open(img_path, mode='r').convert('RGB')

    # Preprocess image
    img_tensor = normalize(to_tensor(resize(pil_img, (224, 224))),
                           [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).to(device=device)

    # Hook the corresponding layer in the model
    cam_extractors = [
        CAM(model),
        GradCAM(model), GradCAMpp(model), SmoothGradCAMpp(model),
        ScoreCAM(model), SSCAM(model), ISCAM(model),
    ]

    # Don't trigger all hooks
    for extractor in cam_extractors:
        extractor._hooks_enabled = False

    num_rows = 2
    num_cols = math.ceil(len(cam_extractors) / num_rows)
    _, axes = plt.subplots(num_rows, num_cols, figsize=(6, 4))
    for idx, extractor in enumerate(cam_extractors):
        extractor._hooks_enabled = True
        model.zero_grad()
        scores = model(img_tensor.unsqueeze(0))

        # Select the class index
        class_idx = scores.squeeze(0).argmax().item() if args.class_idx is None else args.class_idx

        # Use the hooked data to compute activation map
        activation_map = extractor(class_idx, scores).cpu()

        # Clean data
        extractor.clear_hooks()
        extractor._hooks_enabled = False
        # Convert it to PIL image
        # The indexing below means first image in batch
        heatmap = to_pil_image(activation_map, mode='F')
        # Plot the result
        result = overlay_mask(pil_img, heatmap)

        axes[idx // num_cols][idx % num_cols].imshow(result)
        axes[idx // num_cols][idx % num_cols].set_title(extractor.__class__.__name__, size=8)

    # Clear axes
    for row in axes:
        for ax in row:
            ax.axis('off')

    plt.tight_layout()
    if args.savefig:
        plt.savefig(args.savefig, dpi=200, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Saliency Map comparison',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, default='resnet18', help="The name of your training")
    parser.add_argument("--img", type=str,
                        default='https://www.woopets.fr/assets/races/000/066/big-portrait/border-collie.jpg',
                        help="The image to extract CAM from")
    parser.add_argument("--class-idx", type=int, default=232, help='Index of the class to inspect')
    parser.add_argument("--device", type=str, default=None, help='Default device to perform computation on')
    parser.add_argument("--savefig", type=str, default=None, help="Path to save figure")
    args = parser.parse_args()

    main(args)
