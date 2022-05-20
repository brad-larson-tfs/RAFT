import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder



DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img1, img2, flo, filename):
    img1 = img1[0].permute(1,2,0).cpu().numpy()
    img2 = img2[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img1, img2, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    cv2.imwrite(filename, img_flo[:, :, [2,1,0]].astype(int))


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        i = 0
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            viz(image1, image2, flow_up, 'save{:02d}.png'.format(i))
            i += 1

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser(description='Process arguments')

    parser.add_argument('-d', '--debug', action='store_true',help='Wait for debuggee attach')   
    parser.add_argument('-debug_port', type=int, default=3000, help='Debug port')
    parser.add_argument('-debug_address', type=str, default='0.0.0.0', help='Debug port')

    parser.add_argument('--model', help="restore checkpoint", default='models/raft-things.pth')
    parser.add_argument('--path', help="dataset for evaluation", default='demo-frames')
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

    parser.add_argument('-dataset_path', type=str, default='./dataset', help='Local dataset path')
    parser.add_argument('-credentails', type=str, default='creds.json', help='Credentials file.')
    parser.add_argument('-dataset', type=str, default='annotations/lit/dataset.yaml', help='Image dataset file')
    parser.add_argument('-class_dict', type=str, default='model/crisplit/lit.json', help='Model class definition file.')
    parser.add_argument('-training', type=str, default='crisplit', help='Credentials file.')
    parser.add_argument('-num_images', type=int, default=10, help='Maximum number of images to display')
    parser.add_argument('-num_workers', type=int, default=1, help='Data loader workers')
    parser.add_argument('-batch_size', type=int, default=4, help='Dataset batch size')
    parser.add_argument('-test_iterator', type=bool, default=True, help='True to test iterator')
    parser.add_argument('-test_path', type=str, default='./datasets_test/', help='Test path ending in a forward slash')
    parser.add_argument('-test_dataset', type=bool, default=True, help='True to test dataset')
    parser.add_argument('-cuda', type=bool, default=True)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arguments()
    if args.debug:
        print("Wait for debugger attach on {}:{}".format(args.debug_address, args.debug_port))
        import debugpy

        debugpy.listen(address=(args.debug_address, args.debug_port)) # Pause the program until a remote debugger is attached
        debugpy.wait_for_client()  # Pause the program until a remote debugger is attached
        print("Debugger attached")

    demo(args)
