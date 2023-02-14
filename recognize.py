import os
import glob
import argparse
import numpy as np
import PIL.Image as Image
from sklearn import preprocessing

import torch
import torchvision.transforms as transforms
from torchmetrics import CharErrorRate
from nn import CaptchaModel
from util import decode_predictions


def get_image(data_path, image_path, width=200, height=50):
    image = Image.open(os.path.join(data_path, image_path)).convert("RGB")
    image = image.resize(
        (width, height), resample=Image.BILINEAR
    )
    image = np.array(image)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std, inplace=True)
    ])

    image = transform(image)
    image = image.unsqueeze(0)
    return image


def calc_cer(images_path, weights_path):
    images_paths = glob.glob(os.path.join(images_path, "*.png"))
    targets = [x.split("/")[-1][:-4] for x in images_paths]
    targets_chars = [[c for c in x] for x in targets]
    targets_flat = [c for clist in targets_chars for c in clist]

    lbl_enc = preprocessing.LabelEncoder()
    lbl_enc.fit(targets_flat)

    model = CaptchaModel(num_chars=len(lbl_enc.classes_))
    model.load_state_dict(torch.load(weights_path))

    preds = []
    with torch.no_grad():
        model.eval()
        for image_path in images_paths:
            image = get_image(image_path.split("/")[-2], image_path.split("/")[-1])
            pred, _ = model(image)
            current_preds = decode_predictions(pred, lbl_enc)
            preds += current_preds
    cer = CharErrorRate()
    return cer(preds, targets).item()


def recognize(opt):
    image_files = glob.glob(os.path.join(opt.data_path, "*.png"))
    targets_orig = [x.split("/")[-1][:-4] for x in image_files]
    targets = [[c for c in x] for x in targets_orig]
    targets_flat = [c for clist in targets for c in clist]

    lbl_enc = preprocessing.LabelEncoder()
    lbl_enc.fit(targets_flat)

    image = get_image(opt.data_path, opt.image_path, opt.width, opt.height)
    model = CaptchaModel(num_chars=len(lbl_enc.classes_))
    model.load_state_dict(torch.load(opt.saved_model))

    with torch.no_grad():
        model.eval()
        pred, _ = model(image)
        current_preds = decode_predictions(pred, lbl_enc)
        print(f'Ground truth: {opt.image_path[:-4]}\n'
              f'Prediction: {current_preds[0]}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Captcha recognition using RNN')
    parser.add_argument('--data_path', type=str, default='samples', help='path to data folder')
    parser.add_argument('--saved_model', type=str, default='weights/weights.pth', help='path to saved model')
    parser.add_argument('--height', type=int, default=50, help='height of the input image')
    parser.add_argument('--width', type=int, default=200, help='width of the input image')
    parser.add_argument('--image_path', type=str, default='n8pfe.png', help='path to test image')
    opt = parser.parse_args()

    recognize(opt)
