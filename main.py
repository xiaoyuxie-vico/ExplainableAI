'''
python grad_cam.py <path_to_image>

Pipline:
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    4. Makes the visualization.
'''

import cv2
import numpy as np
from torchvision import models
from torchsummary import summary

from utils.guided_backprop_relu_model import GuidedBackpropReLUModel
from utils.grad_cam import GradCam
from utils.tools import deprocess_image
from utils.tools import get_args
from utils.tools import preprocess_image
from utils.tools import show_cam_on_image
from utils.tools import check_folder


def main():
    check_folder()
    args = get_args()

    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    model = models.resnet50(pretrained=True)
    # print(list(model._modules.keys()))
    # for name, module in model._modules.items():
    #     print('name', name)

    # model_summary = summary(model, (3, 224, 224))
    # print(model_summary)

    grad_cam = GradCam(
        model=model, 
        feature_module=model.layer4,
        target_layer_names=['2'], 
        use_cuda=args.use_cuda
    )

    img = cv2.imread(args.image_path, 1)
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    input = preprocess_image(img)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    target_index = None
    mask = grad_cam(input, target_index)

    show_cam_on_image(img, mask)

    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    # print(model._modules.items())
    gb = gb_model(input, index=target_index)
    gb = gb.transpose((1, 2, 0))
    cam_mask = cv2.merge([mask, mask, mask])
    cam_gb = deprocess_image(cam_mask*gb)
    gb = deprocess_image(gb)

    cv2.imwrite('./out/gb.jpg', gb)
    cv2.imwrite('./out/cam_gb.jpg', cam_gb)


if __name__ == '__main__':
    main()
