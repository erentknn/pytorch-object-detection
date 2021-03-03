import time
import argparse
import logging
import warnings

import numpy as np
import cv2
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

torch.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s-%(name)s-%(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input", type=str, required=True,
                    help="path to input video file")
parser.add_argument("-o", "--output", type=str,
                    help="path to (optional) output video file")
parser.add_argument("-m", "--model", type=str, default="resnet_fpn",
                    help="which model to use, currently you can use only \
                          resnet_fpn and mobilenetv2, default=resnet_fpn")
parser.add_argument("-c", "--classes", type=int, default=2,
                    help="number of your classes including background \
                          e.g if you have 1 class you should write 2, \
                          1 class + background. default=2")
parser.add_argument("-w", "--weights", type=str, required=True,
                    help="path to your weights file")
parser.add_argument("-cnf", "--confidence", type=float, default=0.5,
                    help="level of confidence, default=0.5")
parser.add_argument("-ht", "--height", type=int, default=1200,
                    help="height of output, default=1200")
parser.add_argument("-wt", "--width", type=int, default=700,
                    help="width of output, default=700")


args = vars(parser.parse_args())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Device using: {}".format(device))


def get_model(model_arch, classes, weights):
    if model_arch == "resnet_fpn":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=False)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, classes)
    elif model_arch == "mobilenetv2":
        backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        backbone.out_channels = 1280
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                           aspect_ratios=((0.5, 1.0, 2.0),))
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=["0"],
            output_size=7,
            sampling_ratio=2)

        model = FasterRCNN(
            backbone,
            num_classes=classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler)
    else:
        raise NotImplementedError(
            "Use resnet_fpn or mobilenetv2")
    model.load_state_dict(torch.load(weights))
    model.to(device)
    model.eval()
    logger.info("Created model and loaded weights succesfully.")
    return model


def preprocess(frame):
    frame = frame.astype(np.float32)
    frame /= 255.0
    inputs = torch.as_tensor(frame).unsqueeze(0)
    inputs = inputs.permute(0, 3, 1, 2)
    inputs = inputs.to(device)
    return frame, inputs


@torch.no_grad()
def inference(model, inputs, confidence):
    start_time = time.time()
    outputs = model.forward(inputs)
    end_time = time.time()
    fps = (1 / (end_time - start_time))
    tholded_output = outputs[0]["boxes"][outputs[0]["scores"] >= confidence]
    return tholded_output, fps


def display_output(frame, outputs, color, fps):
    for box in outputs:
        tmp = box.type(torch.int16).tolist()
        h, w = tmp[2] - tmp[0], tmp[3] - tmp[1]
        cv2.rectangle(img=frame,
                      pt1=(tmp[0], tmp[1]),
                      pt2=(tmp[0] + h, tmp[1] + w), color=color, thickness=2)
    fps_label = "FPS: %.2f" % (fps)
    cv2.putText(frame, fps_label, (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow('result', frame)


def main():
    model = get_model(args["model"], args["classes"], args["weights"])
    video = args["input"]
    cap = cv2.VideoCapture(video)
    color = 255
    while cv2.waitKey(1) < 1:
        retaining, frame = cap.read()
        frame = cv2.resize(frame, (args["height"], args["width"]))
        if not retaining and frame is None:
            logger.info("End of stream")
        frame, inputs = preprocess(frame)
        outputs, fps = inference(model, inputs, args["confidence"])
        display_output(frame, outputs, color, fps)


if __name__ == "__main__":
    main()
