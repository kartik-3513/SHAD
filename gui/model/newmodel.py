import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample
)
from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip
)
import pytorchvideo
import pytorch_lightning
from pytorchvideo.data.encoded_video import EncodedVideo
from typing import Dict

class VideoClassificationLightningModule(pytorch_lightning.LightningModule):
    def __init__(self):
        super().__init__()
        device = "cuda"
        model_name = 'x3d_s'
        model = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=True)
        list(model.children())[0][-1].proj = torch.nn.Linear(in_features=2048, out_features=2, bias=True)
        for i in range(5):
            for param in list(model.children())[0][i].parameters():
                param.requires_grad = False
        self.model = model.to(device)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # The model expects a video tensor of shape (B, C, T, H, W), which is the
        # format provided by the dataset
        y_hat = self.model(batch["video"])

        # Compute cross entropy loss, loss.backwards will be called behind the scenes
        # by PyTorchLightning after being returned from this method.
        loss = F.cross_entropy(y_hat, batch["label"])

        # Log the train loss to Tensorboard
        self.log("train_loss", loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self.model(batch["video"])
        loss = F.cross_entropy(y_hat, batch["label"])
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

class NewModel:
    
    def __init__(self):
        self.model = VideoClassificationLightningModule.load_from_checkpoint("chk/epoch=0-step=6500.ckpt")
        self.model = model.eval()
        self.model_name = "x3d_s"
        self.device = "cuda"

        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        self.frames_per_second = 30
        model_transform_params  = {
            "x3d_xs": {
                "side_size": 182,
                "crop_size": 182,
                "num_frames": 4,
                "sampling_rate": 12,
            },
            "x3d_s": {
                "side_size": 182,
                "crop_size": 182,
                "num_frames": 13,
                "sampling_rate": 6,
            },
            "x3d_m": {
                "side_size": 256,
                "crop_size": 256,
                "num_frames": 16,
                "sampling_rate": 5,
            }
        }

        # Get transform parameters based on model
        transform_params = model_transform_params[model_name]

        self.transform =  ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(transform_params["num_frames"]),
                    Lambda(lambda x: x/255.0),
                    NormalizeVideo(mean, std),
                    ShortSideScale(size=transform_params["side_size"]),
                    CenterCropVideo(
                        crop_size=(transform_params["crop_size"], transform_params["crop_size"])
                    )
                ]
            ),
        )

        # The duration of the input clip is also specific to the model.
        self.clip_duration = (transform_params["num_frames"] * transform_params["sampling_rate"])/self.frames_per_second
    
    def process(self, video_path):
        start_sec = 0
        end_sec = start_sec + self.clip_duration
        video = EncodedVideo.from_path(video_path)
        video_duration = video.duration.numerator/video.duration.denominator
        
        out = []
        
        while end_sec < video_duration:
            video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
            video_data = self.transform(video_data)

            inputs = video_data["video"]
            inputs = inputs.to(device)

            preds = self.model(inputs[None, ...])
            post_act = torch.nn.Softmax(dim=1)
            preds = post_act(preds)
            pred_classes = preds.topk(k=1).indices[0]
            if pred_classes[0] == 0:
                pred_classname = "Normal"
            else:
                pred_classname = "Suspicious"

            out.append(pred_classname)

            start_sec = end_sec
            end_sec = start_sec + clip_duration
        
        newout = []
        num_frames = video_duration * self.frames_per_second
        for label in out:
            for i in range(int(num_frames / len(out))):
                newout.append(label)
        
        return newout