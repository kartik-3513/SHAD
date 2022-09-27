import json
import torch
from torchvision.transforms import Compose, Lambda
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
)

from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)

from .packPathway import PackPathway


class Model:
    side_size = 256
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    crop_size = 256
    num_frames = 32
    sampling_rate = 1
    frames_per_second = 30

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = "slowfast_r50"
        self.model = torch.hub.load(
            "facebookresearch/pytorchvideo", model=model_name, pretrained=True
        )
        self.model = self.model.to(self.device)
        self.model = self.model.eval()

        with open("model/kinetics_classnames.json", "r") as f:
            kinetics_classnames = json.load(f)
        self.kinetics_id_to_classname = {}
        for k, v in kinetics_classnames.items():
            self.kinetics_id_to_classname[v] = str(k).replace('"', "")

        self.transform = ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(self.num_frames),
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(self.mean, self.std),
                    ShortSideScale(size=self.side_size),
                    CenterCropVideo(self.crop_size),
                    PackPathway(),
                ]
            ),
        )
        self.clip_duration = (
            self.num_frames * self.sampling_rate
        ) / self.frames_per_second

    def process(self, video_path):
        start_sec = 0
        end_sec = start_sec + self.clip_duration
        video = EncodedVideo.from_path(video_path)
        video_duration = video.duration.numerator / video.duration.denominator

        out = []

        while end_sec < video_duration:
            video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
            video_data = self.transform(video_data)

            inputs = video_data["video"]
            inputs = [i.to(self.device)[None, ...] for i in inputs]

            preds = self.model(inputs)
            post_act = torch.nn.Softmax(dim=1)
            preds = post_act(preds)
            pred_classes = preds.topk(k=5).indices
            pred_class_names = [
                self.kinetics_id_to_classname[int(i)] for i in pred_classes[0]
            ]
            print("Predicted labels: %s" % ", ".join(pred_class_names))

            out.append((video_data["video"][0], pred_class_names[0]))

            start_sec = end_sec
            end_sec = start_sec + self.clip_duration

        newout = []
        num_frames = video_duration * self.frames_per_second
        for (bunch, label) in out:
            for i in range(int(num_frames / len(out))):
                newout.append(label)

        return newout
