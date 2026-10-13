import torch
import torch.nn as nn
import torch.nn.functional as F
from openood.networks.clip import clip


def load_clip_to_cpu(
    backbone_name,
):

    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class VanillaCLIP(nn.Module):
    def __init__(
        self,
        cfg,
    ):

        super().__init__()
        backbone = cfg.backbone.name
        assert backbone in clip.available_models()

        print(f"Loading CLIP (backbone: {backbone})")
        clip_model = load_clip_to_cpu(backbone)
        clip_model = clip_model.cuda()

        self.model = clip_model
        self.logit_scale = clip_model.logit_scale.data

        print("Turning off gradients in both the image and the text encoder")

        for name, param in clip_model.named_parameters():
            param.requires_grad_(False)


    @torch.no_grad()
    def encode_image(
        self,
        x,
        normalize=True,
    ):

        image_features = self.model.encode_image(x)

        if normalize:
            image_features = F.normalize(image_features, dim=-1)

        return image_features


    def get_logit_scale(self):
        return self.logit_scale.exp()
