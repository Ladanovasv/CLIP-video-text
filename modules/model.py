import torchvision
from torch import nn
import torch
from transformers import BertTokenizer, BertModel


class CLIP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.text_encoder = torchvision.models.video.r2plus1d_18(
            pretrained=True)
        self.video_encoder = torchvision.models.video.r2plus1d_18(
            pretrained=True)

        self.logit_scale = nn.Parameter(torch.ones([]))

    # def get_sequence_visual_output(self, text):
    #     return()

    def forward(self, text_features, video_features):
        # text_features = self.text_encoder(
        #     token_input, token_type_ids=seg_input)[0]
        # print(text)
        # print(text_features)
        # video_features = self.self.video_encoder(video)

        # # normalized features
        image_features = video_features / \
            video_features.norm(dim=-1, keepdim=True)
        text_features = text_features / \
            text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_text, logits_per_image
