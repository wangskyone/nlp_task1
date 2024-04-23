import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from torchvision import transforms, models


class model_common(nn.Module):
    def __init__(self, text_encoder_path: str):
        super(model_common, self).__init__()
        config = AutoConfig.from_pretrained(text_encoder_path)
        self.bert = AutoModel.from_pretrained(text_encoder_path, config=config)
        self.img_encoder = models.resnet50(pretrained=True)
        if 'roberta' in text_encoder_path.lower() or 'bert' in text_encoder_path.lower():
            fc = nn.Linear(self.img_encoder.fc.in_features, 768)
            self.img_encoder.fc = fc

    def forward(self, image_input, input_idx, attention_mask, token_type_ids, mode='img_word'):
        if mode == 'img_word':
            img_out = self.img_encoder(image_input)
            text_out = self.bert(input_idx, attention_mask, token_type_ids).last_hidden_state[:, 0]
            return img_out, text_out
        elif mode == 'word':
            text_out = self.bert(input_idx, attention_mask, token_type_ids).last_hidden_state[:, 0]
            return text_out
        else:
            img_out = self.img_encoder(image_input)
            return img_out

