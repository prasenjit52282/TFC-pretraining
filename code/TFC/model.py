from torch import nn
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer

"""Normal Attention to fuse feature dim to 1"""
class BAttention(nn.Module):
    def __init__(self, input_shape):
        super(BAttention, self).__init__()
        self.W = nn.Parameter(torch.randn(input_shape[1], 1))
        self.b = nn.Parameter(torch.zeros(input_shape[0],1))

    def forward(self, x):
        e = torch.tanh(torch.matmul(x, self.W) + self.b)
        e = e.squeeze(dim=-1)
        alpha = nn.functional.softmax(e, dim=-1)
        alpha = alpha.unsqueeze(dim=-1)
        context = x * alpha
        context = torch.sum(context, dim=1)
        return context

"""Two contrastive encoders"""
class TFC(nn.Module):
    def __init__(self, configs):
        super(TFC, self).__init__()

        encoder_layers_t = TransformerEncoderLayer(configs.TSlength_aligned, dim_feedforward=2*configs.TSlength_aligned, nhead=2, batch_first=True)
        self.transformer_encoder_t = TransformerEncoder(encoder_layers_t, 2)

        self.batten_t=BAttention((configs.input_channels,configs.TSlength_aligned))

        self.projector_t = nn.Sequential(
            nn.Linear(configs.TSlength_aligned, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        encoder_layers_f = TransformerEncoderLayer(configs.TSlength_aligned, dim_feedforward=2*configs.TSlength_aligned,nhead=2, batch_first=True)
        self.transformer_encoder_f = TransformerEncoder(encoder_layers_f, 2)

        self.batten_f=BAttention((configs.input_channels,configs.TSlength_aligned))

        self.projector_f = nn.Sequential(
            nn.Linear(configs.TSlength_aligned, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )


    def forward(self, x_in_t, x_in_f):
        """Use Transformer"""
        x = self.transformer_encoder_t(x_in_t)
        # h_time = x.reshape(x.shape[0], -1)
        h_time = self.batten_t(x)

        """Cross-space projector"""
        z_time = self.projector_t(h_time)

        """Frequency-based contrastive encoder"""
        f = self.transformer_encoder_f(x_in_f)
        # h_freq = f.reshape(f.shape[0], -1)
        h_freq = self.batten_f(f)

        """Cross-space projector"""
        z_freq = self.projector_f(h_freq)

        return h_time, z_time, h_freq, z_freq


"""Downstream classifier only used in finetuning"""
class target_classifier(nn.Module):
    def __init__(self, configs):
        super(target_classifier, self).__init__()
        self.logits = nn.Linear(2*128, 64)
        self.logits_simple = nn.Linear(64, configs.num_classes_target)

    def forward(self, emb):
        emb_flat = emb.reshape(emb.shape[0], -1)
        emb = torch.sigmoid(self.logits(emb_flat))
        pred = self.logits_simple(emb)
        return pred
