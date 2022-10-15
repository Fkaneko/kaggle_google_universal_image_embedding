from typing import Optional, Tuple

import torch
from torch import functional as F
from torch import nn


class SimpleHead(torch.nn.Module):
    def __init__(
        self,
        in_features: int = 512,
        out_features: int = 64,
        compress_module: Optional[nn.Module] = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        if compress_module is not None:
            self.compress_module = compress_module
        else:
            self.compress_module = torch.nn.Linear(
                in_features=in_features, out_features=out_features, device=device
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.compress_module(x)
        return x


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int = 128,
        intermediate_dim: int = 96,
        out_features: int = 64,
        drop_out_rate: float = 0.0,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.drop_out_rate = drop_out_rate
        self.activation_fn = nn.ReLU()
        self.fc1 = nn.Linear(in_features, intermediate_dim, device=device)
        self.fc2 = nn.Linear(intermediate_dim, out_features, device=device)
        self.drop_out = nn.Dropout(p=drop_out_rate)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.drop_out(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.drop_out(hidden_states)
        return hidden_states


class MLPCompressor(nn.Module):
    def __init__(
        self,
        in_features: int = 512,
        out_features: int = 64,
        use_baseline: bool = False,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()

        self.use_baseline = use_baseline
        mlp_1_out_dim = in_features // 2
        self.mlp_1 = MLP(
            in_features=in_features,
            intermediate_dim=int((in_features - mlp_1_out_dim) * 0.5) + mlp_1_out_dim,
            out_features=mlp_1_out_dim,
            drop_out_rate=0.2,
            device=device,
        )
        self.bn_1 = nn.BatchNorm1d(mlp_1_out_dim, device=device)
        self.act = nn.ReLU()
        self.mlp_2 = MLP(
            in_features=mlp_1_out_dim,
            intermediate_dim=mlp_1_out_dim // 2,
            out_features=out_features,
            drop_out_rate=0.1,
            device=device,
        )
        self.baseline_pooling = nn.AdaptiveAvgPool1d(output_size=out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        baseline = self.baseline_pooling(x)
        x = self.mlp_1(x)
        x = self.bn_1(x)
        x = self.act(x)
        x = self.mlp_2(x)
        if self.use_baseline:
            x += baseline
        return x


class TransformerCompress(nn.Module):
    def __init__(
        self,
        in_features: int = 512,
        d_model: int = 64,
        out_features: int = 64,
        use_baseline: bool = False,
        num_transformer_layers: int = 3,
        nhead: int = 4,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()

        self.unit_ch = 32
        self.use_baseline = use_baseline
        self.length = in_features // self.unit_ch
        self.conv1d = nn.Conv1d(
            in_channels=self.unit_ch,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            device=device,
        )
        dim_feedforward = int(d_model * 4)
        # from pytorch MultiHeadAttention args description
        # :math:`(L, N, E)` when ``batch_first=False`` or :math:`(N, L, E)` when ``batch_first=True``,
        #  where :math:`L` is the target sequence length, :math:`N` is the batch size, and :math:`E` is the
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            device=device,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=transformer_layer, num_layers=num_transformer_layers
        )
        self.pos_embedding = nn.Embedding(self.length, embedding_dim=d_model, device=device)

        intermediate_dim = max(out_features, int((d_model - out_features) * 0.5) + out_features)

        self.layer_norm = nn.LayerNorm(d_model, device=device)
        self.mlp = MLP(
            in_features=d_model,
            intermediate_dim=intermediate_dim,
            out_features=out_features,
            device=device,
        )

        self.baseline_pooling = nn.AdaptiveAvgPool1d(output_size=out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        baseline = self.baseline_pooling(x)

        bs, in_channels = x.shape
        # (bs, ch, length)
        x = self.conv1d(x.view(bs, self.unit_ch, self.length))
        # -> (bs, length, ch) -> (length, bs, ch)
        x = x.transpose(2, 1).transpose(1, 0).contiguous()

        # (length, ch)
        pos_embedding = self.pos_embedding(torch.arange(0, self.length, device=x.device))
        # (length, bs, ch), at batch_first = False, pytorch default
        x = x + pos_embedding.unsqueeze(1)
        x = self.transformer_encoder(x)

        # (length, bs, ch)
        x = self.layer_norm(x)
        x = self.mlp(x)

        # (bs, ch)
        x = x.sum(dim=0) / self.length

        if self.use_baseline:
            x += baseline
        return x


class AutoEncoder(torch.nn.Module):
    def __init__(
        self,
        in_features: int = 512,
        out_features: int = 64,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.compress_module = SimpleHead(
            in_features=in_features, out_features=out_features, device=device
        )
        self.decoder = torch.nn.Linear(
            in_features=out_features, out_features=in_features, device=device
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.compress_module(x)
        reconstructed = self.decoder(encoded)
        return encoded, reconstructed


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_features = 512
    out_features = 64
    batch_size = 32
    # prepare input
    input = torch.randn((batch_size, in_features), dtype=torch.float32, device=device)
    model = TransformerCompress(d_model=128, device=device)
    x = model(input)
    print(x.shape)

    model = MLPCompressor(device=device)
    x = model(input)
    print(x.shape)

    # model = AutoEncoder(in_features=in_features, out_features=out_features, device=device)

    # # prepare input
    # input = torch.randn((in_features), dtype=torch.float32, device=device)

    # encoded, out = model(x=input.unsqueeze(0))
    # assert encoded.shape == (1, out_features)
    # assert out.shape == (1, in_features)
