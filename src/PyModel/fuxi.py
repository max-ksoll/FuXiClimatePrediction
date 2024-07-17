import torch
from torchvision.models.swin_transformer import SwinTransformerBlockV2
from torch import nn
import logging
from torchvision.ops import Permute
from typing import Tuple, Union, Dict

logger = logging.getLogger(__name__)


class FuXi(torch.nn.Module):
    def __init__(
        self,
        input_var,
        channels,
        transformer_block_count,
        lat,
        long,
        heads=8,
        raw_fc_layer=False,
    ):
        super(FuXi, self).__init__()
        logger.info("Creating FuXi Model")
        self.space_time_cube_embedding = SpaceTimeCubeEmbedding(input_var, channels)
        self.u_transformer = UTransformer(transformer_block_count, channels, heads)
        # TODO durch einen Linear ersetzen anstatt Linear pro Channel
        if raw_fc_layer:
            self.fc = torch.nn.Sequential(
                torch.nn.Flatten(start_dim=1, end_dim=-1),
                torch.nn.Linear(
                    channels * (lat // 4) * (long // 4), input_var * lat * long
                ),
                torch.nn.Unflatten(1, (input_var, lat, long)),
            )
        else:
            self.fc = torch.nn.Sequential(
                # put channel dim front
                Permute([0, 2, 3, 1]),
                torch.nn.Linear(channels, input_var),
                # put lat dim front
                Permute([0, 3, 2, 1]),
                torch.nn.Linear(lat // 4, lat),
                # put long dim front
                Permute([0, 1, 3, 2]),
                torch.nn.Linear(long // 4, long),
            )
        self.modules = nn.ModuleList(
            [self.space_time_cube_embedding, self.u_transformer, self.fc]
        )

    def forward(self, x):
        x = self.space_time_cube_embedding(x)
        x = self.u_transformer(x)
        return self.fc(x)

    def step(
        self,
        timeseries,
        lat_weights,
        autoregression_steps=1,
        return_out=True,
        return_loss=True,
    ) -> Dict[str, torch.Tensor]:
        if autoregression_steps > timeseries.shape[1] - 2:
            raise ValueError(
                "autoregression_steps can't be greater than number of samples"
            )

        # NaN-Werte durch 0 ersetzen und eine Maske erstellen
        mask = ~torch.isnan(timeseries)
        timeseries = torch.nan_to_num(timeseries, nan=0.0)

        outputs = []
        loss = torch.zeros(1, device=timeseries.device)
        model_input = timeseries[:, 0:2, :, :, :]

        for step in range(autoregression_steps):
            if step > 0:
                model_input = torch.stack([model_input[:, 1, :, :, :], out], dim=1)
            out = self.forward(model_input)
            if return_out:
                outputs.append(out.detach())

            # Maske für die aktuelle Zeitschritt anwenden
            step_mask = mask[:, step + 2, :, :, :]
            if return_loss:
                loss += (
                    torch.sum(
                        torch.abs(timeseries[:, step + 2, :, :, :] - out)
                        * lat_weights
                        * step_mask
                    )
                    / step_mask.sum()
                )

        return_dict = {}
        if return_out:
            outputs = torch.stack(outputs, 1)
            return_dict["output"] = outputs

        if return_loss:
            return_dict["loss"] = loss / autoregression_steps
        return return_dict


class SpaceTimeCubeEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels):
        logger.info("Creating SpaceTimeCubeEmbedding layer")
        super(SpaceTimeCubeEmbedding, self).__init__()
        self.conv3d = nn.Conv3d(
            in_channels, out_channels, kernel_size=(2, 4, 4), stride=(2, 4, 4)
        )
        self.layer_norm = nn.LayerNorm(out_channels)
        self.layers = torch.nn.ModuleList([self.conv3d, self.layer_norm])

    def forward(self, x):
        x = x.permute(
            0, 2, 1, 3, 4
        )  # Move the channel dimension to the end for LayerNorm
        x = self.conv3d(x)
        x = x.permute(
            0, 2, 3, 4, 1
        )  # Move the channel dimension to the end for LayerNorm
        x = self.layer_norm(x)
        x = x.permute(0, 1, 4, 2, 3)
        x = torch.squeeze(x, dim=1)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.layers = torch.nn.ModuleList(
            [
                nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(2, channels),
                nn.SiLU(),
                nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(2, channels),
                nn.SiLU(),
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        logger.info("Creating DownBlock Layer")
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1
        )
        self.residual_block = ResidualBlock(out_channels)
        self.layers = torch.nn.ModuleList([self.conv1, self.residual_block])
        self.conv_adjust = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.conv1(x)
        residual = self.residual_block(out)
        out = torch.cat((out, residual), dim=1)
        out = self.conv_adjust(out)
        return out


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        logger.info("Creating UpBlock Layer")
        self.upsample = nn.ConvTranspose2d(
            in_channels * 2, in_channels, kernel_size=2, stride=2
        )
        self.residual_block = ResidualBlock(out_channels)
        self.layers = torch.nn.ModuleList([self.upsample, self.residual_block])
        self.conv_adjust = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)

    def forward(self, x, skip_connection):
        x = torch.cat([x, skip_connection], dim=1)
        x = self.upsample(x)
        x = torch.cat([x, self.residual_block(x)], dim=1)
        x = self.conv_adjust(x)
        return x


class UTransformer(torch.nn.Module):
    def __init__(self, layers, in_channels, heads):
        super().__init__()
        logger.info("Creating UTransformer Layer")
        self.downblock = DownBlock(in_channels, in_channels)
        window_size = [8, 8]
        self.attentionblock = []
        for i in range(layers):
            block = SwinTransformerBlockV2(
                dim=in_channels,
                num_heads=heads,
                window_size=window_size,
                shift_size=[0 if i % 2 == 0 else w // 2 for w in window_size],
                stochastic_depth_prob=0.2,
                mlp_ratio=4.0,
                dropout=0.05,
                attention_dropout=0.05,
            )
            self.attentionblock.append(block)

        self.upblock = UpBlock(in_channels, in_channels)
        self.layers = torch.nn.ModuleList(
            [self.downblock, self.upblock] + self.attentionblock
        )

    def forward(self, x):
        down = self.downblock(x)
        x = down
        x = torch.permute(x, (0, 2, 3, 1))
        for block in self.attentionblock:
            x = block(x)

        x = torch.permute(x, (0, 3, 1, 2))
        x = self.upblock(x, down)
        return x
