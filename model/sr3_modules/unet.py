import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction

from .downsampling import (
    PixelUnshuffleDown,
    ConvStride2Down,
    FrequencyPreservedPooling,
    FrequencyPreservedPooling_DropHigh,
)

from .upsampling import (
    PixelShuffleUp,
    LCTCUp,
    FreqAvgUp,
)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# PositionalEncoding Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

# Modifying the downample and upsample process

def make_downsample(
    downsample_type,
    chan,
    fpdh_drop_prob=0.3,
):
    downsample_type = downsample_type.lower()

    if downsample_type == "original":
        module = Downsample(chan)
        out_chan = chan

    elif downsample_type == "convstride2":
        module = ConvStride2Down(
            in_channels=chan,
            out_channels=chan * 2,
        )
        out_chan = chan * 2

    elif downsample_type == "pixelunshuffle":
        module = PixelUnshuffleDown(
            in_channels=chan,
        )
        out_chan = chan * 4

    elif downsample_type == "fp":
        module = FrequencyPreservedPooling(
            channels=chan,
        )
        out_chan = chan * 4

    elif downsample_type == "fpdh":
        module = FrequencyPreservedPooling_DropHigh(
            channels=chan,
            drop_prob=fpdh_drop_prob,
        )
        out_chan = chan * 4

    else:
        raise ValueError(
            f"Unknown downsample_type: {downsample_type}"
        )

    return module, out_chan


def make_upsample(
    upsample_type,
    chan,
    out_chan,
):
    upsample_type = upsample_type.lower()

    if upsample_type == "original":
        return nn.Sequential(
            nn.Upsample(
                scale_factor=2,
                mode="nearest",
            ),
            nn.Conv2d(
                chan,
                out_chan,
                kernel_size=3,
                padding=1,
            ),
        )

    elif upsample_type == "pixelshuffle":
        return PixelShuffleUp(
            in_channels=chan,
            out_channels=out_chan,
        )

    elif upsample_type == "lctc_7":
        return LCTCUp(
            in_channels=chan,
            out_channels=out_chan,
            large_kernel=7,
            small_kernel=None,
        )

    elif upsample_type == "lctc_11_3":
        return LCTCUp(
            in_channels=chan,
            out_channels=out_chan,
            large_kernel=11,
            small_kernel=3,
        )

    elif upsample_type == "freqavgup":
        return FreqAvgUp(
            in_channels=chan,
            out_channels=out_chan,
            padding="constant",
        )

    else:
        raise ValueError(
            f"Unknown upsample_type: {upsample_type}"
        )


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=32):
        super().__init__()
        if noise_level_emb_dim is not None:
            self.noise_func = FeatureWiseAffine(
                noise_level_emb_dim, dim_out, use_affine_level)

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        b, c, h, w = x.shape
        h = self.block1(x)
        if time_emb is not None:
            h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()

        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if(self.with_attn):
            x = self.attn(x)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channel=6,
        out_channel=3,
        inner_channel=32,
        norm_groups=32,
        channel_mults=(1, 2, 4, 8, 8),
        attn_res=(8),
        res_blocks=3,
        dropout=0,
        with_noise_level_emb=True,
        image_size=128,
        downsample_type="original",
        upsample_type="original",
        fpdh_drop_prob=0.3,
    ):
        super().__init__()
        self.downsample_type = downsample_type
        self.upsample_type = upsample_type
        self.fpdh_drop_prob = fpdh_drop_prob

        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None

        num_mults = len(channel_mults)

        pre_channel = inner_channel
        now_res = image_size

        downs = [
            nn.Conv2d(
                in_channel,
                inner_channel,
                kernel_size=3,
                padding=1,
            )
        ]

        # Une entrée par feature sauvegardée dans le forward.
        feat_channels = [pre_channel]

        # Canaux utilisés à chaque résolution de l'encodeur.
        # Exemple avec inner_channel=32 et fp :
        # [32, 128, 512, 2048]
        encoder_channels = [pre_channel]

        for ind in range(num_mults):
            is_last = ind == num_mults - 1
            use_attn = now_res in attn_res

            # Comme dans NAFNet, les blocs du niveau courant
            # conservent le nombre de canaux courant.
            for _ in range(res_blocks):
                downs.append(
                    ResnetBlocWithAttn(
                        pre_channel,
                        pre_channel,
                        noise_level_emb_dim=noise_level_channel,
                        norm_groups=norm_groups,
                        dropout=dropout,
                        with_attn=use_attn,
                    )
                )

                feat_channels.append(pre_channel)

            if not is_last:
                down_module, next_channel = make_downsample(
                    self.downsample_type,
                    pre_channel,
                    self.fpdh_drop_prob,
                )

                downs.append(down_module)

                # Le résultat du downsampling est également sauvegardé
                # par le forward dans la pile des skips.
                feat_channels.append(next_channel)

                # Évolution dynamique des canaux.
                pre_channel = next_channel
                encoder_channels.append(pre_channel)

                now_res = now_res // 2

        self.downs = nn.ModuleList(downs)

        # On conserve cette liste pour construire ensuite
        # le décodeur dynamiquement.
        self.encoder_channels = encoder_channels

        print(
        f"[UNet] downsample_type={self.downsample_type}, "
        f"encoder_channels={self.encoder_channels}"
        )

        self.encoder_feature_channels = feat_channels.copy()

        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=True),
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=False)
        ])

        ups = []

        for ind in reversed(range(num_mults)):
            is_last = ind == 0
            use_attn = now_res in attn_res

            # Nombre de canaux du niveau encodeur correspondant.
            #
            # Exemple convstride2 :
            # encoder_channels = [32, 64, 128, 256]
            #
            # Pour ind=3 :
            # level_channel = 256
            level_channel = encoder_channels[ind]

            for block_ind in range(res_blocks + 1):
                if not feat_channels:
                    raise RuntimeError(
                        "La pile feat_channels est vide pendant "
                        f"la construction du niveau {ind}, bloc {block_ind}."
                    )

                skip_channel = feat_channels.pop()

                # Avec notre construction dynamique, tous les skips
                # associés à ce niveau doivent avoir level_channel canaux.
                if skip_channel != level_channel:
                    raise RuntimeError(
                        "Incohérence dans les canaux des skips : "
                        f"niveau={ind}, "
                        f"bloc={block_ind}, "
                        f"skip_channel={skip_channel}, "
                        f"level_channel={level_channel}"
                    )

                # Dans le forward, l'entrée réelle sera :
                #
                # torch.cat((x, skip), dim=1)
                #
                # Elle contient donc :
                #
                # pre_channel + skip_channel
                concatenated_channel = pre_channel + skip_channel

                ups.append(
                    ResnetBlocWithAttn(
                        concatenated_channel,
                        level_channel,
                        noise_level_emb_dim=noise_level_channel,
                        norm_groups=norm_groups,
                        dropout=dropout,
                        with_attn=use_attn,
                    )
                )

                # La sortie du ResNet block revient aux canaux
                # du niveau courant.
                pre_channel = level_channel

            if not is_last:
                # Le niveau moins profond est celui d'indice ind - 1.
                target_channel = encoder_channels[ind - 1]

                ups.append(
                    make_upsample(
                        self.upsample_type,
                        pre_channel,
                        target_channel,
                    )
                )

                # L'upsampler fait réellement évoluer les canaux.
                pre_channel = target_channel
                now_res = now_res * 2

        if feat_channels:
            raise RuntimeError(
                "Certains skips n'ont pas été utilisés : "
                f"{feat_channels}"
            )

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)

    def forward(self, x, time):
        t = self.noise_level_mlp(time) if exists(
            self.noise_level_mlp) else None

        feats = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)

        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(torch.cat((x, feats.pop()), dim=1), t)
            else:
                x = layer(x)

        return self.final_conv(x)
    
    def get_sampling_scalars(self):
        scalars = {}

        for name, parameter in self.named_parameters():
            if name.endswith(".alpha") or name.endswith(".beta"):
                scalars[name] = float(
                    parameter.detach()
                    .float()
                    .mean()
                    .cpu()
                )

        return scalars
