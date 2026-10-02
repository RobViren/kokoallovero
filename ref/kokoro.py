"""Kokoro-82M inference, reimplemented from hexgrad's kokoro package (https://huggingface.co/hexgrad/Kokoro-82M,
Apache-2.0), which builds on StyleTTS2 and iSTFTNet (MIT). Loads the published kokoro-v1_0.pth and voice packs
unchanged. Takes Kokoro phoneme strings (ag2p makes them); batch of one, so no padding masks anywhere."""
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

REPO = "hexgrad/Kokoro-82M"
WEIGHTS = "kokoro-v1_0.pth"
SAMPLE_RATE = 24000


class AlbertLayer(nn.Module):
    def __init__(self, d, heads, ff):
        super().__init__()
        self.heads = heads
        self.attention = nn.Module()
        for name in ("query", "key", "value", "dense"):
            setattr(self.attention, name, nn.Linear(d, d))
        self.attention.LayerNorm = nn.LayerNorm(d, eps=1e-12)
        self.ffn = nn.Linear(d, ff)
        self.ffn_output = nn.Linear(ff, d)
        self.full_layer_layer_norm = nn.LayerNorm(d, eps=1e-12)

    def forward(self, x):
        b, t, d = x.shape
        att = self.attention
        q, k, v = (p(x).view(b, t, self.heads, -1).transpose(1, 2) for p in (att.query, att.key, att.value))
        x = att.LayerNorm(x + att.dense(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, t, d)))
        return self.full_layer_layer_norm(x + self.ffn_output(F.gelu(self.ffn(x), approximate="tanh")))


class Albert(nn.Module):
    """PL-BERT: ALBERT with one shared layer applied num_hidden_layers times."""

    def __init__(self, n_token, hidden_size, num_attention_heads, intermediate_size, max_position_embeddings,
                 num_hidden_layers, embedding_size=128, **_):
        super().__init__()
        self.embeddings = nn.Module()
        self.embeddings.word_embeddings = nn.Embedding(n_token, embedding_size)
        self.embeddings.position_embeddings = nn.Embedding(max_position_embeddings, embedding_size)
        self.embeddings.token_type_embeddings = nn.Embedding(2, embedding_size)
        self.embeddings.LayerNorm = nn.LayerNorm(embedding_size, eps=1e-12)
        self.embedding_hidden_mapping_in = nn.Linear(embedding_size, hidden_size)
        self.layer = AlbertLayer(hidden_size, num_attention_heads, intermediate_size)
        self.depth = num_hidden_layers

    def forward(self, ids):
        e = self.embeddings
        pos = torch.arange(ids.shape[1], device=ids.device)[None]
        x = e.word_embeddings(ids) + e.token_type_embeddings(torch.zeros_like(ids)) + e.position_embeddings(pos)
        x = self.embedding_hidden_mapping_in(e.LayerNorm(x))
        for _ in range(self.depth):
            x = self.layer(x)
        return x


def bilstm(d_in, d_out):
    return nn.LSTM(d_in, d_out // 2, batch_first=True, bidirectional=True)


class ChannelNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        return F.layer_norm(x.transpose(1, -1), self.gamma.shape, self.gamma, self.beta).transpose(1, -1)


class TextEncoder(nn.Module):
    def __init__(self, channels, kernel_size, depth, n_symbols):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, channels)
        self.cnn = nn.ModuleList(nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2), ChannelNorm(channels), nn.LeakyReLU(0.2),
        ) for _ in range(depth))
        self.lstm = bilstm(channels, channels)

    def forward(self, ids):
        x = self.embedding(ids).transpose(1, 2)
        for block in self.cnn:
            x = block(x)
        return self.lstm(x.transpose(1, 2))[0].transpose(1, 2)


class AdaLayerNorm(nn.Module):
    def __init__(self, style_dim, channels):
        super().__init__()
        self.fc = nn.Linear(style_dim, channels * 2)

    def forward(self, x, s):
        gamma, beta = self.fc(s)[:, None].chunk(2, dim=-1)
        return (1 + gamma) * F.layer_norm(x, x.shape[-1:]) + beta


class DurationEncoder(nn.Module):
    def __init__(self, style_dim, d_model, depth):
        super().__init__()
        self.lstms = nn.ModuleList()
        for _ in range(depth):
            self.lstms += [bilstm(d_model + style_dim, d_model), AdaLayerNorm(style_dim, d_model)]

    def forward(self, x, s):
        style = s[:, None].expand(-1, x.shape[1], -1)
        x = torch.cat([x, style], -1)
        for lstm, norm in zip(self.lstms[::2], self.lstms[1::2]):
            x = torch.cat([norm(lstm(x)[0], s), style], -1)
        return x


class AdaIN1d(nn.Module):
    def __init__(self, style_dim, channels):
        super().__init__()
        self.fc = nn.Linear(style_dim, channels * 2)

    def forward(self, x, s):
        gamma, beta = self.fc(s)[..., None].chunk(2, dim=1)
        return (1 + gamma) * F.instance_norm(x, eps=1e-5) + beta


class AdainResBlk1d(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.conv1 = nn.Conv1d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv1d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN1d(style_dim, dim_in)
        self.norm2 = AdaIN1d(style_dim, dim_out)
        self.conv1x1 = nn.Conv1d(dim_in, dim_out, 1, bias=False) if dim_in != dim_out else nn.Identity()
        self.pool = (nn.ConvTranspose1d(dim_in, dim_in, 3, 2, 1, output_padding=1, groups=dim_in)
                     if upsample else nn.Identity())

    def forward(self, x, s):
        r = self.pool(F.leaky_relu(self.norm1(x, s), 0.2))
        r = self.conv2(F.leaky_relu(self.norm2(self.conv1(r), s), 0.2))
        shortcut = self.conv1x1(F.interpolate(x, scale_factor=2) if self.upsample else x)
        return (r + shortcut) * 2 ** -0.5


class ProsodyPredictor(nn.Module):
    def __init__(self, style_dim, d_hid, depth, max_dur):
        super().__init__()
        self.text_encoder = DurationEncoder(style_dim, d_hid, depth)
        self.lstm = bilstm(d_hid + style_dim, d_hid)
        self.duration_proj = nn.Linear(d_hid, max_dur)
        self.shared = bilstm(d_hid + style_dim, d_hid)
        for name in ("F0", "N"):
            setattr(self, name, nn.ModuleList([
                AdainResBlk1d(d_hid, d_hid, style_dim),
                AdainResBlk1d(d_hid, d_hid // 2, style_dim, upsample=True),
                AdainResBlk1d(d_hid // 2, d_hid // 2, style_dim),
            ]))
        self.F0_proj = nn.Conv1d(d_hid // 2, 1, 1)
        self.N_proj = nn.Conv1d(d_hid // 2, 1, 1)

    def f0_and_noise(self, en, s):
        x = self.shared(en.transpose(1, 2))[0].transpose(1, 2)
        curves = []
        for blocks, proj in ((self.F0, self.F0_proj), (self.N, self.N_proj)):
            y = x
            for block in blocks:
                y = block(y, s)
            curves.append(proj(y).squeeze(1))
        return curves


def snake(x, alpha):
    return x + (1 / alpha) * torch.sin(alpha * x) ** 2


class AdaINResBlock1(nn.Module):
    def __init__(self, channels, kernel_size, dilations, style_dim):
        super().__init__()
        self.convs1 = nn.ModuleList(nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=d * (kernel_size - 1) // 2)
                                    for d in dilations)
        self.convs2 = nn.ModuleList(nn.Conv1d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2)
                                    for _ in dilations)
        self.adain1 = nn.ModuleList(AdaIN1d(style_dim, channels) for _ in dilations)
        self.adain2 = nn.ModuleList(AdaIN1d(style_dim, channels) for _ in dilations)
        self.alpha1 = nn.ParameterList(nn.Parameter(torch.ones(1, channels, 1)) for _ in dilations)
        self.alpha2 = nn.ParameterList(nn.Parameter(torch.ones(1, channels, 1)) for _ in dilations)

    def forward(self, x, s):
        for c1, c2, n1, n2, a1, a2 in zip(self.convs1, self.convs2, self.adain1, self.adain2, self.alpha1, self.alpha2):
            x = x + c2(snake(n2(c1(snake(n1(x, s), a1)), s), a2))
        return x


class HarmonicSource(nn.Module):
    """Neural source filter excitation: F0 and its 8 overtones as sines, noise where unvoiced, merged to one
    channel. The random initial phases and the noise are the only stochastic part of the model."""

    def __init__(self, upsample_scale, harmonics=9, sine_amp=0.1, noise_std=0.003, voiced_threshold=10):
        super().__init__()
        self.upsample_scale, self.harmonics = upsample_scale, harmonics
        self.sine_amp, self.noise_std, self.voiced_threshold = sine_amp, noise_std, voiced_threshold
        self.l_linear = nn.Linear(harmonics, 1)

    def forward(self, f0):
        fn = f0 * torch.arange(1, self.harmonics + 1, device=f0.device, dtype=f0.dtype)
        rad = (fn / SAMPLE_RATE) % 1
        initial_phase = torch.rand(fn.shape[0], self.harmonics, device=f0.device)
        initial_phase[:, 0] = 0
        rad[:, 0] += initial_phase
        # Integrate phase at frame rate and interpolate back up, as trained; a sample-rate cumsum sounds different.
        rad = F.interpolate(rad.transpose(1, 2), scale_factor=1 / self.upsample_scale, mode="linear")
        phase = torch.cumsum(rad, dim=2) * 2 * math.pi
        phase = F.interpolate(phase * self.upsample_scale, scale_factor=self.upsample_scale, mode="linear")
        sines = torch.sin(phase).transpose(1, 2) * self.sine_amp
        uv = (f0 > self.voiced_threshold).float()
        noise = (uv * self.noise_std + (1 - uv) * self.sine_amp / 3) * torch.randn_like(sines)
        return torch.tanh(self.l_linear(sines * uv + noise))


class Stft(nn.Module):
    def __init__(self, n_fft, hop):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

    def transform(self, x):
        spec = torch.stft(x, self.n_fft, self.hop, window=self.window, return_complex=True)
        return spec.abs(), spec.angle()

    def inverse(self, magnitude, phase):
        return torch.istft(magnitude * torch.exp(phase * 1j), self.n_fft, self.hop, window=self.window)


class Generator(nn.Module):
    def __init__(self, style_dim, resblock_kernel_sizes, upsample_rates, upsample_initial_channel,
                 resblock_dilation_sizes, upsample_kernel_sizes, gen_istft_n_fft, gen_istft_hop_size):
        super().__init__()
        self.n_kernels = len(resblock_kernel_sizes)
        self.post_n_fft = gen_istft_n_fft
        self.f0_scale = math.prod(upsample_rates) * gen_istft_hop_size
        self.m_source = HarmonicSource(self.f0_scale)
        self.stft = Stft(gen_istft_n_fft, gen_istft_hop_size)
        self.ups, self.resblocks, self.noise_convs, self.noise_res = (nn.ModuleList() for _ in range(4))
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            ch = upsample_initial_channel // 2 ** (i + 1)
            self.ups.append(nn.ConvTranspose1d(2 * ch, ch, k, u, padding=(k - u) // 2))
            self.resblocks.extend(AdaINResBlock1(ch, rk, rd, style_dim)
                                  for rk, rd in zip(resblock_kernel_sizes, resblock_dilation_sizes))
            if i + 1 < len(upsample_rates):
                stride = math.prod(upsample_rates[i + 1:])
                self.noise_convs.append(nn.Conv1d(gen_istft_n_fft + 2, ch, stride * 2, stride, (stride + 1) // 2))
                self.noise_res.append(AdaINResBlock1(ch, 7, (1, 3, 5), style_dim))
            else:
                self.noise_convs.append(nn.Conv1d(gen_istft_n_fft + 2, ch, 1))
                self.noise_res.append(AdaINResBlock1(ch, 11, (1, 3, 5), style_dim))
        self.conv_post = nn.Conv1d(ch, gen_istft_n_fft + 2, 7, padding=3)

    def forward(self, x, s, f0):
        f0 = F.interpolate(f0[:, None], scale_factor=self.f0_scale).transpose(1, 2)
        har = torch.cat(self.stft.transform(self.m_source(f0).squeeze(-1)), dim=1)
        last = len(self.ups) - 1
        for i, up in enumerate(self.ups):
            x = up(F.leaky_relu(x, 0.1))
            if i == last:
                x = F.pad(x, (1, 0), mode="reflect")
            x = x + self.noise_res[i](self.noise_convs[i](har), s)
            blocks = self.resblocks[i * self.n_kernels:(i + 1) * self.n_kernels]
            x = sum(block(x, s) for block in blocks) / self.n_kernels
        x = self.conv_post(F.leaky_relu(x))
        bins = self.post_n_fft // 2 + 1
        return self.stft.inverse(torch.exp(x[:, :bins]), torch.sin(x[:, bins:]))


class Decoder(nn.Module):
    def __init__(self, dim_in, style_dim, **istftnet):
        super().__init__()
        self.encode = AdainResBlk1d(dim_in + 2, 1024, style_dim)
        self.decode = nn.ModuleList([AdainResBlk1d(1024 + 2 + 64, 1024, style_dim) for _ in range(3)]
                                    + [AdainResBlk1d(1024 + 2 + 64, 512, style_dim, upsample=True)])
        self.F0_conv = nn.Conv1d(1, 1, 3, 2, 1)
        self.N_conv = nn.Conv1d(1, 1, 3, 2, 1)
        self.asr_res = nn.Sequential(nn.Conv1d(512, 64, 1))
        self.generator = Generator(style_dim, **istftnet)

    def forward(self, asr, f0_curve, n_curve, s):
        f0, n = self.F0_conv(f0_curve[:, None]), self.N_conv(n_curve[:, None])
        x = self.encode(torch.cat([asr, f0, n], dim=1), s)
        asr_res = self.asr_res(asr)
        for block in self.decode:
            x = block(torch.cat([x, asr_res, f0, n], dim=1), s)
        return self.generator(x, s, f0_curve)


def fold_weight_norm(sd):
    """The checkpoint stores weight-normed convs as weight_g/weight_v; inference only needs their product."""
    out = {}
    for k, v in sd.items():
        if k.endswith(".weight_v"):
            g = sd[k[:-1] + "g"]
            out[k[:-2]] = v * (g / v.flatten(1).norm(dim=1).view_as(g))
        elif not k.endswith(".weight_g"):
            out[k] = v
    return out


BERT_KEYS = {"encoder.embedding_hidden_mapping_in.": "embedding_hidden_mapping_in.",
             "encoder.albert_layer_groups.0.albert_layers.0.": "layer."}


class Kokoro(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab = config["vocab"]
        self.context_length = config["plbert"]["max_position_embeddings"]
        hidden, style = config["hidden_dim"], config["style_dim"]
        self.bert = Albert(config["n_token"], **config["plbert"])
        self.bert_encoder = nn.Linear(config["plbert"]["hidden_size"], hidden)
        self.predictor = ProsodyPredictor(style, hidden, config["n_layer"], config["max_dur"])
        self.text_encoder = TextEncoder(hidden, config["text_encoder_kernel_size"], config["n_layer"], config["n_token"])
        self.decoder = Decoder(hidden, style, **config["istftnet"])

    @classmethod
    def from_hub(cls, repo_id=REPO, device="cpu"):
        config = json.loads(Path(hf_hub_download(repo_id, "config.json")).read_text(encoding="utf-8"))
        model = cls(config)
        checkpoint = torch.load(hf_hub_download(repo_id, WEIGHTS), map_location="cpu", weights_only=True)
        for part, sd in checkpoint.items():
            sd = fold_weight_norm({k.removeprefix("module."): v for k, v in sd.items()})
            if part == "bert":
                sd = {next((k.replace(a, b) for a, b in BERT_KEYS.items() if k.startswith(a)), k): v
                      for k, v in sd.items() if not k.startswith("pooler.")}
            elif part == "predictor":
                sd = {k.replace("duration_proj.linear_layer.", "duration_proj."): v for k, v in sd.items()}
            getattr(model, part).load_state_dict(sd)
        model.repo_id = repo_id
        return model.to(device).eval()

    @property
    def device(self):
        return self.bert_encoder.weight.device

    def load_voice(self, name):
        """A voice pack: one style vector per phoneme-string length, shape (510, 1, 256)."""
        path = name if name.endswith(".pt") else hf_hub_download(self.repo_id, f"voices/{name}.pt")
        return torch.load(path, map_location=self.device, weights_only=True)

    @torch.inference_mode()
    def forward(self, phonemes, voice, speed=1.0):
        """Phoneme string to 24 kHz float audio on CPU. Symbols outside the vocab are dropped, but the style
        vector is still picked by the full string length, as upstream does."""
        ids = [self.vocab[p] for p in phonemes if p in self.vocab]
        if len(ids) + 2 > self.context_length:
            raise ValueError(f"{len(ids)} phonemes exceed the context of {self.context_length - 2}")
        ids = torch.tensor([[0, *ids, 0]], device=self.device)
        ref = voice[len(phonemes) - 1].to(self.device)
        timbre, prosody = ref[:, :128], ref[:, 128:]
        d = self.predictor.text_encoder(self.bert_encoder(self.bert(ids)), prosody)
        durations = torch.sigmoid(self.predictor.duration_proj(self.predictor.lstm(d)[0])).sum(-1) / speed
        frames = durations.round().clamp(min=1).long()[0]
        en = d.transpose(1, 2).repeat_interleave(frames, dim=2)
        f0, noise = self.predictor.f0_and_noise(en, prosody)
        asr = self.text_encoder(ids).repeat_interleave(frames, dim=2)
        return self.decoder(asr, f0, noise, timbre).squeeze().cpu()
