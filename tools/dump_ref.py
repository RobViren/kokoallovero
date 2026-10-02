"""Torch intermediates for stage-by-stage parity on one phoneme string (a line of kkg2p output).
Usage: dump_ref.py <phonemes.txt>
Writes weights/ref/*.npy, weights/ref/phonemes.txt, and weights/ref/ulp/*.npy: the same run with the style vector
nudged by one float32 ulp and identical noise, which is the parity floor for everything downstream of the style."""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ref"))
from kokoro import Kokoro  # noqa: E402

SEED = 1234


def generator(g, x, s, f0, D):
    f0 = F.interpolate(f0[:, None], scale_factor=g.f0_scale).transpose(1, 2)
    D["source"] = g.m_source(f0).squeeze(-1)
    har = D["har"] = torch.cat(g.stft.transform(D["source"]), dim=1)
    last = len(g.ups) - 1
    for i, up in enumerate(g.ups):
        x = up(F.leaky_relu(x, 0.1))
        if i == last:
            x = F.pad(x, (1, 0), mode="reflect")
        D[f"gen.up{i}"] = x
        nc = D[f"gen.noiseconv{i}"] = g.noise_convs[i](har)
        n = D[f"gen.noise{i}"] = g.noise_res[i](nc, s)
        x = x + n
        blocks = g.resblocks[i * g.n_kernels:(i + 1) * g.n_kernels]
        x = D[f"gen.res{i}"] = sum(block(x, s) for block in blocks) / g.n_kernels
    x = D["conv_post"] = g.conv_post(F.leaky_relu(x))
    bins = g.post_n_fft // 2 + 1
    return g.stft.inverse(torch.exp(x[:, :bins]), torch.sin(x[:, bins:]))


def synth(model, ps, voice, D):
    ids = torch.tensor([[0, *[model.vocab[p] for p in ps if p in model.vocab], 0]])
    ref = voice[len(ps) - 1]
    timbre, prosody = ref[:, :128], ref[:, 128:]
    pr, dec = model.predictor, model.decoder
    D["bert"] = model.bert(ids)
    D["bert_enc"] = model.bert_encoder(D["bert"])
    d = D["dur_enc"] = pr.text_encoder(D["bert_enc"], prosody)
    D["durations"] = torch.sigmoid(pr.duration_proj(pr.lstm(d)[0])).sum(-1)
    frames = D["frames"] = D["durations"].round().clamp(min=1).long()[0]
    en = d.transpose(1, 2).repeat_interleave(frames, dim=2)
    D["f0"], D["n"] = pr.f0_and_noise(en, prosody)
    D["text_enc"] = model.text_encoder(ids)
    asr = D["text_enc"].repeat_interleave(frames, dim=2)
    f0, n = dec.F0_conv(D["f0"][:, None]), dec.N_conv(D["n"][:, None])
    x = D["dec.encode"] = dec.encode(torch.cat([asr, f0, n], dim=1), timbre)
    asr_res = dec.asr_res(asr)
    for i, block in enumerate(dec.decode):
        x = D[f"dec.decode{i}"] = block(torch.cat([x, asr_res, f0, n], dim=1), timbre)
    return generator(dec.generator, x, timbre, D["f0"], D)


def capture_noise(D):
    rand, randn_like = torch.rand, torch.randn_like

    def rand_(*a, **k):
        D["initial_phase"] = r = rand(*a, **k)
        return r

    def randn_like_(*a, **k):
        D["noise"] = r = randn_like(*a, **k)
        return r

    torch.rand, torch.randn_like = rand_, randn_like_
    return lambda: setattr(torch, "rand", rand) or setattr(torch, "randn_like", randn_like)


def save(out, D):
    out.mkdir(parents=True, exist_ok=True)
    for k, v in D.items():
        a = np.ascontiguousarray(v.detach().numpy())
        np.save(out / f"{k}.npy", np.ascontiguousarray(a.squeeze(0) if a.ndim > 1 and a.shape[0] == 1 else a))


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    model = Kokoro.from_hub()
    voice = model.load_voice("af_heart")
    ps = Path(sys.argv[1]).read_text(encoding="utf-8").strip()[:510]
    out = ROOT / "weights" / "ref"
    for name, v in (("", voice), ("ulp", voice * (1 + 2 ** -23))):
        D = {}
        restore = capture_noise(D)
        torch.manual_seed(SEED)
        D["audio"] = synth(model, ps, v, D)
        restore()
        D["initial_phase"][:, 0] = 0
        torch.manual_seed(SEED)
        assert torch.equal(D["audio"][0], model(ps, v)), "instrumented forward drifted from ref/kokoro.py"
        save(out / name, D)
    (out / "phonemes.txt").write_text(ps + "\n", encoding="utf-8")
    print(f"phonemes {len(ps)} frames {int(D['frames'].sum())} samples {D['audio'].shape[-1]}: {ps}")


main()
