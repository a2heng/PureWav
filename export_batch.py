"""Export purevox6 core denoiser to ONNX.

Pipeline: STFT(B,2,T,F) → ERB compress → U-Net → ERB expand → sigmoid → mask * STFT
STFT/ISTFT stays in numpy.
"""
import os, sys, types

CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    'models', 'lightweight-denoise-48k', 'checkpoint_epoch_14.tar')
OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      'v6_erb_skip_proj_batch.onnx')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'models', 'lightweight-denoise-48k'))

import numpy
if not hasattr(numpy, '_core'):
    import numpy.core as _core
    shim = types.ModuleType('numpy._core')
    for attr in dir(_core):
        setattr(shim, attr, getattr(_core, attr))
    sys.modules['numpy._core'] = shim
    sys.modules['numpy._core.multiarray'] = _core.multiarray
    sys.modules['numpy._core.umath'] = _core.umath

import torch, torch.nn as nn
from torch.utils._config_module import ConfigModule
_o = ConfigModule.__setattr__
def _s(self, n, v):
    try: _o(self, n, v)
    except AttributeError: object.__setattr__(self, n, v)
ConfigModule.__setattr__ = _s


class DenoiseWrapper(nn.Module):
    """STFT(B,2,T,F) → mask * STFT(B,2,T,F)

    Input is real STFT after view_as_real+permute: (B, 2, T, F)
    where dim=1 is [real, imag].
    """
    def __init__(self, model):
        super().__init__()
        self.erb = model.erb
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.dpgrnn0 = model.dpgrnn0
        self.dpgrnn1 = model.dpgrnn1

    def forward(self, spec_ri):
        # spec_ri: (B, 2, T, F) — real/imag stacked on dim=1
        feat = torch.log10(torch.norm(spec_ri, dim=1, keepdim=True).clamp(1e-12))
        # feat: (B, 1, T, F)
        feat = self.erb.bm(feat)
        # feat: (B, 1, T, 141)
        out, skips = self.encoder(feat)
        x = self.dpgrnn0(out)
        x = self.dpgrnn1(x)
        mask = self.decoder(x, skips)
        mask = self.erb.bs(mask)
        mask = torch.sigmoid(mask)
        # mask: (B, 1, T, F) → broadcast multiply
        return spec_ri * mask


def main():
    from v6_erb_skip_proj import purevox6

    model = purevox6()
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    sd = {k.replace('_orig_mod.', ''): v for k, v in ck['model'].items()}
    model.load_state_dict(sd)
    model.eval()

    wrapper = DenoiseWrapper(model).eval()
    dummy = torch.randn(1, 2, 100, 481)

    torch.onnx.export(
        wrapper, (dummy,), OUTPUT,
        input_names=['spec'], output_names=['enhanced_spec'],
        opset_version=14,
        dynamic_axes={'spec': {2: 'frames'}, 'enhanced_spec': {2: 'frames'}},
    )
    print(f"Exported: {OUTPUT} ({os.path.getsize(OUTPUT)/1024:.0f} KB)")


if __name__ == '__main__':
    main()
