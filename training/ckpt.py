"""Gradient checkpointing for the GCT global (cross-frame) blocks.

The released code only checkpoints the per-frame blocks, because a global
block writes its keys/values into the shared KV-cache dict, and recomputing
it in the backward pass would append them a second time.  Here each global
block runs as a pure function of (tokens, its cache entries before the call)
on a private copy of the cache dict, returning the new entries, which are
then written back.  Recomputation therefore sees exactly what the forward saw.
"""
import types

from torch.utils.checkpoint import checkpoint


def enable(aggregator):
    for blk in aggregator.global_blocks:
        if getattr(blk, "_ckpt_orig", None) is not None:
            continue
        blk._ckpt_orig = blk.forward
        blk.forward = types.MethodType(_forward, blk)


def _forward(self, x, kv_cache=None, global_idx=0, **kw):
    if not (self.training and kv_cache is not None and isinstance(kv_cache, dict)):
        return self._ckpt_orig(x, kv_cache=kv_cache, global_idx=global_idx, **kw)
    keys = [f"k_{global_idx}", f"v_{global_idx}", f"k_{global_idx}_special",
            f"v_{global_idx}_special"]
    before = [kv_cache.get(k) for k in keys]
    skip = kv_cache.get("_skip_append", False)

    def run(x, *prev):
        local = dict(zip(keys, prev))
        local["_skip_append"] = skip
        out = self._ckpt_orig(x, kv_cache=local, global_idx=global_idx, **kw)
        return (out,) + tuple(local.get(k) for k in keys)

    res = checkpoint(run, x, *before, use_reentrant=False)
    for k, v in zip(keys, res[1:]):
        kv_cache[k] = v
    return res[0]


def enable_depth_head(model, chunk=2):
    """The DPT depth head works frame by frame; run it in checkpointed chunks
    of frames so its full-resolution activations are never all held at once."""
    import torch
    orig = model._predict_depth

    def run(tokens, images, patch_start_idx, gather_outputs=True):
        if not model.training:
            return orig(tokens, images, patch_start_idx, gather_outputs)
        S = images.shape[1]
        outs = []
        for s in range(0, S, chunk):
            t = [x[:, s:s + chunk] for x in tokens]
            r = checkpoint(lambda im, *tt: tuple(orig(list(tt), im, patch_start_idx).values()),
                           images[:, s:s + chunk], *t, use_reentrant=False)
            outs.append(r)
        return {"depth": torch.cat([o[0] for o in outs], 1),
                "depth_conf": torch.cat([o[1] for o in outs], 1)}

    model._predict_depth = run
