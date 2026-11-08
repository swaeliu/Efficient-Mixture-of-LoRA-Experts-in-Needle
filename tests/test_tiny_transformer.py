"""Phase 4C — Needle-native tiny transformer: shapes, causality, gradients, checkpoints, dtype."""
import os
import tempfile

import numpy as np
import pytest

import needle as ndl
from models import TinyTransformer, named_parameters, load_state, save_state
from models.tiny_transformer import CausalSelfAttention, LayerNorm


def _model(seed=0, **kw):
    np.random.seed(seed)
    cfg = dict(vocab_size=11, d_model=8, n_heads=2, n_layers=2, d_ff=16, max_seq_len=6)
    cfg.update(kw)
    return TinyTransformer(**cfg)


def test_shapes_and_dtype():
    m = _model()
    tok = np.random.default_rng(0).integers(0, 11, (3, 6))
    logits = m(tok)
    assert logits.shape == (3, 6, 11) and logits.dtype == np.float32
    loss = m.loss(logits, tok)
    assert loss.shape == () and loss.dtype == np.float32 and np.isfinite(loss.numpy())
    assert 0.0 <= m.accuracy(logits, tok) <= 1.0


def test_attention_output_shape_and_causal_mask():
    np.random.seed(1)
    att = CausalSelfAttention(8, 2)
    x = ndl.Tensor(np.random.randn(2, 5, 8).astype(np.float32), dtype="float32", requires_grad=False)
    y = att(x)
    assert y.shape == (2, 5, 8)
    # changing the future must not change the past (exactly: masked weights are exactly zero)
    x2 = x.numpy().copy()
    x2[:, 3:, :] += 10.0
    y2 = att(ndl.Tensor(x2, dtype="float32", requires_grad=False))
    np.testing.assert_array_equal(y.numpy()[:, :3], y2.numpy()[:, :3])
    assert not np.allclose(y.numpy()[:, 3:], y2.numpy()[:, 3:])


def test_model_is_causal_end_to_end():
    m = _model()
    rng = np.random.default_rng(2)
    tok = rng.integers(0, 11, (2, 6))
    tok2 = tok.copy()
    tok2[:, 4:] = (tok2[:, 4:] + 1) % 11
    l1, l2 = m(tok).numpy(), m(tok2).numpy()
    np.testing.assert_array_equal(l1[:, :4], l2[:, :4])
    assert not np.allclose(l1[:, 4:], l2[:, 4:])


def test_layernorm_matches_numpy_on_3d():
    np.random.seed(3)
    ln = LayerNorm(8)
    x = np.random.randn(2, 5, 8).astype(np.float32)
    y = ln(ndl.Tensor(x, dtype="float32", requires_grad=False)).numpy()
    ref = (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5)
    np.testing.assert_allclose(y, ref, rtol=1e-4, atol=1e-5)


def test_sequence_length_limit_and_index_check():
    m = _model()
    with pytest.raises(ValueError):
        m(np.zeros((1, 7), int))
    with pytest.raises(IndexError):
        m(np.full((1, 3), 11))


def _fd(f, arr, idx_list, eps=1e-6):
    g = {}
    for idx in idx_list:
        ap, am = arr.copy(), arr.copy()
        ap[idx] += eps
        am[idx] -= eps
        g[idx] = (f(ap) - f(am)) / (2 * eps)
    return g


def test_finite_difference_gradients_float64():
    np.random.seed(4)
    m = TinyTransformer(vocab_size=5, d_model=4, n_heads=2, n_layers=1, d_ff=6, max_seq_len=3, dtype="float64")
    tok = np.random.default_rng(4).integers(0, 5, (2, 3))
    tgt = np.roll(tok, -1, axis=1)
    loss = m.loss(m(tok), tgt)
    loss.backward()
    params = dict(named_parameters(m))
    for name in ("tok_emb.weight", "blocks.0.attn.q_proj.weight", "blocks.0.attn.v_proj.bias", "blocks.0.ffn.fc2.weight",
                 "blocks.0.ln1.ln.weight", "lm_head.weight"):
        p = params[name]
        base = p.numpy().copy()
        rng = np.random.default_rng(5)
        idxs = [tuple(rng.integers(0, s) for s in base.shape) for _ in range(6)]

        def f(arr, p=p):
            p.data = ndl.Tensor(arr, dtype="float64", requires_grad=False)
            v = float(m.loss(m(tok), tgt).numpy())
            p.data = ndl.Tensor(base, dtype="float64", requires_grad=False)
            return v

        fd = _fd(f, base, idxs)
        for idx, g in fd.items():
            np.testing.assert_allclose(p.grad.numpy()[idx], g, rtol=1e-5, atol=1e-7, err_msg=name)


def test_training_reduces_loss_and_is_deterministic():
    def run():
        m = _model(seed=6)
        rng = np.random.default_rng(6)
        tok = rng.integers(0, 11, (8, 6))
        tgt = np.roll(tok, -1, axis=1)
        opt = ndl.optim.Adam(m.parameters(), lr=1e-2)
        losses = []
        for _ in range(15):
            opt.reset_grad()
            loss = m.loss(m(tok), tgt)
            loss.backward()
            opt.step()
            losses.append(float(loss.numpy()))
        return losses

    a, b = run(), run()
    assert a[-1] < a[0] and all(np.isfinite(a))
    np.testing.assert_array_equal(a, b)


def test_save_load_roundtrip():
    m = _model(seed=7)
    tok = np.random.default_rng(7).integers(0, 11, (2, 6))
    ref = m(tok).numpy()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.npz")
        save_state(m, path)
        m2 = _model(seed=8)
        assert not np.allclose(m2(tok).numpy(), ref)
        load_state(m2, path)
        np.testing.assert_array_equal(m2(tok).numpy(), ref)
        names = [n for n, _ in named_parameters(m)]
        assert len(names) == len(set(names)) and "blocks.1.ffn.fc1.weight" in names
        with pytest.raises(KeyError):
            load_state(_model(n_layers=1), path)
