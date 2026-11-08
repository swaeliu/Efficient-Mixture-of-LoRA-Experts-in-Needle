"""Phase 4C — adapter injection into the tiny transformer: freezing, equivalences, training."""
import numpy as np
import pytest

import needle as ndl
from models import (TinyTransformer, adapter_layers, adapter_trainable_parameters, count_parameters, freeze_backbone,
                    inject_adapters, named_parameters, routing_stats, sparse_execution_stats)
from molora import DenseMoLoRALinear, LoRALinear, SparseMoLoRALinear, trainable_parameters


def _backbone(seed=0, dtype="float32"):
    np.random.seed(seed)
    m = TinyTransformer(vocab_size=11, d_model=8, n_heads=2, n_layers=2, d_ff=16, max_seq_len=6, dtype=dtype)
    freeze_backbone(m)
    return m


def _data(seed=0):
    rng = np.random.default_rng(seed)
    tok = rng.integers(0, 11, (4, 6))
    return tok, np.roll(tok, -1, axis=1)


@pytest.mark.parametrize("kind", ["lora", "dense", "sparse"])
def test_injection_preserves_initial_outputs_and_shares_weights(kind):
    m = _backbone()
    tok, _ = _data()
    ref = m(tok).numpy()
    fc1_w = m.blocks[0].ffn.fc1.weight
    np.random.seed(1)
    layers = inject_adapters(m, kind, rank=2, alpha=4.0, n_experts=3, top_k=2)
    assert len(layers) == 4  # fc1, fc2 x 2 blocks
    assert all(isinstance(l, {"lora": LoRALinear, "dense": DenseMoLoRALinear, "sparse": SparseMoLoRALinear}[kind]) for l in layers)
    assert np.shares_memory(m.blocks[0].ffn.fc1.weight.cached_data, fc1_w.cached_data)
    np.testing.assert_allclose(m(tok).numpy(), ref, rtol=1e-6, atol=1e-6)  # zero-B init -> identical logits


def test_only_adapter_parameters_are_trainable_and_counted():
    m = _backbone()
    n_backbone = count_parameters(m.parameters())
    np.random.seed(2)
    inject_adapters(m, "sparse", rank=2, alpha=4.0, n_experts=4, top_k=2)
    train = adapter_trainable_parameters(m)
    assert {id(p) for p in train} == {id(p) for p in trainable_parameters(m)}
    frozen = [p for p in m.parameters() if not p.requires_grad]
    assert count_parameters(frozen) == n_backbone
    # 2 blocks x (fc1: in 8 out 16, fc2: in 16 out 8): adapters E r (in+out) + router (in E + E)
    expected = 2 * ((4 * 2 * 24 + 8 * 4 + 4) + (4 * 2 * 24 + 16 * 4 + 4))
    assert count_parameters(train) == expected
    assert count_parameters(m.parameters()) == n_backbone + expected


@pytest.mark.parametrize("kind", ["lora", "dense", "sparse"])
def test_frozen_backbone_unchanged_after_adapter_training(kind):
    m = _backbone()
    np.random.seed(3)
    inject_adapters(m, kind, rank=2, alpha=4.0, n_experts=3, top_k=2)
    before = {n: p.numpy().copy() for n, p in named_parameters(m) if not p.requires_grad}
    assert len(before) >= 30
    tok, tgt = _data(1)
    params = adapter_trainable_parameters(m)
    opt = ndl.optim.Adam(params, lr=1e-2)
    losses = []
    for _ in range(4):
        opt.reset_grad()
        loss = m.loss(m(tok), tgt)
        loss.backward()
        opt.step()
        losses.append(float(loss.numpy()))
    after = dict(named_parameters(m))
    for n, arr in before.items():
        np.testing.assert_array_equal(after[n].numpy(), arr, err_msg=n)
    assert all(np.isfinite(losses)) and losses[-1] < losses[0]
    moved = []
    for l in adapter_layers(m):
        if isinstance(l, DenseMoLoRALinear):
            moved.append(any(np.any(e.lora_B.numpy() != 0) for e in l.experts))
        else:
            moved.append(bool(np.any(l.lora_B.numpy() != 0)))
    assert any(moved)


def _pair(kind_a, kind_b, kwa, kwb, dtype="float64"):
    ma, mb = _backbone(seed=4, dtype=dtype), _backbone(seed=4, dtype=dtype)
    np.random.seed(5)
    la = inject_adapters(ma, kind_a, rank=2, alpha=4.0, n_experts=3, **kwa)
    np.random.seed(5)
    lb = inject_adapters(mb, kind_b, rank=2, alpha=4.0, n_experts=3, **kwb)
    rng = np.random.default_rng(6)
    for a, b in zip(la, lb):  # make experts and routers non-zero and identical
        for i in range(3):
            A, B = rng.standard_normal(a.experts[i].lora_A.shape), rng.standard_normal(a.experts[i].lora_B.shape)
            a.load_expert(i, A=A, B=B)
            b.load_expert(i, A=A, B=B)
        W, bb = rng.standard_normal(a.router.weight.shape), rng.standard_normal(a.router.bias.shape)
        a.load_router(weight=W, bias=bb)
        b.load_router(weight=W, bias=bb)
    return ma, mb


def _grads(m, tok, tgt):
    loss = m.loss(m(tok), tgt)
    loss.backward()
    return float(loss.numpy()), [p.grad.numpy() if p.grad is not None else None for p in adapter_trainable_parameters(m)]


def test_dense_equals_sparse_k_equals_E_inside_transformer():
    ma, mb = _pair("dense", "sparse", {}, {"top_k": 3})
    tok, tgt = _data(2)
    np.testing.assert_allclose(mb(tok).numpy(), ma(tok).numpy(), rtol=1e-12, atol=1e-12)
    la, ga = _grads(ma, tok, tgt)
    lb, gb = _grads(mb, tok, tgt)
    np.testing.assert_allclose(lb, la, rtol=1e-12)
    for a, b in zip(ga, gb):
        np.testing.assert_allclose(b, a, rtol=1e-10, atol=1e-11)


def test_reference_equals_dispatch_inside_transformer():
    ma, mb = _pair("sparse", "sparse", {"top_k": 2, "execution_mode": "reference"}, {"top_k": 2, "execution_mode": "dispatch"})
    tok, tgt = _data(3)
    np.testing.assert_allclose(mb(tok).numpy(), ma(tok).numpy(), rtol=1e-12, atol=1e-12)
    la, ga = _grads(ma, tok, tgt)
    lb, gb = _grads(mb, tok, tgt)
    np.testing.assert_allclose(lb, la, rtol=1e-12)
    for a, b in zip(ga, gb):
        np.testing.assert_allclose(b, a, rtol=1e-10, atol=1e-11)
    st = sparse_execution_stats(mb)
    assert st["layers"] == 4 and st["expert_flop_fraction_vs_dense"] == pytest.approx(2 / 3)


def test_routing_stats_and_reproducible_finetune():
    def run():
        m = _backbone(seed=7)
        np.random.seed(8)
        inject_adapters(m, "sparse", rank=2, alpha=4.0, n_experts=4, top_k=2)
        tok, tgt = _data(4)
        opt = ndl.optim.Adam(adapter_trainable_parameters(m), lr=1e-2)
        for _ in range(3):
            opt.reset_grad()
            m.loss(m(tok), tgt).backward()
            opt.step()
        return np.concatenate([p.numpy().ravel() for p in adapter_trainable_parameters(m)]), routing_stats(m, tok)

    pa, sa = run()
    pb, sb = run()
    np.testing.assert_array_equal(pa, pb)
    assert len(sa) == 4 and all(abs(sum(s["utilization"]) - 2.0) < 1e-5 for s in sa)  # float32 utilisation
    assert all(0 <= s["entropy_nats"] <= s["max_entropy_nats"] + 1e-12 for s in sa)
    assert sa == sb


def test_inject_twice_raises_and_attn_targets_work():
    m = _backbone()
    inject_adapters(m, "lora", rank=2, where=("ffn", "attn_qv"))
    assert len(adapter_layers(m)) == 8
    with pytest.raises(TypeError):
        inject_adapters(m, "lora", rank=2)
    with pytest.raises(ValueError):
        inject_adapters(_backbone(), "magic")
    tok, _ = _data()
    assert m(tok).shape == (4, 6, 11)
