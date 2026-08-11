"""ML milestone tests: dataset tensors, policy shapes, training, agent."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from dataset.build import DatasetConfig, build_dataset  # noqa: E402
from ml.dataset import ImitationDataset  # noqa: E402
from ml.policy import ImitationPolicy, load_checkpoint, save_checkpoint  # noqa: E402
from tests.fakes import build_synthetic_recording  # noqa: E402


@pytest.fixture()
def dataset_dir(tmp_path):
    recording = build_synthetic_recording(
        tmp_path / "recordings",
        n_frames=60,
        fps=10,
        events=[
            {"t": 0.15, "device": "keyboard", "event": "down", "code": "KeyW"},
            {"t": 0.75, "device": "keyboard", "event": "up", "code": "KeyW"},
            {"t": 0.5, "device": "mouse", "event": "move", "x": 100, "y": 50},
            {"t": 2.0, "device": "mouse", "event": "move", "x": 130, "y": 55},
            {"t": 3.0, "device": "mouse", "event": "button_down", "button": "left"},
            {"t": 3.4, "device": "mouse", "event": "button_up", "button": "left"},
        ],
    )
    out = tmp_path / "dataset"
    build_dataset(recording, DatasetConfig(observation_duration=0.4, fps=5, stride=0.1), out)
    return out


def test_dataset_tensor_shapes(dataset_dir):
    ds = ImitationDataset(dataset_dir)
    assert len(ds) == json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))["count"]
    assert ds.keys == ["KeyW"]
    assert ds.buttons == ["left"]
    sample = ds[0]
    obs = sample["observation"]
    assert obs.dtype == torch.float32
    assert obs.ndim == 4 and obs.shape[0] == 3  # 0.4s @ 5fps -> 3 frames
    assert float(obs.min()) >= 0.0 and float(obs.max()) <= 1.0
    assert sample["keys"].shape == (1,)
    assert sample["buttons"].shape == (1,)
    assert sample["dx"].shape == ()
    assert sample["mouse_valid"] in (0.0, 1.0)


def test_dataset_onehot_and_mouse_mask(dataset_dir):
    ds = ImitationDataset(dataset_dir)
    for index in range(len(ds)):
        sample = ds[index]
        t = float(sample["t"])
        assert bool(sample["keys"][0]) == (0.15 <= t < 0.75)
        assert bool(sample["buttons"][0]) == (3.0 <= t < 3.4)
    # mouse samples: a nonzero delta exists after the second move
    deltas = [
        (float(s["dx"]), float(s["dy"]))
        for s in (ImitationDataset(dataset_dir)[i] for i in range(len(ds)))
        if float(s["mouse_valid"])
    ]
    assert deltas, "expected at least one valid mouse sample"
    assert (30.0, 5.0) in deltas  # move (100,50) -> (130,55)


def test_dataset_vocab_override(dataset_dir):
    ds = ImitationDataset(dataset_dir, keys=["KeyW", "KeyD"], buttons=["left", "right"])
    assert ds.n_keys == 2 and ds.n_buttons == 2
    sample = ds[0]
    assert sample["keys"].shape == (2,)
    assert sample["buttons"].shape == (2,)


def test_policy_forward_shapes(dataset_dir):
    ds = ImitationDataset(dataset_dir)
    policy = ImitationPolicy(ds.n_keys, ds.n_buttons, hidden=16)
    batch = torch.stack([ds[i]["observation"] for i in range(4)])
    out = policy(batch)
    assert out.keys_logits.shape == (4, ds.n_keys)
    assert out.buttons_logits.shape == (4, ds.n_buttons)
    assert out.dx.shape == (4,) and out.dy.shape == (4,)
    assert torch.isfinite(out.dx).all()
    assert torch.isfinite(out.keys_logits).all()


def test_train_saves_checkpoint(dataset_dir, tmp_path, capsys):
    from ml.train import main as train_main

    ckpt = tmp_path / "ckpt.pt"
    code = train_main(
        ["--dataset", str(dataset_dir), "--out", str(ckpt),
         "--epochs", "3", "--batch-size", "4", "--hidden", "16", "--lr", "0.01"]
    )
    assert code == 0
    assert ckpt.exists()
    policy, data = load_checkpoint(ckpt)
    assert data["vocab"]["keys"] == ["KeyW"]
    assert data["vocab"]["buttons"] == ["left"]
    assert policy.n_keys == 1 and policy.n_buttons == 1
    losses = [
        float(line.split("loss=")[1].split()[0])
        for line in capsys.readouterr().out.splitlines()
        if "loss=" in line
    ]
    assert len(losses) == 3
    assert losses[-1] < losses[0]  # behavior cloning actually learns


def test_train_union_vocab(tmp_path, dataset_dir, capsys):
    from ml.train import main as train_main

    other = tmp_path / "other"
    recording = build_synthetic_recording(
        other / "recordings",
        n_frames=30,
        fps=10,
        events=[{"t": 0.1, "device": "keyboard", "event": "down", "code": "KeyD"},
                {"t": 0.6, "device": "keyboard", "event": "up", "code": "KeyD"}],
    )
    other_ds = other / "dataset"
    build_dataset(recording, DatasetConfig(observation_duration=0.4, fps=5, stride=0.1), other_ds)
    ckpt = tmp_path / "union.pt"
    code = train_main(
        ["--dataset", str(dataset_dir), "--dataset", str(other_ds),
         "--out", str(ckpt), "--epochs", "1", "--batch-size", "4", "--hidden", "16"]
    )
    assert code == 0
    _, data = load_checkpoint(ckpt)
    assert data["vocab"]["keys"] == ["KeyD", "KeyW"]


def test_train_missing_dataset_fails(tmp_path, capsys):
    from ml.train import main as train_main

    assert train_main(["--dataset", str(tmp_path / "nope"), "--out", str(tmp_path / "x.pt")]) == 1
    assert "error:" in capsys.readouterr().err


def test_checkpoint_roundtrip(dataset_dir, tmp_path):
    ds = ImitationDataset(dataset_dir)
    policy = ImitationPolicy(ds.n_keys, ds.n_buttons, hidden=16)
    save_checkpoint(policy, tmp_path / "c.pt", ds.keys, ds.buttons, {"epochs": 1})
    loaded, data = load_checkpoint(tmp_path / "c.pt")
    assert data["vocab"] == {"keys": ["KeyW"], "buttons": ["left"]}
    sample = ds[0]
    out = loaded(sample["observation"].unsqueeze(0))
    assert torch.isfinite(out.dx[0])


def test_agent_dry_run(dataset_dir, tmp_path, capsys):
    from ml.inject import run_agent
    from ml.train import main as train_main

    ckpt = tmp_path / "ckpt.pt"
    train_main(["--dataset", str(dataset_dir), "--out", str(ckpt),
                "--epochs", "1", "--batch-size", "4", "--hidden", "16"])
    code = run_agent(["--checkpoint", str(ckpt), "--dataset", str(dataset_dir), "--max-samples", "3"])
    assert code == 0
    output = capsys.readouterr().out
    assert "mode: dry run" in output
    assert output.count("t=") == 3


def test_agent_key_code_mapping():
    from ml.inject import _code_to_key

    from pynput.keyboard import Key, KeyCode

    assert _code_to_key("Key.space") is Key.space
    assert isinstance(_code_to_key("KeyW"), KeyCode)
    assert _code_to_key("Key.vk_65").vk == 65
