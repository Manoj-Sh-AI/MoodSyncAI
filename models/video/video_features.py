"""
models/visual/video_features.py
Frame extraction and preprocessing utilities for the video emotion model.
Mirrors the pipeline defined in notebooks/7_video_emotion_recognition_ravdess.ipynb
and reused in notebooks/8_video_text_fusion.ipynb.
"""

from typing import List, Optional

import av
import numpy as np
import torch
import torchvision.transforms as T

# ---------------------------------------------------------------------------
# ImageNet statistics used during RAVDESS training
# ---------------------------------------------------------------------------
IMAGENET_MEAN = (0.43216, 0.394666, 0.37645)
IMAGENET_STD  = (0.22803, 0.22145,  0.216989)

# Default hyper-parameters (overridden by video_model_meta.json at runtime)
DEFAULT_NUM_FRAMES = 16
DEFAULT_FRAME_SIZE = 112


def build_frame_transform(frame_size: int = DEFAULT_FRAME_SIZE) -> T.Compose:
    """Return the validation transform pipeline used during RAVDESS training.

    Applying the *same* normalisation at inference as during training is
    critical — any deviation shifts the feature distribution and degrades
    accuracy.
    """
    return T.Compose([
        T.ToPILImage(),
        T.Resize((frame_size, frame_size)),
        T.CenterCrop(frame_size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# PyAV / OpenCV helpers
# ---------------------------------------------------------------------------

def _decode_with_pyav(video_path: str, num_frames: int) -> List[np.ndarray]:
    """Uniformly sample frames via PyAV.  Returns RGB uint8 arrays (H, W, 3)."""
    container = av.open(video_path)
    stream = container.streams.video[0]
    total = stream.frames or int(
        stream.duration * float(stream.time_base) * stream.average_rate
    )
    indices = set(np.linspace(0, max(total - 1, 0), num_frames, dtype=int).tolist())
    frames_raw: List[np.ndarray] = []
    for i, frame in enumerate(container.decode(video=0)):
        if i in indices:
            frames_raw.append(frame.to_ndarray(format="rgb24"))
        if len(frames_raw) >= num_frames:
            break
    container.close()
    return frames_raw


def _decode_with_opencv(video_path: str, num_frames: int) -> List[np.ndarray]:
    """Fallback frame decoder using OpenCV (handles FLV and other codecs)."""
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required as a fallback decoder.  "
            "Install it with:  pip install opencv-python-headless"
        ) from exc

    cap = cv2.VideoCapture(video_path)
    all_frames: List[np.ndarray] = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        all_frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()

    if not all_frames:
        return []
    idx = np.linspace(0, len(all_frames) - 1, num_frames, dtype=int)
    return [all_frames[i] for i in idx]


def decode_uniform_frames(
    video_path: str,
    num_frames: int = DEFAULT_NUM_FRAMES,
) -> List[np.ndarray]:
    """Uniformly sample *num_frames* RGB frames from a video file.

    Tries PyAV first; falls back to OpenCV if PyAV returns zero frames or
    raises an exception (e.g. for .flv containers).

    Args:
        video_path: Path to the video file.
        num_frames: Number of frames to sample.

    Returns:
        List of (H, W, 3) uint8 NumPy arrays.  May be shorter than
        *num_frames* if the video is very short; the caller is responsible
        for padding if needed.
    """
    try:
        frames = _decode_with_pyav(video_path, num_frames)
        if frames:
            return frames
    except Exception:
        pass

    # Fallback to OpenCV
    try:
        return _decode_with_opencv(video_path, num_frames)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Tensor builder
# ---------------------------------------------------------------------------

def frames_to_tensor(
    frames_rgb: List[np.ndarray],
    transform: Optional[T.Compose] = None,
    num_frames: int = DEFAULT_NUM_FRAMES,
    frame_size: int = DEFAULT_FRAME_SIZE,
) -> torch.Tensor:
    """Convert a list of raw RGB frames to a model-ready clip tensor.

    Args:
        frames_rgb: Raw uint8 NumPy arrays (H, W, 3).
        transform:  Optional pre-built transform.  Built on-the-fly if None.
        num_frames: Target temporal length; frames are sub-sampled uniformly.
        frame_size: Spatial size used to build the transform when *transform*
                    is not provided.

    Returns:
        Tensor of shape (1, 3, T, H, W).
    """
    if transform is None:
        transform = build_frame_transform(frame_size)

    n = len(frames_rgb)
    if n == 0:
        raise ValueError("frames_rgb is empty — no frames to convert.")

    # Uniform sub-sample to exactly num_frames
    indices = np.linspace(0, n - 1, num_frames, dtype=int)
    sampled = [frames_rgb[i] for i in indices]

    # Pad by repeating the last frame if needed
    while len(sampled) < num_frames:
        sampled.append(sampled[-1])

    tensors = [transform(f) for f in sampled]           # each (3, H, W)
    clip = torch.stack(tensors, dim=1)                  # (3, T, H, W)
    return clip.unsqueeze(0)                            # (1, 3, T, H, W)


# ---------------------------------------------------------------------------
# Convenience: path → tensor
# ---------------------------------------------------------------------------

def extract_frames(
    video_path: str,
    num_frames: int = DEFAULT_NUM_FRAMES,
    transform: Optional[T.Compose] = None,
    frame_size: int = DEFAULT_FRAME_SIZE,
) -> torch.Tensor:
    """Decode a video file and return a model-ready clip tensor.

    Args:
        video_path: Path to the .mp4 / .flv / etc. file.
        num_frames: Number of frames to sample uniformly.
        transform: Optional pre-built torchvision transform.
        frame_size: Spatial crop size used when *transform* is None.

    Returns:
        Tensor of shape (1, 3, T, H, W).
    """
    frames = decode_uniform_frames(video_path, num_frames)

    if not frames:
        raise ValueError(f"Could not decode any frames from: {video_path}")

    return frames_to_tensor(
        frames,
        transform=transform,
        num_frames=num_frames,
        frame_size=frame_size,
    )
