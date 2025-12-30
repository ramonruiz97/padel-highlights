from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from dotenv import load_dotenv

# Load environment variables from .env at repo root if present.
REPO_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(REPO_ROOT / ".env", override=False)


@dataclass
class BallDetection:
    frame_index: int
    visible: bool
    x: int
    y: int


class TrackVNETHandler:
    """
    TrackVNET wrapper around TrackNetV2 to load weights and run inference.
    """

    def __init__(
        self,
        weights_path: Optional[Path] = None,
        device: Optional[str] = None,
        image_size: Tuple[int, int] = (288, 512),
    ) -> None:
        self.image_size = image_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.external_root = self._get_external_root()
        self.weights_path = self._resolve_weights(weights_path)
        self.model = self._load_model()

    def _load_model(self) -> torch.nn.Module:
        import sys

        if str(self.external_root) not in sys.path:
            sys.path.append(str(self.external_root))

        from models.tracknet import TrackNet  # imported from external package

        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"TrackNetV2 weights not found at {self.weights_path}. "
                "Provide a valid path via weights_path."
            )

        model = TrackNet().to(self.device)
        state = torch.load(self.weights_path, map_location=self.device)
        model.load_state_dict(state)
        model.eval()
        return model

    def _get_external_root(self) -> Path:
        env_root = os.getenv("TRACKNETV2_ROOT")
        if not env_root:
            raise ValueError(
                "TRACKNETV2_ROOT is not set. Define it in .env or environment to point to the TrackNetV2 repo."
            )
        return Path(env_root)

    def _resolve_weights(self, weights_path: Optional[Path]) -> Path:
        if weights_path:
            return Path(weights_path)
        env_weights = os.getenv("TRACKNETV2_WEIGHTS")
        if not env_weights:
            raise ValueError(
                "TRACKNETV2_WEIGHTS is not set and no weights_path was provided. "
                "Define it in .env or pass weights_path explicitly."
            )
        return Path(env_weights)

    def _frames_to_tensor(self, frames: Sequence[np.ndarray]) -> torch.Tensor:
        tensors = []
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = TF.to_tensor(rgb)
            tensor = TF.resize(tensor, self.image_size, antialias=True)
            tensors.append(tensor)
        return torch.cat(tensors, dim=0).unsqueeze(0).to(self.device)

    def predict_triplet(
        self, frames: Sequence[np.ndarray], frame_start_index: int, frame_size: Tuple[int, int]
    ) -> List[BallDetection]:
        """
        Run inference on exactly 3 consecutive frames (TrackNetV2 expects 3).
        Returns one BallDetection per input frame.
        """
        if len(frames) != 3:
            raise ValueError("TrackNetV2 expects exactly 3 consecutive frames.")

        tensor = self._frames_to_tensor(frames)
        preds = self.model(tensor)[0].detach().cpu().numpy()

        from utils.general import get_shuttle_position

        width, height = frame_size
        detections: List[BallDetection] = []
        for i in range(3):
            visible, cx_pred, cy_pred = get_shuttle_position(preds[i] > 0.5)
            cx = int(cx_pred * width / self.image_size[1])
            cy = int(cy_pred * height / self.image_size[0])
            detections.append(
                BallDetection(
                    frame_index=frame_start_index + i,
                    visible=bool(visible),
                    x=cx,
                    y=cy,
                )
            )
        return detections

    def run_on_video(
        self, video_path: Path, start_frame: int = 0
    ) -> Iterable[BallDetection]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video at {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        buffer: List[np.ndarray] = []
        frame_index = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            buffer.append(frame)

            if len(buffer) == 3:
                start_idx = start_frame + frame_index - 2
                detections = self.predict_triplet(
                    buffer, frame_start_index=start_idx, frame_size=(width, height)
                )
                for det in detections:
                    yield det
                buffer.pop(0)  # slide window by one
            frame_index += 1

        cap.release()
        # TrackNetV2 requires groups of 3 frames; trailing frames <3 are skipped.
