"""Dataset sample discovery for CAMUS segmentation."""

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


VIEWS = ("2CH", "4CH")
PHASES = ("ED", "ES")
DEFAULT_IMAGE_SIZE = (512, 416)


@dataclass(frozen=True)
class CamusSample:
    """Paths and identifiers for one image-mask pair."""

    patient_id: str
    view: str
    phase: str
    image_path: Path
    mask_path: Path


def read_patient_ids(split_file: Path) -> list[str]:
    """Read one patient ID per line from a split file."""

    return [
        line.strip()
        for line in split_file.read_text().splitlines()
        if line.strip()
    ]


def build_samples(camus_root: Path, split_file: Path) -> list[CamusSample]:
    """Build the four annotated ED/ES samples for every patient in a split."""

    samples = []

    for patient_id in read_patient_ids(split_file):
        patient_dir = camus_root / patient_id

        for view in VIEWS:
            for phase in PHASES:
                name = f"{patient_id}_{view}_{phase}"
                samples.append(
                    CamusSample(
                        patient_id=patient_id,
                        view=view,
                        phase=phase,
                        image_path=patient_dir / f"{name}.nii.gz",
                        mask_path=patient_dir / f"{name}_gt.nii.gz"
                    )
                )

    return samples


class CamusDataset(Dataset):
    """Load one CAMUS image-mask pair when an index is requested."""

    def __init__(
        self,
        samples: list[CamusSample],
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    ):
        self.samples = samples
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        #index here is from 0 to 1599 since there are 1600 samples in total (400 patients * 4 samples per patient)
        sample = self.samples[index]
        #load the image and mask using nibabel, convert to float32, and normalize the image to [0, 1].
        image = nib.load(sample.image_path).get_fdata(dtype=np.float32)
        mask = nib.load(sample.mask_path).get_fdata(dtype=np.float32)
        #convert the image and mask to PyTorch tensors, add a channel dimension, and normalize the image to [0, 1].
        image_tensor = torch.from_numpy(image).unsqueeze(0) / 255.0
        mask_tensor = torch.from_numpy(mask)

        image_tensor = F.interpolate(
            image_tensor.unsqueeze(0),
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        mask_tensor = F.interpolate(
            mask_tensor.unsqueeze(0).unsqueeze(0),
            size=self.image_size,
            mode="nearest",
        ).squeeze(0).squeeze(0).long()

        return image_tensor, mask_tensor
