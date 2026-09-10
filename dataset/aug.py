import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

# from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def get_default_aug(
    image_size: int,
    split="",
    enhanced: bool = True,
    no_aug: bool = False,
    profile: str = "kim_enhanced",
):
    if profile not in {"default", "kim", "kim_enhanced"}:
        raise ValueError(f"Unknown fundus augmentation profile: {profile}")

    normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    if profile == "kim":
        if split == "train" and not no_aug:
            return T.Compose(
                [
                    T.RandomResizedCrop(image_size, scale=(0.90, 1.0)),
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomRotation(degrees=10),
                    T.ToTensor(),
                    normalize,
                ]
            )
        return T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                normalize,
            ]
        )

    if profile == "kim_enhanced":
        if split == "train" and not no_aug:
            return T.Compose(
                [
                    T.RandomResizedCrop(image_size, scale=(0.90, 1.0)),
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomRotation(degrees=10),
                    T.RandomApply(
                        [
                            T.ColorJitter(
                                brightness=0.10,
                                contrast=0.10,
                                saturation=0.05,
                                hue=0.01,
                            )
                        ],
                        p=0.5,
                    ),
                    T.ToTensor(),
                    normalize,
                ]
            )
        return T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                normalize,
            ]
        )

    if split == "train" and not no_aug:
        if enhanced:
            return T.Compose(
                [
                    T.Resize((image_size, image_size)),
                    T.RandomApply(
                        [
                            T.ColorJitter(
                                brightness=0.25,
                                contrast=0.25,
                                saturation=0.15,
                                hue=0.02,
                            )
                        ],
                        p=0.8,
                    ),
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomRotation(degrees=12, fill=0),  # small, realistic
                    T.RandomAffine(
                        degrees=0,
                        translate=(0.02, 0.02),  # tiny shifts
                        scale=(0.95, 1.05),
                    ),  # gentle zoom
                    T.RandomApply(
                        [T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2
                    ),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            return T.Compose(
                [
                    T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                    # T.Resize((image_size, image_size)),
                    T.RandomApply(
                        [T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
                    ),
                    T.RandomGrayscale(p=0.2),
                    T.RandomApply([T.GaussianBlur((3, 3), [0.1, 2.0])], p=0.5),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
    else:
        return T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )


def select_oct_slice_indices(
    total_slices: int,
    requested_slices: int,
    *,
    split: str,
    profile: str = "oct_clinical_v1",
    no_aug: bool = False,
) -> list[int]:
    """Select OCT B-scans with a small train-only coherent index shift."""
    if profile not in {"none", "oct_clinical_v1"}:
        raise ValueError(f"Unknown OCT augmentation profile: {profile}")
    if total_slices <= 0:
        return []
    if requested_slices <= 0 or requested_slices >= total_slices:
        return list(range(total_slices))

    indices = np.linspace(0, total_slices - 1, requested_slices).round().astype(int)
    if split == "train" and not no_aug and profile == "oct_clinical_v1":
        shift = int(torch.randint(-2, 3, (1,)).item())
        indices = np.clip(indices + shift, 0, total_slices - 1)
    return indices.tolist()


class OCTVolumeTransform:
    """Apply one coherent augmentation to an entire ``[S,1,H,W]`` volume."""

    def __init__(
        self,
        *,
        split: str,
        profile: str = "oct_clinical_v1",
        no_aug: bool = False,
        max_translation_fraction: float = 0.03,
        max_rotation_degrees: float = 2.0,
        intensity_jitter: float = 0.10,
        gamma_jitter: float = 0.10,
        noise_std: float = 0.01,
        slice_dropout_p: float = 0.05,
    ):
        if profile not in {"none", "oct_clinical_v1"}:
            raise ValueError(f"Unknown OCT augmentation profile: {profile}")
        self.active = split == "train" and not no_aug and profile != "none"
        self.profile = profile
        self.max_translation_fraction = float(max_translation_fraction)
        self.max_rotation_degrees = float(max_rotation_degrees)
        self.intensity_jitter = float(intensity_jitter)
        self.gamma_jitter = float(gamma_jitter)
        self.noise_std = float(noise_std)
        self.slice_dropout_p = float(slice_dropout_p)

    @staticmethod
    def _uniform(low: float, high: float) -> float:
        return float(torch.empty(1).uniform_(low, high).item())

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32)
        if tensor.ndim != 4 or tensor.shape[1] != 1:
            raise ValueError(
                "OCT volume must have shape [slices,1,height,width], got "
                f"{tuple(tensor.shape)}"
            )
        if not self.active:
            return tensor

        tensor = tensor.clone()
        height, width = tensor.shape[-2:]
        angle = self._uniform(
            -self.max_rotation_degrees,
            self.max_rotation_degrees,
        )
        translate_x = int(
            round(
                self._uniform(
                    -self.max_translation_fraction,
                    self.max_translation_fraction,
                )
                * width
            )
        )
        tensor = TF.affine(
            tensor,
            angle=angle,
            translate=[translate_x, 0],
            scale=1.0,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        )

        intensity_scale = self._uniform(
            1.0 - self.intensity_jitter,
            1.0 + self.intensity_jitter,
        )
        gamma = self._uniform(
            1.0 - self.gamma_jitter,
            1.0 + self.gamma_jitter,
        )
        tensor = tensor.clamp(0.0, 1.0).pow(gamma) * intensity_scale
        if self.noise_std > 0:
            tensor = tensor + torch.randn_like(tensor) * self.noise_std

        if self.slice_dropout_p > 0 and tensor.shape[0] > 1:
            dropped = torch.rand(tensor.shape[0]) < self.slice_dropout_p
            if bool(dropped.all()):
                dropped[torch.randint(0, tensor.shape[0], (1,))] = False
            tensor[dropped] = 0.0
        return tensor.clamp_(0.0, 1.0)


def get_oct_volume_aug(
    *,
    split: str,
    profile: str = "oct_clinical_v1",
    no_aug: bool = False,
) -> OCTVolumeTransform:
    return OCTVolumeTransform(split=split, profile=profile, no_aug=no_aug)
