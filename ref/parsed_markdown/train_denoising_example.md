# train_denoising_example.ipynb


# Denoising network using DnCNN

DnCNN을 이용하여 noisy semiconductor image의 **denoising network**를 학습한다.

학습이 끝난 뒤, **best checkpoint 로 test 데이터셋에 대해 검증**한다.

mean filter, median filter 등 **conventional method와 결과를 비교**한다.


```python
# [Cell 1]
# Run this codeblock to mount your Google Drive in Google Colab.
from google.colab import drive

drive.mount("/content/drive/")
```


```python
# [Cell 2]
# loguru 는 Colab 기본 이미지에 없으므로 필요할 때만 설치
try:
    import loguru  # noqa: F401
except ImportError:
    !pip install -q loguru
```


## 1. Config


```python
# [Cell 4]
import glob
import json
import os
import random
import zlib
import time
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from scipy.io import loadmat, savemat
from torch import Tensor, nn
from torch.nn import functional
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# TODO: set your root directory here
ROOT = Path("/content/drive/MyDrive/실습프로젝트")

DATA_ROOT = ROOT / "dataset"
RUN_DIR = ROOT / "logs_denoising"
CHECKPOINT_DIR = ROOT / "code_denoising_new/checkpoint_best.ckpt"

# 학습/검증에 얹을 노이즈를 4종 중 랜덤으로 고를 때 쓰는 sigma 범위
NOISE_RANGES: dict[str, tuple[float, float]] = {
    "gaussian": (0.0, 0.1),
    "rician": (0.0, 0.15),
    "uniform": (0.0, 0.2),
    "salt_and_pepper": (0.0, 0.2),
}

if not ROOT.exists():
    print(f"Root directory {ROOT} does not exist. Please check the path.")


@dataclass
class GeneralConfig:
    # Dataset
    train_dataset: list[str] = field(default_factory=lambda: [str(DATA_ROOT / "train")])
    valid_dataset: list[str] = field(default_factory=lambda: [str(DATA_ROOT / "val")])
    test_dataset: list[str] = field(default_factory=lambda: [str(DATA_ROOT / "test_label")])
    data_type: str = "*.npy"

    test_noisy_dir: str | None = None

    # Logging
    log_lv: str = "INFO"
    run_dir: Path = RUN_DIR
    init_time: float = 0.0

    # Model experiment
    model_type: Literal["dncnn"] = "dncnn"

    # Optimizer
    optimizer: Literal["adam", "adamw"] = "adam"
    loss_model: Literal["l1", "l2"] = "l2"
    lr: float = 1e-4
    lr_decay: float = 0.88
    lr_tol: int = 1

    # Train params
    gpu: str = "0"
    train_batch: int = 16
    valid_batch: int = 1
    train_epoch: int = 10
    logging_density: int = 4
    valid_interval: int = 1
    valid_tol: int = 0
    num_workers: int = 2
    save_val: bool = True
    parallel: bool = False  # Colab 은 보통 GPU 1장이므로 DataParallel 비활성
    device: torch.device | None = None
    save_max_idx: int = 500

    # Experiment
    noise_type: Literal["random"] = "random"
    noise_sigma: float = 0.05  # noise_type 이 "random" 이 아닐 때만 사용

    tag: str = ""


@dataclass
class DnCNNConfig:
    # Model architecture
    channels: int = 1
    num_of_layers: int = 17
    kernel_size: int = 3
    padding: int = 1
    features: int = 64


config = GeneralConfig()
dncnnconfig = DnCNNConfig()

config.test_noisy_dir = str(DATA_ROOT / "test_noise_only")

for k, v in asdict(config).items():
    print(f"{k}: {v}")
```


## 2. utils




```python
# [Cell 6]
def timestamp() -> str:
    return datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")


def separator(cols: int = 100) -> str:
    return "#" * cols


def seconds_to_dhms(seconds: float) -> str:
    s = seconds % 60
    m = (seconds // 60) % 60
    h = seconds // (60 * 60) % 1000
    return f"{int(h):02}h {int(m):02}m {int(s):02}s"


def call_next_id(run_dir: Path) -> int:
    run_ids = []
    os.makedirs(run_dir, exist_ok=True)
    for entry in os.listdir(run_dir):
        if (run_dir / entry).is_dir():
            try:
                run_ids.append(int(entry.split("_")[0]))
            except ValueError:
                continue
    return max(run_ids, default=-1) + 1


def validate_tensors(tensors: list[Tensor]) -> None:
    for i, t in enumerate(tensors):
        if not isinstance(t, Tensor):
            raise TypeError(f"Tensor at index {i} is not a torch.Tensor, got {type(t)} instead.")


def validate_tensor_dimensions(tensors: list[Tensor], expected_dim: int) -> None:
    for i, t in enumerate(tensors):
        if t.dim() != expected_dim:
            raise ValueError(f"Tensor at index {i} has {t.dim()} dimensions, expected {expected_dim} dimensions.")


def validate_tensor_channels(tensor: Tensor, expected_channels: int) -> None:
    if tensor.shape[1] != expected_channels:
        raise ValueError(f"Expected tensor with {expected_channels} channels, but got {tensor.shape[1]} channels.")
```


## 3. logger


```python
# [Cell 8]
import inspect
import sys
import traceback
from functools import wraps

from loguru import logger as logurulogger

LOGURU_LEVEL = ["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]
FORMAT = "{time:YYYY-MM-DD HH:mm:ss} <level>[{level}]</level> {message}"
LOGURU_LV_LOWER = [x.lower() for x in LOGURU_LEVEL]


class SingletonMeta(type):
    _instances: dict[Any, Any] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class CustomLogger(metaclass=SingletonMeta):
    def __getattr__(self, name: str) -> Any:
        if name not in LOGURU_LV_LOWER:
            return getattr(logurulogger, name)

        def custom_handler(message: str | Any) -> None:
            if name == "error":
                caller_name = inspect.stack()[1].function
                getattr(logurulogger, name)(f"Error in func {caller_name}")

            if isinstance(message, str):
                processed_message = message.split("\n")
                for _item in processed_message:
                    getattr(logurulogger, name)(_item)
            else:
                getattr(logurulogger, name)(message)

        return custom_handler

    def trace(self, message: Any) -> None:
        self.__getattr__("trace")(message)

    def debug(self, message: Any) -> None:
        self.__getattr__("debug")(message)

    def info(self, message: Any) -> None:
        self.__getattr__("info")(message)

    def success(self, message: Any) -> None:
        self.__getattr__("success")(message)

    def warning(self, message: Any) -> None:
        self.__getattr__("warning")(message)

    def error(self, message: Any) -> None:
        self.__getattr__("error")(message)

    def critical(self, message: Any) -> None:
        self.__getattr__("critical")(message)


def logger_add_handler(
    _logger: CustomLogger,
    file: str | Path | None = None,
    level: str | None = None,
) -> None:
    _logger.remove()
    if level is None:
        level = "TRACE"
    _logger.add(
        sys.stdout,
        colorize=True,
        format=FORMAT,
        level=level,
    )
    if file is None:
        return
    _logger.add(
        file,
        colorize=True,
        format=FORMAT,
        level=level,
    )


logger = CustomLogger()
logger_add_handler(logger)


# common/wrapper.py 의 error_wrap
def error_wrap(func: Callable) -> Callable:
    @wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as err:
            logger.error(f"Error in {func.__name__}: {err}\n{traceback.format_exc()}")
            return None

    return sync_wrapper
```


## 4. MetricController


```python
# [Cell 10]
class MetricController:
    def __init__(self) -> None:
        self.state_dict: dict[str, list[float]] = {}

    def reset(self) -> None:
        self.state_dict = {}

    def add(
        self,
        key: str,
        value: torch.Tensor,
    ) -> None:
        if not isinstance(key, str):
            raise TypeError(f"{key} is not a string")

        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{value} is not torch.Tensor")

        if key not in self.state_dict:
            self.state_dict[key] = []

        if value.dim() == 4 and value.shape[1:] == (1, 1, 1):
            value = value.view(-1)
        elif value.dim() == 1:
            pass
        else:
            raise ValueError(f"Expected value to have shape (b, 1, 1, 1) or (b,), but got {value.shape}")

        value = value.cpu().detach().numpy()
        self.state_dict[key].extend(value.flatten())

    def mean(self, key: str) -> float:
        if key not in self.state_dict or len(self.state_dict[key]) == 0:
            raise ValueError(f"No values found for key: {key}")
        return np.mean(self.state_dict[key])

    def std(self, key: str) -> float:
        if key not in self.state_dict or len(self.state_dict[key]) == 0:
            raise ValueError(f"No values found for key: {key}")
        return np.std(self.state_dict[key], ddof=1)
```


## 5. PSNR / SSIM


```python
# [Cell 12]
IMG_DIM: int = 4


class SSIMcal(torch.nn.Module):
    def __init__(
        self,
        win_size: int = 11,
        k1: float = 0.01,
        k2: float = 0.03,
    ):
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size**2)
        np_ = win_size**2
        self.cov_norm = np_ / (np_ - 1)

    def forward(
        self,
        img: torch.Tensor,
        ref: torch.Tensor,
        data_range: torch.Tensor,
    ) -> torch.Tensor:
        data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2

        ux = functional.conv2d(img, self.w.to(img.device))
        uy = functional.conv2d(ref, self.w.to(img.device))
        uxx = functional.conv2d(img * img, self.w.to(img.device))
        uyy = functional.conv2d(ref * ref, self.w.to(img.device))
        uxy = functional.conv2d(img * ref, self.w.to(img.device))

        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)

        A1 = 2 * ux * uy + C1
        A2 = 2 * vxy + C2
        B1 = ux**2 + uy**2 + C1
        B2 = vx + vy + C2

        S = (A1 * A2) / (B1 * B2)
        return torch.mean(S, dim=[2, 3], keepdim=True)


ssim_cal = SSIMcal()


def calculate_ssim(
    img: torch.Tensor,
    ref: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if not (img.dim() == IMG_DIM and ref.dim() == IMG_DIM):
        raise ValueError("All tensors must be 4D.")

    if mask is not None and (mask.dim() != IMG_DIM):
        raise ValueError("Mask must be 4D.")

    if img.shape[1] == 2:
        img = torch.sqrt(img[:, :1, ...] ** 2 + img[:, 1:, ...] ** 2)
        ref = torch.sqrt(ref[:, :1, ...] ** 2 + ref[:, 1:, ...] ** 2)

    if mask is None:
        img_mask = img
        ref_mask = ref
    else:
        if mask.shape[1] == 2:
            mask = torch.sqrt(mask[:, :1, ...] ** 2 + mask[:, 1:, ...] ** 2)
        img_mask = img * mask
        ref_mask = ref * mask

    ones = torch.ones(ref.shape[0], device=ref.device)
    ssim = ssim_cal.forward(img_mask, ref_mask, ones)
    return ssim


def calculate_psnr(
    img: torch.Tensor,
    ref: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if not (img.dim() == IMG_DIM and ref.dim() == IMG_DIM):
        raise ValueError("All tensors must be 4D.")

    if mask is not None and mask.dim() != IMG_DIM:
        raise ValueError("Mask must be 4D.")

    if img.shape[1] == 2:
        img = torch.sqrt(img[:, :1, ...] ** 2 + img[:, 1:, ...] ** 2)
        ref = torch.sqrt(ref[:, :1, ...] ** 2 + ref[:, 1:, ...] ** 2)

    if mask is not None:
        if mask.shape[1] == 2:
            mask = torch.sqrt(mask[:, :1, ...] ** 2 + mask[:, 1:, ...] ** 2)

        img_mask = img * mask
        ref_mask = ref * mask

        mse = torch.sum((img_mask - ref_mask) ** 2, dim=(1, 2, 3)) / torch.sum(mask, dim=(1, 2, 3))
    else:
        mse = torch.mean(functional.mse_loss(img, ref, reduction="none"), dim=(1, 2, 3), keepdim=True)

    img_max = torch.amax(ref, dim=(1, 2, 3), keepdim=True)
    psnr = 10 * torch.log10(img_max**2 / (mse + 1e-12))
    return psnr
```


## 6. DnCNN

![image.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAy0AAAJ+CAIAAAAxH3i8AAAQAElEQVR4AeydCXyU1dX/pQiCICSCgIQlYVFBJQRZC2ii1eIK2NYVK2jrQquCb1+tWymtK7WyaOvy1gLqX7FaAbWKohIFBARNoIoLYIgQBAokKAgilP83OXJ9nJnMM0kmk1l+fk5vz3Puudt3tt/cO3n4wX79JwIiIAIiIAIiIAIiUBcEfnCQ/hMBERABERCB2BHQSCIgAt8RkA77joU8ERABERABERABEYglAemwWNLWWKlKQOsWAREQAREQgVAEpMNCUVFMBERABERABERABGqfQG3psNqfuUYQAREQAREQAREQgcQmIB2W2I+fZi8CIiACImAEVIpAIhKQDkvER01zFgEREAEREAERSAYC0mHJ8ChqDalKQOsWAREQARFIbALSYYn9+Gn2IiACIiACIiACiUsg0XRY4pLWzEVABERABERABETg+wTiSIfVyxolE4GQBL7/pNWVCIiACMSQQCVDTXpmjUwEQhKo5CkTOhxHOiz0BBUVAREQAREQAREQgSQlIB2WpA+sliUCNSCgpiIgAiIgArEhIB0WG84aRQREQAREQAREQAQCCUiHGRGVIiACIiACIiACIhBrAtJhsSau8URABERABETgoIPEQATKCUiHlVPQ/0RABERABERABEQg9gSkw2LPXCOKQKoS0LpFQAREQAS+T0A67Ps8dCUCIiACIiACIiACsSIgHVa7pNW7CIiACIiACIiACFRGQDqsMjKKi4AIiIAIiEDiEdCME4uAdFhiPV6arQiIgAiIgAiIQPIQkA5LnsdSKxGBVCWgdYuACIhAohKQDkvUR07zFgEREAEREAERSHQC0mGJ+Qhq1iIgAiIgAiIgAolPQDos8R9DrUAEREAEREAEapuA+q8dAtJhtcNVvYqACIiACIiACIiAHwHpMD9CqhcBEUhVAlq3CIiACNQ2Aemw2ias/kVABERABERABEQgNAHpsNBcUjWqdYuACIiACIiACMSOgHRY7FhrJBEQAREQAREQge8TSPUr6bBUfwZo/SIgAiIgAiIgAnVFQDqsrshrXBEQgVQloHWLgAiIwAEC0mEHSOj/RUAEREAEREAERCC2BKTDYss7VUfTukVABERABERABIIJSIcFM1FEBERABERABEQgsQkkyuylwxLlkdI8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YBsJskQnIBGW6I+g5i8CiUigoKDgiy++2LdvH5P/5ptvkFDvvRfi37DfvHnzihUrXCb5+ETs/qi09dr777+/evXqHTvKtw8sznZaaWnpRx99ZJfhy+3bt5PftGn5RkPlmVWo4fy0SZMmzCfhzyWRYlVYt1LDEmCL6/Otu49s0YhtMEv855sbJj2zZtrLn9llcPm3F4tJeH7htzeJ4VAyrWkDOgnOVCQRCWhLIBEfNc1ZBGJJgK2m/Pz8JUuWeAflkiBV3iA+QQzHa6QRpIkLIk0QXvPnzye+cOFCJBRVlFySjO8MeeQyyccnYltcCDiXZg5tly1bRidmixYtWr58OflW61siB+mZw82QmTY9ypC1wcH27duzP7d169bgqviJ+O+Hxc9ck2YmC1aUPyfYBqveivJ6HbF33/7K/smj6vWpViIgAvFLQDMTgfgjkJ6ezqTsF1040bJVq1ah7doE/RFANfpnM6xVq1a7IvtnKKvRf7Sa+OswJO348eOjNZ76gUDJlt2fbtjZvlVj+5kXkchtwLGHt2zecPX6HeyrRd5KmSIgAiIgAiIQLQLoG7vzPqeW0erT9VNcXHzIIYd069bNRarnHH300WytuX9goHqdxKCVvw5jEkgxSlkUCXDIuGPX3hOzW7jTyUg650TyhKPTPt+6e847of/970g6SYgcTVIEREAERCAeCPTq1QtJhPCyyTRv3rxr165IHC7ZaqoNHcbJZmlpacuWLd2gjFVV42STHbuSkpLamGFVJxM+PyIdFr4L1VaPwN9eLMaqtK3FRtoDz3369Bsl1RtRrURABERABESgSgTYUmrdunX37t1zK/7LycnJyMioX7/+119/vXz58ip1FXkyPc+fP78mEoptMLaQOOWMfNC6ypQOqyvyGlcEREAEREAE4p3ARx99xO4UqstNlG0w9pkWLVq0e7funeSoVN/x12EoYN23ovqA1VIEREAE4pSApiUC/gS2b9/O7hSqi+0lsyVLliTEPpP/2uIjw1+HMU+kGKVMBERABERABERABEQgigQi0mFRHE9diUBdEtDYIiACIiACIhBPBPx1GPuQum9FPD1kmosIiIAIiIAIiECCEPCbpr8OowekGKVMBERABERABERABEQgigQi0mFRHE9diYAIiIAIJDcBrU4ERCByAtJhkbNSpgiIgAiIgAiIgAhEk4C/DsvN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hFEEMeZhGxa3liAJ5m/evNkyaauqqpxV6GUUApFkDMcZQhdpbIn9WxeZ9LIvutoyckimtQTyeTTfWoZj5lvLI2nmQ4kdMoSWPROn17poEWock14MB3q9evXiKpYuMyLE6cWYgc1zBOuiXohoQ6raYzxbrSUCSUxAOiyJX66OJgKJRGDnzp3p6ekBJbFEOkAC7pWryZSUFIqRHe6d9zJw4EDeUYeZShABEYiIgHRYRLiULALxIeDHVSi0HD58mLs/Px4+LmdGdRUWFiKnbDXqW1yYcs9Ikcwi7bTct3LLyTtqJ0ddIiACnSAgHdYJaBoiAiIQEwLcgnEpFpOpNelZZ3HXefz4ceTXpNafzMzMXbt2cccaDhteDRZOpnJEQAQiIuAZHRbRrpUsAiIgAiIQOYHq6ury8vLS1p+Kiora2trI59AIERCBaBKQDosmTc0lAiIgAiKQMAS0URHwAAHpMA+8BG1BBERABERABETAlwSkw3z52nVovxLQuUVABERABDxFQDrMU69DmxEBERABERABEfARgaTXYT56lzqqCIiACIiACIhAYhGQDkus96XdioAIiIAIeJyAticCERDwrw47te05mQiEJBDB/wIpNYhA61dTqRGBEASC/mNRQARE4Cz/6jC9fBEQgagR0EQiIAIiIAKdIiAd1ilsGiQCIiACIiACIiACXSYgHdZJhBomAiIgAiIgAiIgAl0kIB3WRYAaLgIiIAIiIALxIKA1kpKAdFhSvlYdSgREQAREQAREIAEISIclwEvSFkXArwR0bhEQARFIcgLSYUn+gnU8ERABERABERABzxKQDvPYq9F2REAEREAEREAEfENAOsw3r1oHFQEREAEREIFgAop0KwHpsG7Fr8VFQAREQAREQAR8TEA6zMcvX0cXAb8S0LlFQAREwCMEpMM88iK0DREQAREQAREQAd8RkA7zySvXMUVABERABERABDxHQDrMc69EGxIBERABERCBxCegE4RFQDosLExKEgEREAEREAEREIGoE5AOizpSTSgCIuBXAjq3CIiACERIQDosQmBKFwEREAEREAEREIEoEZAOixJIv06jc4uACIiACIiACHSagHRYp9FpoAiIgAiIgAiIQLwJJNl60mFJ9kJ1HBEQAREQAREQgYQhIB2WMK9KGxUBEfArAZ1bBEQgaQlIhyXtq9XBREAEREAEREAEPE5AOszjL8iv29O5RUAEREAERMAHBKTDfPCSdUQREAEREAEREIH2CXRTr3RYN4HXsiIgAiIgAiIgAr4nIB3m+/8EBEAERMCvBHRuERCBbicgHdbtr0AbEAEREAEREAER8CkB6TCfvni/HlvnFgEREAEREAEPEZAO89DL0FZEQAREQAREQASSi0AHp5EO6wCQukVABERABERABEQgRgSkw2IEVtOKgAiIgF8J6NwiIAJhE/CWDlv+f2tlIiACIiACIhALAmH/y6hEEYgfAW/psPidWyuJQFQJaDIREAEREAER6AQBr+iw7/z7OJkIiIAIiIAIxJpAJ/6l1BARiB2BTuuw2G1JM4uACIiACIiACIiALwhIh/niNeuQIiACIpD4BHQCEUhCAtJhSfhSdSQREAEREAEREIGEICAdlhCvSZv0KwGdWwREQAREIKkJSIcl9evV4URABERABERABDxMwHM6zMOstDUREAEREAEREAERiCYB6bBo0tRcIiACIiACCUdAGxaBbiQgHdaN8LW0CIiACIiACIiArwlIh/n69evwfiWgc4uACIiACHiCgHSYJ16DNiECIiACIiACIuBDAv8PAAD//wKkUSYAAAAGSURBVAMAZgluqx2ExHcAAAAASUVORK5CYII=)


```python
# [Cell 14]
class DnCNN(nn.Module):
    def __init__(
        self,
        channels: int,
        num_of_layers: int,
        kernel_size: int,
        padding: int,
        features: int,
    ):
        super().__init__()

        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.SiLU(inplace=True))

        for _ in range(num_of_layers - 1):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.GroupNorm(4, features))
            layers.append(nn.SiLU(inplace=True))

        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))

        self.dncnn = nn.Sequential(*layers)

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        if len(x.shape) != 4:
            raise ValueError(f"Input tensor must be 4D, but got {len(x.shape)}D tensor.")

        out = x + self.dncnn(x)
        return out


# sanity check
_m = DnCNN(channels=1, num_of_layers=3, kernel_size=3, padding=1, features=8)
print("DnCNN out:", _m(torch.randn(1, 1, 64, 64)).shape)
del _m
```


## 7. NoiseSimulator

**이미지마다 아래 4종 중 하나를 랜덤으로 골라 `NOISE_RANGES` 범위에서 sigma 를 뽑아** 적용한다.

- Gaussian noise

- Rician noise

- Uniform noise

- Salt and pepper noise


```python
# [Cell 16]
class NoisyType(str, Enum):
    Gaussian = "gaussian"
    Rician = "rician"
    Uniform = "uniform"
    SaltAndPepper = "salt_and_pepper"

    @classmethod
    def from_string(cls, value: str) -> "NoisyType":
        try:
            return cls(value)
        except ValueError as err:
            raise ValueError(f"Invalid NoisyType value: {value}. Must be one of {list(cls)} : {err}") from err


def gaussian_noise(img, sigma: float) -> torch.Tensor:
    noise = torch.randn_like(img) * sigma
    noisy_img = img + noise
    return noisy_img


def rician_noise(img, sigma: float) -> torch.Tensor:
    noise_real = torch.randn_like(img) * sigma
    noise_imag = torch.randn_like(img) * sigma
    noisy_img = torch.abs(img + noise_real + 1j * noise_imag)
    return noisy_img


def uniform_noise(img, sigma: float) -> torch.Tensor:
    noise = (torch.rand_like(img) * 2.0 - 1.0) * sigma
    return img + noise


def salt_and_pepper_noise(img, sigma: float) -> torch.Tensor:
    salt_prob = sigma / 2
    pepper_prob = sigma / 2
    noisy_img = img.clone()
    total_pixels = img.numel()

    # Salt noise
    num_salt = int(total_pixels * salt_prob)
    coords = [torch.randint(0, dim, (num_salt,)) for dim in img.shape]
    noisy_img[tuple(coords)] = img.max()
    # Pepper noise
    num_pepper = int(total_pixels * pepper_prob)
    coords = [torch.randint(0, dim, (num_pepper,)) for dim in img.shape]
    noisy_img[tuple(coords)] = 0  # Set to black

    return noisy_img


class RandomNoiseSimulator:
    def __init__(
        self,
        noise_ranges: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self.noise_ranges = dict(noise_ranges) if noise_ranges is not None else dict(NOISE_RANGES)
        self.names = list(self.noise_ranges.keys())

    def _sample(self, rng) -> tuple[str, float]:
        name = rng.choice(self.names)
        low, high = self.noise_ranges[name]
        return name, rng.uniform(low, high)

    def __call__(
        self,
        img: torch.Tensor,
        seed: int | None = None,
    ) -> torch.Tensor:
        if seed is None:
            # 학습용: 매번 새로 뽑는다 (같은 이미지도 epoch 마다 다른 노이즈)
            name, sigma = self._sample(random)
            return NoiseSimulator(NoisyType.from_string(name), sigma)(img)

        # 검증/테스트용: 이미지마다 항상 같은 노이즈가 나오도록 seed 를 고정한다
        name, sigma = self._sample(random.Random(seed))
        torch_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        try:
            noisy = NoiseSimulator(NoisyType.from_string(name), sigma)(img)
        finally:
            torch.random.set_rng_state(torch_state)
        return noisy

    def describe(self, seed: int) -> tuple[str, float]:
        return self._sample(random.Random(seed))

```


## 8. DataLoader


```python
# [Cell 18]
prob_flip: float = 0.5


class DataKey(IntEnum):
    Label = 0
    Noisy = 1
    Name = 2


@dataclass
class LoaderConfig:
    data_type: str
    batch: int
    num_workers: int
    shuffle: bool
    noisy_type: Literal["random"]
    noise_sigma: float
    noisy_path: str | None = None
    max_images: int | None = None


class DataWrapper(Dataset):
    file_list: list[str]
    training_mode: bool
    noise_simulator: NoiseSimulator

    def __init__(
        self,
        file_path: list[str],
        data_type: str,
        training_mode: bool,
        noisy_type: str,
        noise_sigma: float,
        noisy_path: str | None = None,
        max_images: int | None = None,
    ):
        self.training_mode = training_mode
        self.random_noise = str(noisy_type) == "random"
        if self.random_noise:
            self.noise_simulator = RandomNoiseSimulator()

        self.noisy_path = Path(noisy_path) if noisy_path is not None else None

        super().__init__()
        total_list: list[str] = []
        for _file_path in file_path:
            total_list += glob.glob(f"{_file_path}/{data_type}")

        total_list = sorted(total_list)
        if max_images is not None:
            total_list = total_list[:max_images]

        self.file_list = total_list

    @staticmethod
    def _load_from_npy(
        file_npy: str,
    ) -> torch.Tensor:
        img = torch.from_numpy(np.load(file_npy)).type(torch.float)
        if len(img.shape) == 2:
            img = img.unsqueeze(0)
        return img

    def _augment(
        self,
        label: torch.Tensor,
        noisy: torch.Tensor | None = None,
    ):
        # label 과 noisy 에 동일한 flip 을 적용한다
        if random.random() > prob_flip:
            label = torch.flip(label, dims=[1])
            noisy = torch.flip(noisy, dims=[1]) if noisy is not None else None
        if random.random() > prob_flip:
            label = torch.flip(label, dims=[2])
            noisy = torch.flip(noisy, dims=[2]) if noisy is not None else None

        return label, noisy

    def __getitem__(
        self,
        idx: int,
    ):
        _name = self.file_list[idx].split("/")[-1]
        label = self._load_from_npy(self.file_list[idx])

        if self.noisy_path is None:
            if self.training_mode:
                label, _ = self._augment(label)
                noisy = self.noise_simulator(label)
            elif self.random_noise:
                # 검증/테스트는 seed 를 고정
                noisy = self.noise_simulator(label, seed=zlib.crc32(_name.encode()))
            else:
                noisy = self.noise_simulator(label)
        else:
            noisy_file = self.noisy_path / _name
            if not noisy_file.exists():
                raise FileNotFoundError(f"Matching noisy file not found: {noisy_file}")
            noisy = self._load_from_npy(str(noisy_file))
            if self.training_mode:
                label, noisy = self._augment(label, noisy)

        return (
            label,
            noisy,
            _name,
        )

    def __len__(self) -> int:
        return len(self.file_list)


def get_data_wrapper_loader(
    file_path: list[str],
    training_mode: bool,
    loader_cfg: LoaderConfig,
) -> tuple[DataLoader, DataWrapper, int]:
    dataset = DataWrapper(
        file_path=file_path,
        data_type=loader_cfg.data_type,
        training_mode=training_mode,
        noisy_type=loader_cfg.noisy_type,
        noise_sigma=loader_cfg.noise_sigma,
        noisy_path=loader_cfg.noisy_path,
        max_images=loader_cfg.max_images,
    )

    if len(dataset) == 0:
        raise FileNotFoundError(f"No data found in {file_path} with pattern {loader_cfg.data_type}")

    _ = dataset[0]

    dataloader = DataLoader(
        dataset,
        batch_size=loader_cfg.batch,
        num_workers=loader_cfg.num_workers,
        pin_memory=True,
        persistent_workers=loader_cfg.num_workers > 0,
        shuffle=loader_cfg.shuffle,
    )

    return (
        dataloader,
        dataset,
        len(dataset),
    )
```


### 데이터 확인


```python
# [Cell 20]
import matplotlib.pyplot as plt

_loader, _, _len = get_data_wrapper_loader(
    file_path=config.train_dataset,
    training_mode=True,
    loader_cfg=LoaderConfig(
        data_type=config.data_type,
        batch=6,
        num_workers=0,
        shuffle=True,
        noisy_type=config.noise_type,
        noise_sigma=config.noise_sigma,
    ),
)
print(f"train dataset length: {_len}")
print(f"noise: {config.noise_type}")
if config.noise_type == "random":
    for k, v in NOISE_RANGES.items():
        print(f"  {k:16s} sigma {v[0]} ~ {v[1]}")

_label, _noisy, _name = next(iter(_loader))
n = _label.shape[0]
fig, axes = plt.subplots(2, n, figsize=(3 * n, 6.4))
for i in range(n):
    axes[0, i].imshow(_label[i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
    axes[0, i].set_title(f"label\n{_name[i]}", fontsize=8)
    axes[0, i].axis("off")
    axes[1, i].imshow(_noisy[i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
    axes[1, i].set_title("noisy", fontsize=8)
    axes[1, i].axis("off")
fig.suptitle("train sample")
fig.tight_layout()
plt.show()

```


## 9. train / test core functions


```python
# [Cell 22]
NETWORK = DnCNN | torch.nn.DataParallel
OPTIM = Adam | AdamW


class ModelType(str, Enum):
    DnCNN = "dncnn"

    @classmethod
    def from_string(cls, value: str) -> "ModelType":
        try:
            return cls(value)
        except ValueError as err:
            raise ValueError(f"Invalid ModelType value: {value}. Must be one of {list(cls)} : {err}") from err


def get_network(
    device: torch.device | None,
    model_type: str,
    dncnnconfig: DnCNNConfig,
) -> NETWORK:
    if device is None:
        raise TypeError("device is not to be None")

    if ModelType.from_string(model_type) == ModelType.DnCNN:
        return DnCNN(
            channels=dncnnconfig.channels,
            num_of_layers=dncnnconfig.num_of_layers,
            kernel_size=dncnnconfig.kernel_size,
            padding=dncnnconfig.padding,
            features=dncnnconfig.features,
        )
    else:
        raise KeyError("model type not matched")


def get_optim(
    network: NETWORK | None,
    optimizer: str,
) -> OPTIM | None:
    if network is None:
        return None
    if optimizer == "adam":
        return Adam(network.parameters(), betas=(0.9, 0.99))
    elif optimizer == "adamw":
        return AdamW(network.parameters(), betas=(0.9, 0.99), weight_decay=0.0)
    else:
        raise KeyError("optimizer not matched")


def get_loss_func(
    loss_model: str,
) -> Callable:
    if loss_model == "l1":
        return torch.nn.L1Loss(reduction="none")
    elif loss_model == "l2":
        return torch.nn.MSELoss(reduction="none")
    else:
        raise KeyError("loss func not matched")


def get_learning_rate(
    epoch: int,
    lr: float,
    lr_decay: float,
    lr_tol: int,
) -> float:
    factor = epoch - lr_tol if lr_tol < epoch else 0
    return lr * (lr_decay**factor)


def set_optimizer_lr(
    optimizer: OPTIM | None,
    learning_rate: float,
) -> OPTIM | None:
    if optimizer is None:
        return None
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate
    return optimizer


def log_summary(
    init_time: float,
    state: MetricController,
    log_std: bool = False,
) -> None:
    spend_time = seconds_to_dhms(time.time() - init_time)
    for key in state.state_dict:
        if log_std:
            summary = f"{spend_time} | {key}: {state.mean(key):0.3e} + {state.std(key):0.3e} "
            logger.info(summary)
        else:
            summary = f"{spend_time} | {key}: {state.mean(key):0.3e}"
            logger.info(summary)


def save_checkpoint(
    network: NETWORK,
    run_dir: Path,
    epoch: str | int | None = None,
) -> None:
    if epoch is None:
        epoch = "best"
    os.makedirs(run_dir / "checkpoints", exist_ok=True)
    torch.save(
        {
            "model_state_dict": network.state_dict(),
            "model_config": asdict(dncnnconfig),
        },
        run_dir / f"checkpoints/checkpoint_{epoch}.ckpt",
    )


def zero_optimizers(
    optim_list: list[OPTIM | None],
) -> None:
    for opt in optim_list:
        if opt is not None:
            opt.zero_grad()


def step_optimizers(
    optim_list: list[OPTIM | None],
) -> None:
    for opt in optim_list:
        if opt is not None:
            opt.step()


def save_result_to_mat(
    test_dir: Path,
    batch_cnt: int,
    tesner_dict: dict[str, Tensor | None],
    img_cnt: int,
) -> None:
    os.makedirs(test_dir, exist_ok=True)
    save_dict = {}

    if batch_cnt == 0:
        logger.warning("batch_cnt is 0, no data to save")
        return

    for i in range(batch_cnt):
        for key, value in tesner_dict.items():
            if value is not None:
                save_dict[key] = value.cpu().detach().numpy()[i, ...]

        idx = img_cnt + i + 1
        savemat(f"{test_dir}/{idx}_res.mat", save_dict)


def train_epoch_dncnn(
    _data,
    network: NETWORK,
    epoch: int,
    train_state: MetricController,
) -> int:
    loss_func = get_loss_func(config.loss_model)

    label: Tensor = _data[DataKey.Label].to(config.device)

    img_cnt_minibatch = label.shape[0]

    output = network.forward(_data[DataKey.Noisy].to(config.device))

    loss = torch.mean(loss_func(output, label), dim=(1, 2, 3), keepdim=True)

    torch.mean(loss).backward()
    train_state.add("loss", loss)

    return img_cnt_minibatch


def train_epoch(
    train_loader: DataLoader,
    train_len: int,
    network: NETWORK,
    optim_list: list[OPTIM | None],
    epoch: int,
) -> float:
    train_state = MetricController()
    train_state.reset()
    network.train()

    logging_cnt: int = 1
    img_cnt: int = 0
    batch_pbar = tqdm(
        train_loader,
        desc=f"  train ep {epoch + 1}",
        unit="batch",
        leave=False,
    )
    for _data in batch_pbar:
        zero_optimizers(optim_list=optim_list)
        if ModelType.from_string(config.model_type) == ModelType.DnCNN:
            img_cnt_minibatch = train_epoch_dncnn(
                _data=_data,
                network=network,
                epoch=epoch,
                train_state=train_state,
            )
        else:
            raise KeyError("model type not matched")

        step_optimizers(optim_list=optim_list)
        img_cnt += img_cnt_minibatch
        batch_pbar.set_postfix(loss=f"{train_state.mean('loss'):0.3e}")
        if img_cnt > (train_len / config.logging_density * logging_cnt):
            log_summary(init_time=config.init_time, state=train_state)
            logging_cnt += 1

    log_summary(init_time=config.init_time, state=train_state)
    return float(train_state.mean("loss"))


def test_part_dncnn(
    _data,
    test_dir: Path,
    model: NETWORK,
    save_val: bool,
    test_state: MetricController,
    img_cnt: int,
) -> int:
    noisy = _data[DataKey.Noisy].to(config.device)
    label = _data[DataKey.Label].to(config.device)

    batch_cnt = noisy.shape[0]

    validate_tensors([noisy, label])
    validate_tensor_dimensions([noisy, label], 4)

    output = model(noisy)

    validate_tensors([output])
    validate_tensor_dimensions([output], 4)

    for idx in range(output.shape[0]):
        test_state.add("psnr", calculate_psnr(output[idx : idx + 1, ...], label[idx : idx + 1, ...]))
        test_state.add("ssim", calculate_ssim(output[idx : idx + 1, ...], label[idx : idx + 1, ...]))

    if save_val:
        save_result_to_mat(
            test_dir=test_dir,
            batch_cnt=batch_cnt,
            tesner_dict={
                "noisy": noisy,
                "output": output,
                "label": label,
            },
            img_cnt=img_cnt,
        )

    return batch_cnt


def test_part(
    epoch: int,
    data_loader: DataLoader,
    network: NETWORK,
    run_dir: Path,
    save_val: bool,
    desc: str = "valid",
) -> float:
    test_state = MetricController()
    test_state.reset()
    network.eval()
    model = network.module if isinstance(network, torch.nn.DataParallel) else network

    img_cnt: int = 0
    batch_pbar = tqdm(data_loader, desc=f"  {desc}", unit="batch", leave=False)
    for _data in batch_pbar:
        if ModelType.from_string(config.model_type) == ModelType.DnCNN:
            batch_cnt = test_part_dncnn(
                _data=_data,
                test_dir=run_dir / f"test/ep_{epoch}",
                model=model,
                save_val=save_val and img_cnt <= config.save_max_idx,
                test_state=test_state,
                img_cnt=img_cnt,
            )
        else:
            raise KeyError("model type not matched")

        img_cnt += batch_cnt
        batch_pbar.set_postfix(psnr=f"{test_state.mean('psnr'):0.2f}")

    log_summary(init_time=config.init_time, state=test_state, log_std=True)

    primary_metric = test_state.mean("psnr")
    return primary_metric
```


## 10. Validation 결과 시각화


저장 위치: `{run_dir}/valid_png/ep_{epoch}/{파일명}.png`


```python
# [Cell 24]
import matplotlib.pyplot as plt

VALID_FIG_MAX: int = 4  # validation 마다 저장할 이미지 수


def save_comparison_figures(
    epoch: int,
    data_loader: DataLoader,
    network: NETWORK,
    run_dir: Path,
    max_images: int = VALID_FIG_MAX,
    tag: str = "valid",
    subdir: str | None = None,
    noise_lookup: dict[str, str] | None = None,
    title_prefix: str | None = None,
) -> Path:
    save_dir = run_dir / (subdir if subdir is not None else f"{tag}_png/ep_{epoch:03d}")
    os.makedirs(save_dir, exist_ok=True)

    network.eval()
    model = network.module if isinstance(network, torch.nn.DataParallel) else network
    dataset = data_loader.dataset

    saved = 0
    with torch.no_grad():
        for _data in data_loader:
            label = _data[DataKey.Label].to(config.device)
            noisy = _data[DataKey.Noisy].to(config.device)
            names = _data[DataKey.Name]

            output = model(noisy)

            for i in range(output.shape[0]):
                if saved >= max_images:
                    break

                lab_i, noi_i, out_i = label[i : i + 1], noisy[i : i + 1], output[i : i + 1]
                psnr_in = calculate_psnr(noi_i, lab_i).item()
                ssim_in = calculate_ssim(noi_i, lab_i).item()
                psnr_out = calculate_psnr(out_i, lab_i).item()
                ssim_out = calculate_ssim(out_i, lab_i).item()

                lab = lab_i.cpu().numpy().squeeze()
                noi = noi_i.cpu().numpy().squeeze()
                out = out_i.cpu().numpy().squeeze()
                err = np.abs(out - lab)

                vmax = float(np.percentile(lab, 98) * 1.2)
                err_vmax = float(max(err.max(), 1e-6))

                fig, axes = plt.subplots(1, 4, figsize=(16, 4.6))
                panels = [
                    (lab, "Original (label)", "gray", 0.0, vmax),
                    (noi, f"Noisy\nPSNR {psnr_in:.2f} dB / SSIM {ssim_in:.4f}", "gray", 0.0, vmax),
                    (out, f"Denoised\nPSNR {psnr_out:.2f} dB / SSIM {ssim_out:.4f}", "gray", 0.0, vmax),
                    (err, f"Error |denoised - label|\nmax {err_vmax:.3f}", "magma", 0.0, err_vmax),
                ]
                for ax, (img, title, cmap, lo, hi) in zip(axes, panels):
                    im = ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi)
                    ax.set_title(title, fontsize=10)
                    ax.axis("off")
                fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.02)

                noise_desc = ""
                if getattr(dataset, "random_noise", False) and dataset.noisy_path is None:
                    _nz, _sg = dataset.noise_simulator.describe(zlib.crc32(names[i].encode()))
                    noise_desc = f" | noise={_nz} sigma={_sg:.4f}"
                elif noise_lookup:
                    noise_desc = f" | noise={noise_lookup.get(names[i], 'unknown')}"

                fig.suptitle(
                    f"{title_prefix if title_prefix is not None else f'epoch {epoch}'}"
                    f" | {names[i]}{noise_desc}"
                    f" | PSNR {psnr_in:.2f} -> {psnr_out:.2f} dB",
                    fontsize=11,
                )
                fig.tight_layout(rect=(0, 0, 1, 0.90))
                fig.savefig(save_dir / f"{names[i].replace('.npy', '')}.png", dpi=150, bbox_inches="tight")
                plt.close(fig)
                saved += 1

            if saved >= max_images:
                break

    logger.info(f"Saved {saved} comparison figures to {save_dir}")
    return save_dir
```


## 11. Train


```python
# [Cell 26]
class Trainer:
    run_dir: Path
    network: NETWORK
    train_loader: DataLoader
    train_len: int
    valid_loader: DataLoader
    optims: list[OPTIM | None]

    def __init__(
        self,
    ) -> None:
        config.init_time = time.time()
        config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(config.device)

        # dir setting
        self.run_dir = config.run_dir / f"{call_next_id(config.run_dir):05d}_train"
        logger_add_handler(logger, f"{self.run_dir/'log.log'}", config.log_lv)
        logger.info(separator())
        logger.info(f"Run dir: {self.run_dir}")
        os.makedirs(self.run_dir, exist_ok=True)

        # log config
        logger.info(separator())
        logger.info("General Config")
        config_dict = asdict(config)
        for k in config_dict:
            logger.info(f"{k}:{config_dict[k]}")
        logger.info(separator())
        logger.info("Model Config")
        config_dict = asdict(dncnnconfig)
        for k in config_dict:
            logger.info(f"{k}:{config_dict[k]}")

    def __call__(
        self,
    ) -> None:
        self._set_data()
        self._set_network()
        self._train()

    @error_wrap
    def _set_data(
        self,
    ) -> None:
        logger.info(separator())
        train_loader_cfg = LoaderConfig(
            data_type=config.data_type,
            batch=config.train_batch,
            num_workers=config.num_workers,
            shuffle=True,
            noisy_type=config.noise_type,
            noise_sigma=config.noise_sigma,
        )

        valid_loader_cfg = LoaderConfig(
            data_type=config.data_type,
            batch=config.valid_batch,
            num_workers=config.num_workers,
            shuffle=False,
            noisy_type=config.noise_type,
            noise_sigma=config.noise_sigma,
        )

        self.train_loader, _, self.train_len = get_data_wrapper_loader(
            file_path=config.train_dataset,
            training_mode=True,
            loader_cfg=train_loader_cfg,
        )
        logger.info(f"Train dataset length : {self.train_len}")

        self.valid_loader, _, valid_len = get_data_wrapper_loader(
            file_path=config.valid_dataset,
            training_mode=False,
            loader_cfg=valid_loader_cfg,
        )
        logger.info(f"Valid dataset length : {valid_len}")

    @error_wrap
    def _set_network(
        self,
    ) -> None:
        self.network = get_network(
            device=config.device,
            model_type=config.model_type,
            dncnnconfig=dncnnconfig,
        )

        self.optims = [
            get_optim(
                network=self.network,
                optimizer=config.optimizer,
            ),
        ]

        if config.parallel:
            self.network = torch.nn.DataParallel(self.network).to(config.device)
        else:
            self.network = self.network.to(config.device)

    @error_wrap
    def _train(
        self,
    ) -> None:
        logger.info(separator())
        logger.info("Train start")

        best_metric: float = 0
        primary_metric: float = 0
        postfix: dict[str, str] = {}
        epoch_pbar = tqdm(
            range(config.train_epoch),
            desc="Epoch",
            unit="epoch",
            leave=True,
        )

        for epoch in epoch_pbar:
            lr_epoch = get_learning_rate(
                epoch=epoch,
                lr=config.lr,
                lr_decay=config.lr_decay,
                lr_tol=config.lr_tol,
            )
            postfix["lr"] = f"{lr_epoch:0.3e}"
            epoch_pbar.set_postfix(postfix)
            tqdm.write(f"Epoch {epoch + 1}/{config.train_epoch} | lr={lr_epoch:0.3e}")

            optims = [set_optimizer_lr(optimizer=optim, learning_rate=lr_epoch) for optim in self.optims]

            train_loss = train_epoch(
                train_loader=self.train_loader,
                train_len=self.train_len,
                network=self.network,
                optim_list=optims,
                epoch=epoch,
            )
            postfix["loss"] = f"{train_loss:0.3e}"
            epoch_pbar.set_postfix(postfix)

            save_checkpoint(
                network=self.network,
                run_dir=self.run_dir,
                epoch=epoch,
            )

            if epoch < config.valid_tol:
                continue

            if epoch % config.valid_interval == 0:
                primary_metric = self._valid(epoch)
                if primary_metric is not None:
                    postfix["val_psnr"] = f"{primary_metric:0.2f}"
                    epoch_pbar.set_postfix(postfix)

            # best 기준은 validation PSNR
            if primary_metric is not None and primary_metric > best_metric:
                best_metric = primary_metric
                logger.success(f"Best model renew (valid psnr {best_metric:0.4f})")
                save_checkpoint(
                    network=self.network,
                    run_dir=self.run_dir,
                )

    @error_wrap
    def _valid(
        self,
        epoch: int,
    ) -> float:
        logger.info("Valid")
        with torch.no_grad():
            primary_metric = test_part(
                epoch=epoch,
                data_loader=self.valid_loader,
                network=self.network,
                run_dir=self.run_dir,
                save_val=False,
                desc=f"valid ep {epoch + 1}",
            )

        # validation 이 끝날 때마다 비교 이미지를 png 로 남긴다.
        try:
            save_comparison_figures(
                epoch=epoch,
                data_loader=self.valid_loader,
                network=self.network,
                run_dir=self.run_dir,
            )
        except Exception as err:
            logger.warning(f"Failed to save validation figures: {err}")

        return primary_metric
```


### 학습 실행


```python
# [Cell 28]
trainer = Trainer()
trainer()

TRAIN_RUN_DIR = trainer.run_dir
print("train run dir:", TRAIN_RUN_DIR)
```


## 학습 결과

best checkpoint 는 **validation PSNR** 이 가장 높았던 시점의 가중치이고,
`{run_dir}/checkpoints/checkpoint_best.ckpt` 에 저장된다.



```python
# [Cell 30]
best_ckpt = TRAIN_RUN_DIR / "checkpoints/checkpoint_best.ckpt"
print("best checkpoint:", best_ckpt, "| exists:", best_ckpt.exists())

valid_pngs = sorted((TRAIN_RUN_DIR / "valid_png").rglob("*.png"))
print(f"validation figures: {len(valid_pngs)} png in {TRAIN_RUN_DIR / 'valid_png'}")
for p in valid_pngs[-4:]:
    print(" ", p)

```


## 12. 학습이 끝난 뒤 테스트 (best checkpoint)

학습 중에는 validation 만 돌고, test 는 여기서 **best checkpoint 로 한 번만** 돌린다.
(best 는 validation PSNR 기준으로 고른 가중치다.)


결과물: `{run_dir}/test_metrics.json` (영상별 지표), `{run_dir}/test_png/*.png` (비교 이미지)


```python
# [Cell 32]
TEST_MAX_FIG: int = 8  # 저장할 비교 이미지 수

TEST_RUN_DIR = globals().get("TRAIN_RUN_DIR")
if TEST_RUN_DIR is None:
    raise RuntimeError("학습 셀을 먼저 실행하거나 TEST_RUN_DIR 을 직접 지정할 것 (예: RUN_DIR / '00000_train')")

BEST_CKPT = TEST_RUN_DIR / "checkpoints/checkpoint_best.ckpt"

if config.device is None:
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if config.init_time == 0.0:
    config.init_time = time.time()


def load_checkpoint_network(
    ckpt_path: Path,
    device: torch.device,
) -> DnCNN:
    checkpoint_data = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    if not (("model_state_dict" in checkpoint_data) and ("model_config" in checkpoint_data)):
        raise KeyError(f"Invalid checkpoint: {ckpt_path}")

    modelconfig = DnCNNConfig(**checkpoint_data["model_config"])
    network = DnCNN(
        channels=modelconfig.channels,
        num_of_layers=modelconfig.num_of_layers,
        kernel_size=modelconfig.kernel_size,
        padding=modelconfig.padding,
        features=modelconfig.features,
    )

    _state_dict = {k.replace("module.", ""): v for k, v in checkpoint_data["model_state_dict"].items()}
    network.load_state_dict(_state_dict, strict=True)

    logger.info(f"Loaded checkpoint: {ckpt_path}")
    for k, v in asdict(modelconfig).items():
        logger.info(f"{k}:{v}")

    return network.to(device).eval()


test_network = load_checkpoint_network(BEST_CKPT, config.device)

test_loader, _, test_len = get_data_wrapper_loader(
    file_path=config.test_dataset,
    training_mode=False,
    loader_cfg=LoaderConfig(
        data_type=config.data_type,
        batch=config.valid_batch,
        num_workers=config.num_workers,
        shuffle=False,
        noisy_type=config.noise_type,
        noise_sigma=config.noise_sigma,
        noisy_path=config.test_noisy_dir,
    ),
)
print(f"checkpoint : {BEST_CKPT}")
print(f"label      : {config.test_dataset[0]}")
print(f"noisy      : {config.test_noisy_dir or f'on-the-fly ({config.noise_type})'}")
print(f"test images: {test_len}")
```


```python
# [Cell 33]
noise_lookup: dict[str, str] = {}
if config.test_noisy_dir is not None:
    _meta = Path(config.test_noisy_dir) / "noise_meta.json"
    if _meta.exists():
        with open(_meta) as f:
            noise_lookup = {r["file"]: r["noise_type"] for r in json.load(f)}
        print(f"noise_meta.json: {len(noise_lookup)} entries")

test_state = MetricController()
test_state.reset()
rows: list[dict] = []

with torch.no_grad():
    for _data in tqdm(test_loader, desc="test", unit="batch"):
        label = _data[DataKey.Label].to(config.device)
        noisy = _data[DataKey.Noisy].to(config.device)
        names = _data[DataKey.Name]

        output = test_network(noisy)

        for i in range(output.shape[0]):
            out_i, lab_i, noi_i = output[i : i + 1], label[i : i + 1], noisy[i : i + 1]
            psnr_out = calculate_psnr(out_i, lab_i)
            ssim_out = calculate_ssim(out_i, lab_i)
            test_state.add("psnr", psnr_out)
            test_state.add("ssim", ssim_out)
            rows.append(
                {
                    "file": names[i],
                    "noise_type": noise_lookup.get(names[i], "unknown"),
                    "psnr": psnr_out.item(),
                    "ssim": ssim_out.item(),
                    "psnr_in": calculate_psnr(noi_i, lab_i).item(),
                    "ssim_in": calculate_ssim(noi_i, lab_i).item(),
                }
            )

log_summary(init_time=config.init_time, state=test_state, log_std=True)

with open(TEST_RUN_DIR / "test_metrics.json", "w") as f:
    json.dump(rows, f, indent=2)

print()
print(f"{'noise':<18}{'n':>5}{'PSNR in':>10}{'PSNR out':>10}{'SSIM in':>10}{'SSIM out':>10}")
print("-" * 63)
for nz in [*NOISE_RANGES.keys(), "unknown"]:
    sub = [r for r in rows if r["noise_type"] == nz]
    if not sub:
        continue
    print(
        f"{nz:<18}{len(sub):>5}"
        f"{np.mean([r['psnr_in'] for r in sub]):>10.3f}"
        f"{np.mean([r['psnr'] for r in sub]):>10.3f}"
        f"{np.mean([r['ssim_in'] for r in sub]):>10.4f}"
        f"{np.mean([r['ssim'] for r in sub]):>10.4f}"
    )
print("-" * 63)
print(
    f"{'ALL':<18}{len(rows):>5}"
    f"{np.mean([r['psnr_in'] for r in rows]):>10.3f}"
    f"{np.mean([r['psnr'] for r in rows]):>10.3f}"
    f"{np.mean([r['ssim_in'] for r in rows]):>10.4f}"
    f"{np.mean([r['ssim'] for r in rows]):>10.4f}"
)

test_png_dir = save_comparison_figures(
    epoch=0,
    data_loader=test_loader,
    network=test_network,
    run_dir=TEST_RUN_DIR,
    max_images=TEST_MAX_FIG,
    subdir="test_png",
    noise_lookup=noise_lookup,
    title_prefix="test (best ckpt)",
)
print(f"\nmetrics : {TEST_RUN_DIR / 'test_metrics.json'}")
print(f"figures : {test_png_dir}")
```


### 저장된 test 결과 이미지


```python
# [Cell 35]
from IPython.display import Image, display

for p in sorted(test_png_dir.glob("*.png"))[:3]:
    print(p.name)
    display(Image(filename=str(p)))
```


## 13. 베이스라인 비교 (mean filter / median filter)

conventional filter와 학습한 DnCNN 비교

| 방법 | 원본 | 설명 |
|---|---|---|
| DnCNN | `model/dncnn.py` | 학습된 best checkpoint |
| mean filter | `filter/mean_filter.py` | 3x3 이웃 평균 |
| median filter | `filter/median_filter.py` | 3x3 이웃 중앙값 (salt & pepper 에 강함) |

세 방법 모두 **같은 test 데이터(`test_noise_only`)** 에 대해 PSNR / SSIM 을 잰다.



```python
# [Cell 37]
def _as_bchw(img: Tensor) -> tuple[Tensor, int]:
    """[H,W] / [C,H,W] / [B,C,H,W] 를 모두 [B,C,H,W] 로 맞춘다."""
    dim = img.dim()
    if dim == 2:
        return img[None, None], dim
    if dim == 3:
        return img[None], dim
    if dim == 4:
        return img, dim
    raise ValueError(f"unsupported image dim: {dim}")


def _restore_dim(img: Tensor, dim: int) -> Tensor:
    if dim == 2:
        return img[0, 0]
    if dim == 3:
        return img[0]
    return img


def mean_filter(
    img: Tensor,
    kernel_size: int = 3,
) -> Tensor:
    x, dim = _as_bchw(img)
    pad = kernel_size // 2
    x = functional.pad(x, (pad, pad, pad, pad), mode="reflect")
    out = functional.avg_pool2d(x, kernel_size=kernel_size, stride=1)
    return _restore_dim(out, dim)


def median_filter(
    img: Tensor,
    kernel_size: int = 3,
) -> Tensor:
    x, dim = _as_bchw(img)
    pad = kernel_size // 2
    x = functional.pad(x, (pad, pad, pad, pad), mode="reflect")
    # [B, C, H, W, k, k] -> 마지막 k*k 축에서 중앙값
    patches = x.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
    out = patches.reshape(*patches.shape[:4], -1).median(dim=-1).values
    return _restore_dim(out, dim)


def adaptive_filter(
    img: Tensor,
    kernel_size: int = 5,
    noise_var: Tensor | float | None = None,
) -> Tensor:
    x, dim = _as_bchw(img)
    pad = kernel_size // 2
    xp = functional.pad(x, (pad, pad, pad, pad), mode="reflect")

    local_mean = functional.avg_pool2d(xp, kernel_size=kernel_size, stride=1)
    local_sq = functional.avg_pool2d(xp.pow(2), kernel_size=kernel_size, stride=1)
    local_var = (local_sq - local_mean.pow(2)).clamp_min(0.0)

    if noise_var is None:
        noise_var = local_var.flatten(2).median(dim=-1).values[:, :, None, None]

    ratio = (noise_var / local_var.clamp_min(1e-8)).clamp(max=1.0)
    out = x - ratio * (x - local_mean)
    return _restore_dim(out, dim)
```


```python
# [Cell 38]
BASELINE_KERNEL: int = 3
ADAPTIVE_KERNEL: int = 5

METHODS: list[str] = ["noisy", "dncnn", "mean", "median", "adaptive"]
METHOD_LABEL: dict[str, str] = {
    "noisy": "Noisy (input)",
    "dncnn": "DnCNN (best)",
    "mean": f"Mean {BASELINE_KERNEL}x{BASELINE_KERNEL}",
    "median": f"Median {BASELINE_KERNEL}x{BASELINE_KERNEL}",
    "adaptive": f"Adaptive {ADAPTIVE_KERNEL}x{ADAPTIVE_KERNEL}",
}

baseline_rows: list[dict] = []
noise_samples: dict[str, dict] = {}

test_network.eval()
with torch.no_grad():
    for _data in tqdm(test_loader, desc="baseline", unit="batch"):
        label = _data[DataKey.Label].to(config.device)
        noisy = _data[DataKey.Noisy].to(config.device)
        names = _data[DataKey.Name]

        outputs = {
            "noisy": noisy,
            "dncnn": test_network(noisy),
            "mean": mean_filter(noisy, kernel_size=BASELINE_KERNEL),
            "median": median_filter(noisy, kernel_size=BASELINE_KERNEL),
            "adaptive": adaptive_filter(noisy, kernel_size=ADAPTIVE_KERNEL),
        }

        for i in range(label.shape[0]):
            lab_i = label[i : i + 1]
            row = {"file": names[i], "noise_type": noise_lookup.get(names[i], "unknown")}
            for key in METHODS:
                pred_i = outputs[key][i : i + 1]
                row[f"psnr_{key}"] = calculate_psnr(pred_i, lab_i).item()
                row[f"ssim_{key}"] = calculate_ssim(pred_i, lab_i).item()
            baseline_rows.append(row)

            if row["noise_type"] not in noise_samples:
                sample = {"name": names[i], "metrics": row, "label": lab_i.cpu().numpy().squeeze()}
                for key in METHODS:
                    sample[key] = outputs[key][i : i + 1].cpu().numpy().squeeze()
                noise_samples[row["noise_type"]] = sample


def print_baseline_table(metric: str, width: int = 15) -> None:
    fmt = ".3f" if metric == "psnr" else ".4f"
    print(f"[{metric.upper()}]")
    print(f"{'noise':<18}{'n':>5}" + "".join(f"{METHOD_LABEL[m]:>{width}}" for m in METHODS))
    print("-" * (23 + width * len(METHODS)))
    for nz in [*NOISE_RANGES.keys(), "unknown"]:
        sub = [r for r in baseline_rows if r["noise_type"] == nz]
        if not sub:
            continue
        cells = "".join(f"{np.mean([r[f'{metric}_{m}'] for r in sub]):>{width}{fmt}}" for m in METHODS)
        print(f"{nz:<18}{len(sub):>5}{cells}")
    print("-" * (23 + width * len(METHODS)))
    cells = "".join(f"{np.mean([r[f'{metric}_{m}'] for r in baseline_rows]):>{width}{fmt}}" for m in METHODS)
    print(f"{'ALL':<18}{len(baseline_rows):>5}{cells}")


print()
print_baseline_table("psnr")
print()
print_baseline_table("ssim")

with open(TEST_RUN_DIR / "baseline_metrics.json", "w") as f:
    json.dump(baseline_rows, f, indent=2)
print(f"\nmetrics : {TEST_RUN_DIR / 'baseline_metrics.json'}")
```


### 노이즈 종류별 결과 비교 (4 x 8)


```python
# [Cell 40]
ERROR_CMAP = "magma"
GRID_COLS = ["dncnn", "mean", "median", "adaptive"]

row_order = [nz for nz in [*NOISE_RANGES.keys(), "unknown"] if nz in noise_samples]
if not row_order:
    raise RuntimeError("noise_samples 가 비어 있다. 앞의 baseline 셀을 먼저 실행할 것")

fig, axes = plt.subplots(len(row_order), 2 * len(GRID_COLS), figsize=(25, 4.2 * len(row_order)))
axes = np.asarray(axes).reshape(len(row_order), 2 * len(GRID_COLS))

for r, nz in enumerate(row_order):
    sample = noise_samples[nz]
    lab = sample["label"]
    metrics = sample["metrics"]

    vmax = float(np.percentile(lab, 98) * 1.2)
    errors = {key: np.abs(sample[key] - lab) for key in GRID_COLS}
    err_vmax = float(max(np.percentile(np.stack(list(errors.values())), 99.5), 1e-6))

    for c, key in enumerate(GRID_COLS):
        ax = axes[r, 2 * c]
        ax.imshow(sample[key], cmap="gray", vmin=0.0, vmax=vmax)
        ax.set_title(
            f"{METHOD_LABEL[key]}\nPSNR {metrics[f'psnr_{key}']:.2f} dB / SSIM {metrics[f'ssim_{key}']:.4f}",
            fontsize=9,
        )
        if c == 0:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_ylabel(f"{nz}\nnoisy {metrics['psnr_noisy']:.2f} dB", fontsize=10)
        else:
            ax.axis("off")

        ax_err = axes[r, 2 * c + 1]
        im = ax_err.imshow(errors[key], cmap=ERROR_CMAP, vmin=0.0, vmax=err_vmax)
        ax_err.set_title(f"|error|  max {errors[key].max():.3f}", fontsize=9)
        ax_err.axis("off")

    fig.colorbar(im, ax=axes[r, -1], fraction=0.046, pad=0.02)

fig.suptitle(
    f"DnCNN vs mean/median filter  |  test set: {Path(config.test_dataset[0]).name}"
    f"  |  error scale shared per row",
    fontsize=13,
)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.subplots_adjust(hspace=0.22)

grid_path = TEST_RUN_DIR / "baseline_grid.png"
fig.savefig(grid_path, dpi=150, bbox_inches="tight")
print(f"saved: {grid_path}")
plt.show()
```
