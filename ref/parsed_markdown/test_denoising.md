# test_denoising.ipynb


# Model inference

학습이 끝난 뒤, **best checkpoint 로 test 데이터셋에 대해 검증**한다.

mean filter, median filter 등 **conventional method와 결과를 비교**한다.


1. dataset 폴더의 **test_noise_only** 데이터를 모델의 input으로 넣어서, **test_label(정답)** 데이터를 이용해 metric을 측정해 주세요.
test_noise_only/noise_meta.json 에서 각각의 이미지가 어떤 노이즈에 해당하는지 확인하실 수 있습니다.
아래 표와 같이 noise 방법 별 metric을 측정해주세요.

![image.png](data:image/png;base64,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)

2. metric을 측정을 위해 아래 제공한 **psnr, ssim** 코드를 사용해 주세요.

3. 아래 예시와 같이 **test 결과를 시각화** 해 주세요. 코드는 동일하지 않아도 괜찮습니다.

4. mean/median/adaptive filter 등 **conventional method와 결과를 비교**해주세요. 아래 제공하는 comparison 코드의 DnCNN 자리를 학습한 모델 결과로 대체해 주세요.


# PSNR / SSIM


```python
# [Cell 3]
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


# test 결과 이미지 시각화 예시


```python
# [Cell 5]
from IPython.display import Image, display

for p in sorted(test_png_dir.glob("*.png"))[:3]:
    print(p.name)
    display(Image(filename=str(p)))
```


# Comparison with conventional method
(mean filter / median filter / adaptive filter)

세 방법 모두 **같은 test 데이터(`test_noise_only`)** 에 대해 PSNR / SSIM 을 잰다.



```python
# [Cell 7]
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
# [Cell 8]
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


# 노이즈 종류별 결과 시각화


```python
# [Cell 10]
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
