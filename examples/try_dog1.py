import torch
from torch.functional import F
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF

def fast_dog(img: torch.Tensor, sigma: float = 1.0, k: float = 1.6) -> torch.Tensor:
    """DoG как приближение LoG, separable свёртки"""

    def gaussian_1d(s):
        sz = int(4 * s + 1) | 1
        ax = torch.arange(sz, device=img.device, dtype=img.dtype) - sz // 2
        g = torch.exp(-ax ** 2 / (2 * s ** 2))
        return g / g.sum()

    def blur(x, s):
        g = gaussian_1d(s).view(1, 1, -1, 1)
        c = x.shape[1]
        x = F.conv2d(x, g.expand(c, -1, -1, -1), padding=(g.shape[2] // 2, 0), groups=c)
        g = g.transpose(2, 3)
        x = F.conv2d(x, g.expand(c, -1, -1, -1), padding=(0, g.shape[3] // 2), groups=c)
        return x

    squeeze = img.dim() == 3
    if squeeze:
        img = img.unsqueeze(0)

    dog = blur(img, sigma * k) - blur(img, sigma)
    norm = dog.abs().mean(dim=1, keepdim=True)

    return norm.squeeze(0) if squeeze else norm


if __name__ == "__main__":
    img_path = Path(r"x:\_ai\_gsplat\datasets\328220.jpg")
    img = Image.open(img_path).convert("RGB")
    img_tensor = TF.to_tensor(img)
    
    dog_result = fast_dog(img_tensor, sigma=2.0, k=3.6)
    
    dog_img = TF.to_pil_image(dog_result)
    out_path = img_path.parent / f"{img_path.stem}_dog.jpg"
    dog_img.save(out_path)
    print(f"Saved to {out_path}")