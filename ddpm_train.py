import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from torch_fidelity import calculate_metrics

# ========== Dataset ==========
class SpectrogramPatchDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        fname = self.img_files[idx]
        img_path = os.path.join(self.img_dir, fname)
        img = Image.open(img_path).convert("L")

        if self.transform:
            img = self.transform(img)

        # 从文件名中解析 row 和 col
        parts = fname.split("_")
        row, col = int(parts[-2]), int(parts[-1].replace(".png", ""))
        pos = torch.tensor([row, col], dtype=torch.long)

        return img, pos

# ========== UNet ==========
class PositionalUNet(nn.Module):
    def __init__(self, in_channels=1, n_feat=64, emb_dim=64):
        super(PositionalUNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, n_feat, 3, padding=1), nn.ReLU(),
            nn.Conv2d(n_feat, n_feat, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(n_feat, 2*n_feat, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2*n_feat, n_feat, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(n_feat, in_channels, 2, stride=2), nn.Tanh()
        )
        # 位置嵌入
        self.pos_emb = nn.Embedding(512, emb_dim)  # 假设 row/col < 512
        self.fc = nn.Linear(emb_dim*2, 2*n_feat)

    def forward(self, x, pos):
        feat = self.encoder(x)
        row_emb = self.pos_emb(pos[:,0])
        col_emb = self.pos_emb(pos[:,1])
        pos_vec = torch.cat([row_emb, col_emb], dim=-1)
        pos_vec = self.fc(pos_vec).view(-1, feat.shape[1], 1, 1)
        feat = feat + pos_vec
        out = self.decoder(feat)
        return out

# ========== DDPM ==========
def ddpm_schedules(beta1, beta2, T, device):
    betas = torch.linspace(beta1, beta2, T, device=device)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrtab = torch.sqrt(alphas_cumprod)
    sqrtmab = torch.sqrt(1 - alphas_cumprod)
    return {
        "betas": betas,
        "sqrtab": sqrtab,
        "sqrtmab": sqrtmab,
    }

class DDPM(nn.Module):
    def __init__(self, nn_model, betas, n_T, device):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)
        self.device = device
        for k,v in ddpm_schedules(betas[0], betas[1], n_T, device).items():
            self.register_buffer(k, v)
        self.n_T = n_T
        self.loss_mse = nn.MSELoss()

    def forward(self, x, pos):
        _ts = torch.randint(1, self.n_T, (x.shape[0],), device=self.device)
        noise = torch.randn_like(x, device=self.device)
        x_t = self.sqrtab[_ts,None,None,None]*x + self.sqrtmab[_ts,None,None,None]*noise
        return self.loss_mse(noise, self.nn_model(x_t, pos))

    def sample(self, n_sample, size, pos=None):
        x_i = torch.randn(n_sample, *size, device=self.device)
        for i in range(self.n_T-1, -1, -1):
            eps = self.nn_model(x_i, pos if pos is not None else torch.zeros(n_sample,2,dtype=torch.long,device=self.device))
            z = torch.randn_like(x_i) if i > 0 else 0
            alpha = self.sqrtab[i]
            sigma = self.sqrtmab[i]
            beta = self.betas[i]
            x_i = alpha*x_i + sigma*eps + z*(beta**0.5)
        return x_i

# ========== Train ==========
def save_images_as_gif(images, output_gif_path, fps=10):
    fig, ax = plt.subplots()
    ax.axis('off')  # 去除坐标轴
    def update_frame(frame):
        ax.clear()
        ax.imshow(frame)
    ani = animation.FuncAnimation(fig, update_frame, images, interval=1000/fps)
    ani.save(output_gif_path, writer=PillowWriter(fps=fps))
    print(f"GIF 已保存到 {output_gif_path}")

def train_ddpm():
    data_dir = "./spectrogram_patches"
    save_dir = "./ddpm_outputs"
    os.makedirs(save_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_epoch = 20
    batch_size = 64
    n_T = 200
    lr = 1e-4

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = SpectrogramPatchDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    ddpm = DDPM(PositionalUNet(), betas=(1e-4, 0.02), n_T=n_T, device=device)
    optim = torch.optim.Adam(ddpm.parameters(), lr=lr)

    loss_values = []  # 用于记录每个 epoch 的损失值
    fid_values = []   # 用于记录每个 epoch 的 FID 值

    for ep in range(n_epoch):
        ddpm.train()
        pbar = tqdm(dataloader)
        loss_ema = None
        all_generated_images = []  # 用于收集每个 epoch 生成的图像
        real_images = []  # 用于收集真实图像

        for x, pos in pbar:
            optim.zero_grad()
            x, pos = x.to(device), pos.to(device)
            loss = ddpm(x, pos)
            loss.backward()
            optim.step()
            loss_ema = loss.item() if loss_ema is None else 0.95*loss_ema+0.05*loss.item()
            pbar.set_description(f"Epoch {ep} Loss: {loss_ema:.4f}")

        loss_values.append(loss_ema)  # 记录当前 epoch 的损失

        # 每轮保存生成的图像并生成 GIF
        ddpm.eval()
        with torch.no_grad():
            samples = ddpm.sample(4, (1,128,128), pos=torch.zeros(4,2,dtype=torch.long,device=device))
            all_generated_images.append(samples.cpu())  # 将生成的图片收集起来
            real_images.append(x.cpu())  # 收集真实图像

        # 计算 FID
        if (ep + 1) % 5 == 0:  # 每 5 个 epoch 计算一次 FID
            generated_images = torch.stack(all_generated_images[-1])  # 获取最新生成的图像
            real_images = torch.stack(real_images[-1])  # 获取最新的真实图像
            fid = calculate_metrics(
                input1=generated_images.numpy(), input2=real_images.numpy(),
                metric='fid', device=device)
            fid_values.append(fid)
            print(f"Epoch {ep} FID: {fid}")

        # 每个 epoch 保存一个 GIF 动画
        if (ep + 1) % 5 == 0:  # 每 5 个 epoch 保存一次 GIF
            gif_path = os.path.join(save_dir, f"epoch_{ep}_generated.gif")
            save_images_as_gif(all_generated_images[-1], gif_path, fps=5)

        # 保存示例图像
        with torch.no_grad():
            grid = make_grid(samples, nrow=2, normalize=True)
            save_image(grid, os.path.join(save_dir, f"sample_ep{ep}.png"))
            print(f"保存示例图像到 {save_dir}/sample_ep{ep}.png")

    # 绘制损失曲线
    plt.plot(range(n_epoch), loss_values, label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    print(f"损失曲线已保存到 {os.path.join(save_dir, 'loss_curve.png')}")

    # 绘制 FID 曲线
    plt.plot(range(5, n_epoch+1, 5), fid_values, label="FID")
    plt.xlabel("Epochs")
    plt.ylabel("FID")
    plt.title("FID Curve")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "fid_curve.png"))
    print(f"FID 曲线已保存到 {os.path.join(save_dir, 'fid_curve.png')}")

if __name__ == "__main__":
    train_ddpm()
