import os
import math
import torch
import numpy as np
import scipy.io as sio
from PIL import Image
from tqdm import tqdm


def split_and_save_patches_from_mat_folder(mat_folder, out_dir="spectrogram_patches", patch_size=(128, 128)):
    os.makedirs(out_dir, exist_ok=True)

    mat_files = [os.path.join(mat_folder, f) for f in os.listdir(mat_folder) if f.endswith('.mat')]

    patch_meta = []

    for mat_file in tqdm(mat_files, desc="Processing mat files"):
        # 读取 .mat 文件
        data = sio.loadmat(mat_file)
        if "measurementResults" not in data:
            print(f"⚠️ 跳过 {mat_file}，没有找到 measurementResults")
            continue
        spec = data["measurementResults"]  # shape: (time, freq)

        # 归一化到 [0,255]
        spec_min, spec_max = spec.min(), spec.max()
        spec_norm = (spec - spec_min) / (spec_max - spec_min + 1e-8)
        spec_img = (spec_norm * 255).astype(np.uint8)

        H, W = spec_img.shape
        ph, pw = patch_size

        # 计算可以切多少个 patch
        n_h = math.ceil(H / ph)
        n_w = math.ceil(W / pw)

        for i in range(n_h):
            for j in range(n_w):
                patch = spec_img[i * ph:(i + 1) * ph, j * pw:(j + 1) * pw]

                # 如果不足 patch 大小，padding
                patch_padded = np.zeros((ph, pw), dtype=np.uint8)
                patch_padded[:patch.shape[0], :patch.shape[1]] = patch

                # 保存为 PNG
                base_name = os.path.splitext(os.path.basename(mat_file))[0]
                fname = f"{base_name}_patch_{i}_{j}.png"
                patch_path = os.path.join(out_dir, fname)
                Image.fromarray(patch_padded).save(patch_path)

                # 保存元数据
                patch_meta.append({"file": fname, "row": i, "col": j, "mat_file": base_name})

    torch.save(patch_meta, os.path.join(out_dir, "patch_metadata.pt"))
    print(f"✅ 已保存 {len(patch_meta)} 个 patch 到 {out_dir}")


if __name__ == "__main__":
    mat_folder = "spectrum_mat_data"  # 修改为存放 .mat 文件的文件夹路径
    split_and_save_patches_from_mat_folder(mat_folder, out_dir="spectrogram_patches")
