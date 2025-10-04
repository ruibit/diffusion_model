import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import os
import glob


def mat_to_spectrum_png(mat_file, output_dir="output_images", cmap="jet"):
    """
    将 .mat 文件中的频谱数据转换为 PNG 图片
    参数:
        mat_file: str, 输入的 .mat 文件路径
        output_dir: str, 输出图片目录
        cmap: str, matplotlib 色图名称 (如 'jet', 'viridis', 'plasma')
    """
    # 读取 mat 文件
    data = sio.loadmat(mat_file)

    # 提取频谱矩阵
    if "measurementResults" not in data:
        raise KeyError(f"在文件 {mat_file} 中未找到 'measurementResults' 变量")

    spectrum = np.array(data["measurementResults"])  # shape (time, freq)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 绘制频谱图
    plt.figure(figsize=(10, 6))
    plt.imshow(spectrum, aspect='auto', origin='lower', cmap=cmap)
    plt.colorbar(label="Power (dBm)")
    plt.xlabel("Frequency bin")
    plt.ylabel("Time frame")
    plt.title(f"Spectrum (dBm) - {os.path.basename(mat_file)}")
    plt.tight_layout()

    # 保存为 PNG
    output_path = os.path.join(output_dir, f"{os.path.basename(mat_file).replace('.mat', '.png')}")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"文件 {mat_file} 的频谱图已保存到: {output_path}")


def batch_convert_mat_files(input_dir, output_dir="spectrum_images", cmap="jet"):
    """
    批量转换目录中的所有 .mat 文件为频谱图 PNG
    参数:
        input_dir: str, 输入的 .mat 文件所在目录
        output_dir: str, 输出图片目录
        cmap: str, matplotlib 色图名称
    """
    # 获取所有.mat文件
    mat_files = glob.glob(os.path.join(input_dir, "*.mat"))

    if not mat_files:
        print("没有找到 .mat 文件。")
        return

    print(f"共找到 {len(mat_files)} 个 .mat 文件，开始转换...")

    # 对每个 .mat 文件进行转换
    for mat_file in mat_files:
        try:
            mat_to_spectrum_png(mat_file, output_dir, cmap)
        except Exception as e:
            print(f"转换文件 {mat_file} 时发生错误: {e}")


if __name__ == "__main__":
    input_dir = "spectrum_mat_data"  # 替换为包含 .mat 文件的目录路径
    batch_convert_mat_files(input_dir)
