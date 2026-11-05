import torch
import numpy as np
from pathlib import Path
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3._visualization_utils import plot_results

# 1. 加载模型和处理器
model = build_sam3_image_model(
    checkpoint_path="/root/autodl-tmp/sam3.pt"
)
processor = Sam3Processor(
    model,
    confidence_threshold=0.45  # 只保留置信度较高的实例
)


# 2. 图片所在文件夹
#IMAGE_DIR = Path("/root/sam3-main/assets/images/7.22/zao")  

# 3. 结果目录 - 确保输出到 results/5.28_zao 子目录
RESULT_DIR = Path("/root/autodl-tmp/plot_results/results_23/6.4") # 新增子目录
RESULT_DIR.mkdir(parents=True, exist_ok=True)  # 自动创建目录（包括父目录）

# 4. 统一的文本提示
text_prompt = "pine seedling leaves, needle leaves"

# 5. 处理图片
for i in range(1, 2):  # 这里先跑一张图，需要多张就改 range
    img_path = Path("/root/sam3-main/assets_p1/6.4/zao/position1.png")
    save_path = RESULT_DIR / f"position{i}.png"
    print(f"[INFO] 正在处理图片: {img_path}")
    # 读入图片
    image = Image.open(img_path)
    # 设置当前图片并推理
    inference_state = processor.set_image(image)
    output = processor.set_text_prompt(state=inference_state, prompt=text_prompt)

    # 正确的路径拼接方式：使用 Path 对象的 / 运算符
    save_name = f'{img_path.name}'  # 先组合文件名
    save_path = RESULT_DIR / save_name  # 再与目录拼接

    # 保存结果1：原图 + 模型分割掩膜
    plot_results(image, output, save_name=str(save_path))

    # 保存结果2：黑色背景 + 白色前景二值图
    masks = output.get("masks", None)

    image_w, image_h = image.size
    binary_mask = np.zeros((image_h, image_w), dtype=np.uint8)

    if masks is not None:
        masks_t = masks if isinstance(masks, torch.Tensor) else torch.as_tensor(masks)

        # 兼容 [N, 1, H, W]、[N, H, W] 和单张 [H, W]
        if masks_t.ndim == 4 and masks_t.shape[1] == 1:
            masks_t = masks_t[:, 0]
        elif masks_t.ndim == 2:
            masks_t = masks_t.unsqueeze(0)

        if masks_t.ndim != 3:
            raise ValueError(f"Unexpected masks shape: {tuple(masks_t.shape)}")

        # 将所有预测实例 mask 合并为一个前景区域
        for j in range(int(masks_t.shape[0])):
            mask = masks_t[j].detach().cpu().numpy()
            mask = mask > 0
            binary_mask[mask] = 255

    binary_save_path = save_path.with_name(
        f"{save_path.stem}_binary.png"
    )
    Image.fromarray(binary_mask, mode="L").save(binary_save_path)

    print(f"[OK] 掩膜叠加图已保存: {save_path}")
    print(f"[OK] 二值图已保存: {binary_save_path}")

print("[DONE] 所有图片处理完成。")
