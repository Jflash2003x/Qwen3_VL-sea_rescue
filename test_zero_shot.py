import json
import os
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# ============================================================
# 配置
# ============================================================
MODEL_PATH = '/root/autodl-tmp/qwen_vl/models/Qwen/Qwen3-VL-8B-Instruct'
BASE = '/root/autodl-tmp/qwen_vl/finetune_data'

# ============================================================
# 加载模型（使用 Qwen3VL 正确的类）
# ============================================================
print('正在加载模型...')
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
print('模型加载完成！\n')

# ============================================================
# 读取验证集，取5张不同场景的图片测试
# ============================================================
with open(os.path.join(BASE, 'val.json'), 'r') as f:
    val_data = json.load(f)

# 取5张不同图片
tested_images = set()
test_samples = []
for sample in val_data:
    img_path = sample['messages'][0]['content'][0]['image']
    if img_path not in tested_images and len(test_samples) < 5:
        tested_images.add(img_path)
        test_samples.append(sample)

# ============================================================
# 逐张测试
# ============================================================
for i, sample in enumerate(test_samples):
    img_path_raw = sample['messages'][0]['content'][0]['image']
    img_path_full = os.path.join(BASE, img_path_raw)
    ground_truth = sample['messages'][1]['content'][0]['text']

    print(f'{"=" * 60}')
    print(f'测试 {i+1}/5：{img_path_raw}')
    print(f'{"=" * 60}')

    # 构建消息
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{img_path_full}"},
                {"type": "text", "text": "请检测这张无人机航拍图中水面上的所有目标，返回每个目标的类别和位置坐标。"}
            ]
        }
    ]

    # 推理
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=512)

    output_ids = output_ids[:, inputs.input_ids.shape[1]:]
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    print(f'\n【模型回答】：')
    print(response)
    print(f'\n【正确答案】：')
    print(ground_truth)
    print()

print('=' * 60)
print('零样本测试完成！请对比模型回答和正确答案的差异。')
print('微调后我们会用同样的图片再测一次，对比效果提升。')
print('=' * 60)
