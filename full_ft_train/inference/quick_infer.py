"""
==========================================================================
快速推理脚本：vLLM 推理两个模型 + 绘制对比图
==========================================================================

功能：
  1. 用 vLLM 加载两个模型（原始 + 微调）
  2. 推理少量图片（1-5 张）
  3. 绘制检测结果对比图（GT / 原始模型 / 微调模型）

输出：
  quick_output/
  ├── predictions.json          - 推理结果
  └── comparison_*.png          - 对比图
"""

# ===== 必须在最前面修复环境变量 =====
import os
if 'OMP_NUM_THREADS' in os.environ:
    del os.environ['OMP_NUM_THREADS']
os.environ['OMP_NUM_THREADS'] = '1'

import json
import re
import time
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# ============================================================
# 配置
# ============================================================

MODEL_BASE_PATH = '/root/autodl-tmp/qwen_vl/models/Qwen/Qwen3-VL-8B-Instruct'
MODEL_V2_PATH = '/root/autodl-tmp/qwen_vl/models/Qwen3-VL-8B-Instruct-merged-v2'

DATA_DIR = '/root/autodl-tmp/qwen_vl/finetune_data'
VAL_JSON = os.path.join(DATA_DIR, 'val.json')
VAL_IMG_DIR = os.path.join(DATA_DIR, 'val')

OUTPUT_DIR = 'quick_output'
NUM_TEST = 3  # 只推理 3 张图片快速测试

PROMPT = "请检测这张无人机航拍图中水面上的所有目标，返回每个目标的类别和位置坐标。返回格式必须是 JSON: [{\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"类别\"}]"

LABEL_MAP = {
    '人': '水中人员', 'person': '水中人员', '人员': '水中人员',
    '游泳的人': '水中人员', '溺水者': '水中人员', '游泳者': '水中人员',
    '水中人员': '水中人员', 'swimmer': '水中人员',
    'boat': '船只', '船': '船只', '船只': '船只', '快艇': '船只', '渔船': '船只',
    '浮标': '浮标', '浮球': '浮标', '漂浮物': '浮标', 'buoy': '浮标',
    '救生设备': '救生设备', '救生圈': '救生设备', '救生衣': '救生设备',
    '水上摩托': '水上摩托', 'jetski': '水上摩托', '摩托艇': '水上摩托',
}
VALID_LABELS = {'水中人员', '船只', '水上摩托', '救生设备', '浮标'}

# 绘图颜色（RGB）
BOX_COLORS = {
    'gt': (255, 0, 0),        # 红色 - GT
    'base': (0, 255, 0),      # 绿色 - 原始模型
    'v2': (0, 0, 255),        # 蓝色 - 微调模型
}


# ============================================================
# 解析函数
# ============================================================

def parse_gt_from_text(text):
    """解析 GT（归一化 0-1000）"""
    targets = []
    pattern = r'(水中人员|船只|水上摩托|救生设备|浮标)：\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)'
    for m in re.findall(pattern, text):
        targets.append({
            'label': m[0],
            'bbox_2d': [int(m[1]), int(m[2]), int(m[3]), int(m[4])]
        })
    return targets


def parse_model_response(text):
    """
    解析模型输出（vLLM 输出）
    坐标为归一化 0-1000
    """
    targets = []

    # 尝试找 JSON 格式
    json_match = re.search(r'\[.*?\]', text, re.DOTALL)
    if json_match:
        try:
            items = json.loads(json_match.group())
            for item in items:
                if isinstance(item, dict):
                    bbox = item.get('bbox_2d') or item.get('bbox')
                    label = item.get('label') or item.get('类别')
                    
                    if bbox and label and len(bbox) == 4:
                        label = LABEL_MAP.get(str(label), str(label))
                        if label in VALID_LABELS:
                            targets.append({
                                'label': label,
                                'bbox_2d': [int(b) for b in bbox]
                            })
            if targets:
                return targets
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # 尝试找中文格式 "水中人员：(x1, y1, x2, y2)"
    pattern_cn = r'([\u4e00-\u9fff]+)[\s：:]*[\(（]\s*(\d+)\s*[,，]\s*(\d+)\s*[,，]\s*(\d+)\s*[,，]\s*(\d+)\s*[\)）]'
    for m in re.findall(pattern_cn, text):
        label = LABEL_MAP.get(m[0], m[0])
        if label in VALID_LABELS:
            targets.append({
                'label': label,
                'bbox_2d': [int(m[1]), int(m[2]), int(m[3]), int(m[4])]
            })

    return targets


# ============================================================
# 准备测试数据
# ============================================================

def prepare_test_data(num_test):
    """准备测试图片"""
    print(f"  读取 {VAL_JSON}...")
    
    with open(VAL_JSON, 'r', encoding='utf-8') as f:
        val_data = json.load(f)

    seen = set()
    test_items = []
    
    for sample in val_data:
        img_rel = sample['messages'][0]['content'][0]['image']
        img_name = os.path.basename(img_rel)
        
        if img_name in seen:
            continue

        # 解析 GT
        gt_text = sample['messages'][1]['content'][0]['text']
        gts = parse_gt_from_text(gt_text)
        
        if not gts:
            continue

        # 检查图片是否存在
        img_path = os.path.join(VAL_IMG_DIR, img_name)
        if not os.path.exists(img_path):
            continue

        seen.add(img_name)
        test_items.append({
            'image_name': img_name,
            'image_path': img_path,
            'gt': gts,
        })

        if len(test_items) >= num_test:
            break

    print(f"  ✅ 选取 {len(test_items)} 张图片")
    return test_items


# ============================================================
# vLLM 推理
# ============================================================

def infer_with_vllm(model_path, test_items, model_name):
    """用 vLLM 推理"""
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor

    print(f"\n  【{model_name}】")
    print(f"  加载模型: {model_path}...")
    
    t0 = time.time()
    
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        max_model_len=8192,
        max_num_seqs=4,
        gpu_memory_utilization=0.8,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={
            "min_pixels": 512 * 28 * 28,
            "max_pixels": 1280 * 28 * 28,
        },
        tensor_parallel_size=1,
    )

    processor = AutoProcessor.from_pretrained(model_path)
    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=1024,
        stop=["<|im_end|>"],
    )

    load_time = time.time() - t0
    print(f"  ✅ 模型加载完成 ({load_time:.1f}s)")

    # 构建推理请求
    print(f"  构建 {len(test_items)} 个推理请求...")
    requests = []
    
    for item in test_items:
        pil_image = Image.open(item['image_path']).convert("RGB")
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{item['image_path']}"},
                {"type": "text", "text": PROMPT},
            ]
        }]
        
        prompt_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        requests.append({
            "prompt": prompt_text,
            "multi_modal_data": {"image": pil_image},
        })

    # 批量推理
    print(f"  开始推理...")
    t1 = time.time()
    outputs = llm.generate(requests, sampling_params=sampling_params)
    infer_time = time.time() - t1

    # 解析结果
    predictions = []
    for i, (item, output) in enumerate(zip(test_items, outputs)):
        response = output.outputs[0].text
        preds = parse_model_response(response)

        print(f"    [{i+1}] {item['image_name']}: {len(preds)} 个预测")

        predictions.append({
            'image_name': item['image_name'],
            'preds': preds,
            'raw_response': response[:300],
        })

    print(f"  推理耗时: {infer_time:.1f}s")
    
    # 释放显存
    del llm
    torch.cuda.empty_cache()

    return predictions


# ============================================================
# 绘图函数
# ============================================================

def draw_boxes_on_image(image, boxes, color, label_prefix=""):
    """在图像上绘制检测框
    
    Args:
        image: PIL Image
        boxes: [{"bbox_2d": [x1, y1, x2, y2], "label": "..."}, ...]
        color: (R, G, B)
        label_prefix: 标签前缀（如 "GT", "Base", "V2"）
    """
    draw = ImageDraw.Draw(image)
    img_w, img_h = image.size

    for box in boxes:
        bbox = box['bbox_2d']
        label = box['label']

        # 归一化坐标 (0-1000) 转像素坐标
        x1 = int(bbox[0] / 1000 * img_w)
        y1 = int(bbox[1] / 1000 * img_h)
        x2 = int(bbox[2] / 1000 * img_w)
        y2 = int(bbox[3] / 1000 * img_h)

        # 绘制框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # 绘制标签
        text = f"{label_prefix}{label}" if label_prefix else label
        draw.text((x1, max(y1 - 15, 0)), text, fill=color)

    return image


def create_comparison_image(gt_list, base_preds, v2_preds, image_path):
    """创建对比图（3 列）"""
    
    # 打开原图 3 份
    img_gt = Image.open(image_path).convert("RGB")
    img_base = Image.open(image_path).convert("RGB")
    img_v2 = Image.open(image_path).convert("RGB")

    # 绘制框
    img_gt = draw_boxes_on_image(img_gt, gt_list, BOX_COLORS['gt'], "GT:")
    img_base = draw_boxes_on_image(img_base, base_preds, BOX_COLORS['base'], "Base:")
    img_v2 = draw_boxes_on_image(img_v2, v2_preds, BOX_COLORS['v2'], "V2:")

    # 水平拼接（3 列）
    total_width = img_gt.width * 3
    total_height = img_gt.height
    combined = Image.new('RGB', (total_width, total_height))
    
    combined.paste(img_gt, (0, 0))
    combined.paste(img_base, (img_gt.width, 0))
    combined.paste(img_v2, (img_gt.width * 2, 0))

    return combined


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 70)
    print("  快速推理：vLLM 两模型对比")
    print("=" * 70)

    gpu_name = torch.cuda.get_device_name(0)
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\n  GPU: {gpu_name} ({vram_total:.0f} GB)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 准备测试数据
    print(f"\n【1】准备测试数据")
    test_items = prepare_test_data(NUM_TEST)
    
    if not test_items:
        print("❌ 没有找到有效的测试图片")
        return

    # 2. 推理原始模型
    print(f"\n【2】推理原始模型")
    base_results = infer_with_vllm(MODEL_BASE_PATH, test_items, "原始模型")

    # 3. 推理 V2 微调模型
    print(f"\n【3】推理 V2 微调模型")
    v2_results = infer_with_vllm(MODEL_V2_PATH, test_items, "V2 微调模型")

    # 4. 保存推理结果
    print(f"\n【4】保存推理结果")
    results_data = {
        'base': base_results,
        'v2': v2_results,
    }
    results_file = os.path.join(OUTPUT_DIR, 'predictions.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)
    print(f"  ✅ {results_file}")

    # 5. 绘制对比图
    print(f"\n【5】绘制对比图")
    for i, (item, base_pred, v2_pred) in enumerate(zip(test_items, base_results, v2_results)):
        print(f"  生成对比图 {i+1}/{len(test_items)}: {item['image_name']}")

        combined_img = create_comparison_image(
            item['gt'],
            base_pred['preds'],
            v2_pred['preds'],
            item['image_path']
        )

        # 保存
        output_file = os.path.join(OUTPUT_DIR, f'comparison_{i+1:02d}_{item["image_name"]}')
        combined_img.save(output_file)
        print(f"    ✅ {output_file}")

    # 6. 总结
    print(f"\n{'='*70}")
    print(f"  ✅ 快速推理完成！")
    print(f"\n  输出目录: {OUTPUT_DIR}")
    print(f"  - predictions.json: 推理结果")
    print(f"  - comparison_*.png: 对比图（GT / 原始 / V2）")
    print(f"\n  图例：")
    print(f"    🔴 GT（红框）")
    print(f"    🟢 原始模型（绿框）")
    print(f"    🔵 V2 微调模型（蓝框）")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()