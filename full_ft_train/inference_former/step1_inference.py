"""
==========================================================================
Step 1: 双模型批量推理（RTX Pro 6000 96GB 优化版）
==========================================================================

硬件：RTX Pro 6000 96GB VRAM
优化策略：
  1. 两个模型都用 vLLM + bfloat16 全精度，不量化
  2. vLLM 批处理推理，榨干 GPU 吞吐
  3. max_pixels 提升到 1280*28*28，提高小目标识别率
  4. 测试 200 张图片，数据量更充分
  5. 并发序列数 max_num_seqs=8，充分利用 96GB 显存

输出：
  output/inference_base.json   - 原始模型推理结果
  output/inference_v2.json     - V2 微调模型推理结果
  output/ground_truth.json     - GT 标注
  output/image_list.json       - 测试图片列表

所有坐标均为归一化 0-1000（GT 来自 convert.py 的 bbox_to_normalized，
原始模型的 bbox_2d 也是 0-1000，V2 模型学的也是 0-1000）。
==========================================================================
"""

import os
os.environ["OMP_NUM_THREADS"] = str(max(1, os.cpu_count() or 1))

import json
import re
import time
import torch
from PIL import Image

# ============================================================
# 配置
# ============================================================
MODEL_BASE_PATH = '/root/autodl-tmp/qwen_vl/models/Qwen/Qwen3-VL-8B-Instruct'
MODEL_V2_PATH = '/root/autodl-tmp/qwen_vl/full_ft_train/full_ft_output/best_model'

DATA_DIR = '/root/autodl-tmp/qwen_vl/finetune_data'
VAL_JSON = os.path.join(DATA_DIR, 'val.json')
VAL_IMG_DIR = os.path.join(DATA_DIR, 'val')

OUTPUT_DIR = '/root/autodl-tmp/qwen_vl/full_ft_train/inference_former/inference_output'

NUM_TEST = 1547                    # 96GB 跑 200 张无压力
MAX_TOKENS = 1024                # 输出 token 上限
VLLM_MAX_MODEL_LEN = 8192       # 上下文长度
VLLM_MAX_NUM_SEQS = 8           # 并发请求数，96GB 足够
VLLM_GPU_UTIL = 0.90            # GPU 显存利用率

# 96GB 可以用更高分辨率，小目标识别更好
MIN_PIXELS = 512 * 28 * 28      # ~401K pixels
MAX_PIXELS = 1280 * 28 * 28     # ~1.0M pixels

PROMPT = "请检测这张无人机航拍图中水面上的所有目标，返回每个目标的类别和位置坐标。"

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


# ============================================================
# 解析函数
# ============================================================

def parse_gt_from_text(text):
    """解析 val.json assistant 回复中的 GT（归一化 0-1000）"""
    targets = []
    pattern = r'(水中人员|船只|水上摩托|救生设备|浮标)：\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)'
    for m in re.findall(pattern, text):
        targets.append({'label': m[0], 'bbox': [int(m[1]), int(m[2]), int(m[3]), int(m[4])]})
    return targets


def parse_model_response(text):
    """
    解析模型输出。所有坐标直接当作 0-1000 归一化坐标，不做转换。
    - 原始模型 bbox_2d: Qwen3-VL 默认输出就是归一化 0-1000
    - V2 微调模型: 学的就是 0-1000 格式
    """
    targets = []

    # 格式1: 中文文本 "水中人员：(571, 306, 649, 397)"
    pattern_cn = r'([\u4e00-\u9fff]+)[\s：:]*[\(（]\s*(\d+)\s*[,，]\s*(\d+)\s*[,，]\s*(\d+)\s*[,，]\s*(\d+)\s*[\)）]'
    for m in re.findall(pattern_cn, text):
        label = LABEL_MAP.get(m[0], m[0])
        if label in VALID_LABELS:
            targets.append({'label': label, 'bbox': [int(m[1]), int(m[2]), int(m[3]), int(m[4])]})
    if targets:
        return targets

    # 格式2: JSON {"bbox_2d": [581, 309, 637, 389], "label": "人"}
    json_match = re.search(r'\[.*\]', text, re.DOTALL)
    if json_match:
        try:
            items = json.loads(json_match.group())
            for item in items:
                if isinstance(item, dict):
                    bbox = item.get('bbox_2d') or item.get('bbox') or item.get('坐标')
                    label = item.get('label') or item.get('类别') or item.get('目标')
                    if bbox and label and len(bbox) == 4:
                        label = LABEL_MAP.get(str(label), str(label))
                        if label in VALID_LABELS:
                            targets.append({'label': label, 'bbox': [int(b) for b in bbox]})
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    return targets


# ============================================================
# 准备测试数据
# ============================================================

def prepare_test_data():
    with open(VAL_JSON, 'r', encoding='utf-8') as f:
        val_data = json.load(f)

    seen = set()
    test_items = []
    for sample in val_data:
        img_rel = sample['messages'][0]['content'][0]['image']
        img_name = os.path.basename(img_rel)
        if img_name in seen:
            continue

        gt_text = sample['messages'][1]['content'][0]['text']
        gts = parse_gt_from_text(gt_text)
        if not gts:
            continue

        img_path = os.path.join(VAL_IMG_DIR, img_name)
        if not os.path.exists(img_path):
            continue

        seen.add(img_name)
        test_items.append({
            'image_name': img_name,
            'image_path': img_path,
            'gt': gts,
        })

        if len(test_items) >= NUM_TEST:
            break

    return test_items


# ============================================================
# vLLM 批量推理
# ============================================================

def batch_infer_vllm(model_path, test_items, model_name):
    """
    用 vLLM 批量推理，充分利用 96GB 显存。
    vLLM 内部会自动调度并发，比逐张快很多。
    """
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor

    print(f"\n  加载 vLLM 模型: {model_name}...")
    t0 = time.time()

    llm = LLM(
        model=model_path,
        dtype="bfloat16",                     # 96GB 无需量化，全精度
        max_model_len=VLLM_MAX_MODEL_LEN,
        max_num_seqs=VLLM_MAX_NUM_SEQS,       # 并发 8 个请求
        gpu_memory_utilization=VLLM_GPU_UTIL,  # 用 90% 显存
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={
            "min_pixels": MIN_PIXELS,
            "max_pixels": MAX_PIXELS,          # 高分辨率
        },
        tensor_parallel_size=1,
    )

    processor = AutoProcessor.from_pretrained(model_path)
    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=MAX_TOKENS,
        stop=["<|im_end|>"],
    )

    load_time = time.time() - t0
    vram = torch.cuda.max_memory_allocated() / 1024**3
    print(f"  ✅ 加载完成 | 耗时: {load_time:.1f}s | 显存: {vram:.1f}GB")

    # 构建全部请求
    print(f"  构建 {len(test_items)} 个推理请求...")
    requests = []
    for item in test_items:
        pil_image = Image.open(item['image_path']).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": f"file://{item['image_path']}"},
            {"type": "text", "text": PROMPT},
        ]}]
        prompt_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        requests.append({
            "prompt": prompt_text,
            "multi_modal_data": {"image": pil_image},
        })

    # 批量推理
    print(f"  开始批量推理...")
    t1 = time.time()
    outputs = llm.generate(requests, sampling_params=sampling_params)
    total_infer_time = time.time() - t1

    # 解析结��
    predictions = []
    for i, (item, output) in enumerate(zip(test_items, outputs)):
        response = output.outputs[0].text
        preds = parse_model_response(response)

        predictions.append({
            'image_name': item['image_name'],
            'preds': [{'label': p['label'], 'bbox': p['bbox']} for p in preds],
            'time': round(total_infer_time / len(test_items), 2),  # 平均时间
            'raw_response': response[:400],
        })

        # 进度打印（每 20 张）
        if (i + 1) % 20 == 0 or i < 3:
            gt_n = len(item['gt'])
            pred_n = len(preds)
            print(f"    [{i+1:3d}/{len(test_items)}] {item['image_name']:<14} GT:{gt_n} Pred:{pred_n}")

    throughput = len(test_items) / total_infer_time
    print(f"\n  📊 {model_name} 推理统计:")
    print(f"     总耗时: {total_infer_time:.1f}s | 吞吐: {throughput:.2f} 张/秒")
    print(f"     平均每张: {total_infer_time/len(test_items):.2f}s")

    # 释放
    del llm
    torch.cuda.empty_cache()
    print(f"  显存已释放")

    return predictions, total_infer_time


# ============================================================
# 主流程
# ============================================================

def main():
    gpu_name = torch.cuda.get_device_name(0)
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3

    print("=" * 60)
    print("  Step 1: 双模型批量推理（vLLM）")
    print(f"  GPU: {gpu_name} ({vram_total:.0f} GB)")
    print(f"  测试图片数: {NUM_TEST}")
    print(f"  图片分辨率: min={MIN_PIXELS/(28*28):.0f}×28² max={MAX_PIXELS/(28*28):.0f}×28²")
    print(f"  并发数: {VLLM_MAX_NUM_SEQS}")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 准备数据
    print("\n[1/4] 准备测试数据...")
    test_items = prepare_test_data()
    print(f"  选取了 {len(test_items)} 张图片")
    total_gt = sum(len(item['gt']) for item in test_items)
    print(f"  GT 目标总数: {total_gt}")

    # 保存 GT
    gt_data = {item['image_name']: item['gt'] for item in test_items}
    gt_path = os.path.join(OUTPUT_DIR, 'ground_truth.json')
    with open(gt_path, 'w', encoding='utf-8') as f:
        json.dump(gt_data, f, ensure_ascii=False, indent=2)

    # 保存图片列表（step3 画图用）
    img_list = [{'image_name': item['image_name'], 'image_path': item['image_path']} for item in test_items]
    with open(os.path.join(OUTPUT_DIR, 'image_list.json'), 'w', encoding='utf-8') as f:
        json.dump(img_list, f, ensure_ascii=False, indent=2)

    print(f"  GT 保存到: {gt_path}")

    # 推理原始模型
    print("\n[2/4] 推理原始模型...")
    base_preds, base_time = batch_infer_vllm(MODEL_BASE_PATH, test_items, "原始模型 Qwen3-VL-8B")

    base_path = os.path.join(OUTPUT_DIR, 'inference_base.json')
    with open(base_path, 'w', encoding='utf-8') as f:
        json.dump(base_preds, f, ensure_ascii=False, indent=2)
    print(f"  保存到: {base_path}")

    # 推理 V2 微调模型
    print("\n[3/4] 推理 V2 微调模型...")
    v2_preds, v2_time = batch_infer_vllm(MODEL_V2_PATH, test_items, "V2 微调模型")

    v2_path = os.path.join(OUTPUT_DIR, 'inference_v2.json')
    with open(v2_path, 'w', encoding='utf-8') as f:
        json.dump(v2_preds, f, ensure_ascii=False, indent=2)
    print(f"  保存到: {v2_path}")

    # 坐标系验证
    print("\n[4/4] 坐标系验证（前 3 张）:")
    for item, bp, vp in zip(test_items[:3], base_preds[:3], v2_preds[:3]):
        gts = item['gt']
        print(f"  {item['image_name']}:")
        if gts:
            print(f"    GT[0]:        {gts[0]}")
        if bp['preds']:
            print(f"    Base Pred[0]: {bp['preds'][0]}")
        if vp['preds']:
            print(f"    V2 Pred[0]:   {vp['preds'][0]}")

    print(f"\n{'='*60}")
    print(f"  ✅ Step 1 完成！")
    print(f"  原始模型: {base_time:.1f}s ({len(test_items)/base_time:.2f} 张/秒)")
    print(f"  V2 模型:  {v2_time:.1f}s ({len(test_items)/v2_time:.2f} 张/秒)")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()