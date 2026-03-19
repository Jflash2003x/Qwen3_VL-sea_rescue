"""
调试脚本：查看预测和 GT 的格式
"""

import json
import os

inference_output_dir = "/root/autodl-tmp/qwen_vl/full_ft_train/inference/inference_output"
gt_json = "/root/autodl-tmp/qwen_vl/finetune_data_v2/test_split.json"

# 加载数据
with open(os.path.join(inference_output_dir, "original_predictions.json"), 'r') as f:
    predictions = json.load(f)

with open(gt_json, 'r') as f:
    gt_data = json.load(f)

# 创建 GT 映射
gt_map = {}
for sample in gt_data:
    img_path = sample["messages"][0]["content"][0]["image"]
    gt_text = sample["messages"][1]["content"]
    try:
        gt_list = json.loads(gt_text)
        gt_map[img_path] = gt_list
    except:
        gt_map[img_path] = []

print("="*70)
print("预测格式检查")
print("="*70)

# 检查前 3 个预测
for i in range(min(3, len(predictions))):
    pred = predictions[i]
    print(f"\n【预测 #{i}】")
    print(f"  image_path: {pred.get('image_path')}")
    print(f"  num_predictions: {pred.get('num_predictions')}")
    if pred.get('predictions'):
        print(f"  第一个预测:")
        p = pred['predictions'][0]
        print(f"    {json.dumps(p, indent=4, ensure_ascii=False)}")

print("\n" + "="*70)
print("GT 格式检查")
print("="*70)

# 检查前 3 个 GT
count = 0
for img_path, gts in gt_map.items():
    if count >= 3:
        break
    print(f"\n【GT: {img_path}】")
    print(f"  目标数: {len(gts)}")
    if gts:
        print(f"  第一个 GT:")
        print(f"    {json.dumps(gts[0], indent=4, ensure_ascii=False)}")
    count += 1

print("\n" + "="*70)
print("坐标范围检查")
print("="*70)

# 检查坐标范围
print("\n预测坐标范围:")
all_pred_bboxes = []
for pred in predictions:
    for p in pred.get('predictions', []):
        bbox = p.get('bbox_2d', [])
        if bbox:
            all_pred_bboxes.extend(bbox)

if all_pred_bboxes:
    print(f"  Min: {min(all_pred_bboxes):.2f}")
    print(f"  Max: {max(all_pred_bboxes):.2f}")
    print(f"  Mean: {sum(all_pred_bboxes)/len(all_pred_bboxes):.2f}")
else:
    print("  无坐标数据")

print("\nGT 坐标范围:")
all_gt_bboxes = []
for gts in gt_map.values():
    for gt in gts:
        bbox = gt.get('bbox_2d', [])
        if bbox:
            all_gt_bboxes.extend(bbox)

if all_gt_bboxes:
    print(f"  Min: {min(all_gt_bboxes):.2f}")
    print(f"  Max: {max(all_gt_bboxes):.2f}")
    print(f"  Mean: {sum(all_gt_bboxes)/len(all_gt_bboxes):.2f}")
else:
    print("  无坐标数据")

print("\n标签检查:")
all_pred_labels = set()
for pred in predictions:
    for p in pred.get('predictions', []):
        label = p.get('label')
        if label:
            all_pred_labels.add(label)

all_gt_labels = set()
for gts in gt_map.values():
    for gt in gts:
        label = gt.get('label')
        if label:
            all_gt_labels.add(label)

print(f"  预测标签: {all_pred_labels}")
print(f"  GT 标签: {all_gt_labels}")
print(f"  共同标签: {all_pred_labels & all_gt_labels}")