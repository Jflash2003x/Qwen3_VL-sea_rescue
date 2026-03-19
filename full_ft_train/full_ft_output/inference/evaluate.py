"""
==========================================================================
评估脚本：计算模型性能指标
==========================================================================

功能：
  1. 读取推理结果和 ground truth
  2. 匹配预测和标注（IoU@0.5）
  3. 计算 Precision、Recall、F1、mAP
  4. 逐类别和逐图片统计
  5. 多 IoU 阈值分析（0.1-0.9）

输出：
  evaluate_output/
  ├── original_metrics_overall.json
  ├── original_metrics_by_class.json
  ├── original_metrics_by_image.json
  ├── finetuned_metrics_overall.json
  ├── finetuned_metrics_by_class.json
  ├── finetuned_metrics_by_image.json
  └── comparison_report.txt
"""

import os
import json
from collections import defaultdict
from tqdm import tqdm

# 标签映射
LABEL_MAP = {
    "boat": "船只",
    "person": "水中人员",
    "人": "水中人员",
    "浮标": "浮标",
    "船只": "船只",
    "水中人员": "水中人员",
}

CLASSES = ["船只", "水中人员", "浮标"]


def normalize_label(label):
    """标签归一化"""
    return LABEL_MAP.get(label, label)


def compute_iou(box1, box2):
    """计算两个框的 IoU
    box: [x1, y1, x2, y2]
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # 计算交集
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
        return 0.0
    
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    
    # 计算并集
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def match_predictions_to_gt(predictions, ground_truth, iou_threshold=0.5):
    """匹配预测到 GT，返回 TP、FP、FN"""
    
    # 标签映射
    predictions_normalized = [
        {
            "bbox_2d": p["bbox_2d"],
            "label": normalize_label(p["label"]),
        }
        for p in predictions
    ]
    
    gt_normalized = [
        {
            "bbox_2d": gt["bbox_2d"],
            "label": normalize_label(gt["label"]),
        }
        for gt in ground_truth
    ]
    
    matched_gt = set()
    tp_list = []
    fp_list = []
    
    # 对每个预测，找最高 IoU 的 GT
    for pred in predictions_normalized:
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(gt_normalized):
            if gt_idx in matched_gt:
                continue
            
            if pred["label"] != gt["label"]:
                continue
            
            iou = compute_iou(pred["bbox_2d"], gt["bbox_2d"])
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp_list.append({
                "pred": pred,
                "gt": gt_normalized[best_gt_idx],
                "iou": best_iou,
            })
            matched_gt.add(best_gt_idx)
        else:
            fp_list.append(pred)
    
    # 未匹配的 GT 是 FN
    fn_list = [gt_normalized[i] for i in range(len(gt_normalized)) if i not in matched_gt]
    
    return tp_list, fp_list, fn_list


def calculate_metrics(tp, fp, fn):
    """计算 Precision、Recall、F1"""
    precision = len(tp) / (len(tp) + len(fp)) if (len(tp) + len(fp)) > 0 else 0
    recall = len(tp) / (len(tp) + len(fn)) if (len(tp) + len(fn)) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "tp": len(tp),
        "fp": len(fp),
        "fn": len(fn),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def evaluate_model(predictions_file, gt_json_path, output_dir, model_name):
    """评估单个模型"""
    print(f"\n{'='*70}")
    print(f"  评估 {model_name}")
    print(f"{'='*70}")
    
    # 读取预测
    if not os.path.exists(predictions_file):
        print(f"❌ 预测文件不存在: {predictions_file}")
        return None, None, None
    
    with open(predictions_file, 'r', encoding='utf-8') as f:
        predictions_data = json.load(f)
    
    # 读取 GT
    with open(gt_json_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    
    # 创建 GT 映射
    gt_map = {}
    for sample in gt_data:
        img_path = sample["messages"][0]["content"][0]["image"]
        gt_text = sample["messages"][1]["content"]
        try:
            gt_list = json.loads(gt_text)
            gt_map[img_path] = gt_list
        except json.JSONDecodeError:
            gt_map[img_path] = []
    
    # 逐图片评估
    print("\n逐图片匹配...")
    metrics_by_image = []
    metrics_by_class = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    
    for pred_result in tqdm(predictions_data, desc="处理中"):
        img_path = pred_result["image_path"]
        predictions = pred_result["predictions"]
        
        gt_list = gt_map.get(img_path, [])
        
        # 多 IoU 阈值下的评估
        iou_metrics = {}
        for iou_th in [0.1, 0.3, 0.5, 0.7, 0.9]:
            tp, fp, fn = match_predictions_to_gt(predictions, gt_list, iou_th)
            metrics = calculate_metrics(tp, fp, fn)
            iou_metrics[f"iou_{iou_th}"] = metrics
        
        # IoU@0.5 作为主要指标
        tp, fp, fn = match_predictions_to_gt(predictions, gt_list, 0.5)
        main_metrics = calculate_metrics(tp, fp, fn)
        
        # 逐类别统计
        for class_name in CLASSES:
            class_preds = [p for p in predictions if normalize_label(p["label"]) == class_name]
            class_gt = [g for g in gt_list if normalize_label(g["label"]) == class_name]
            class_tp, class_fp, class_fn = match_predictions_to_gt(class_preds, class_gt, 0.5)
            
            metrics_by_class[class_name]["tp"] += len(class_tp)
            metrics_by_class[class_name]["fp"] += len(class_fp)
            metrics_by_class[class_name]["fn"] += len(class_fn)
        
        metrics_by_image.append({
            "image_path": img_path,
            "metrics_iou05": main_metrics,
            "metrics_multi_iou": iou_metrics,
            "num_predictions": len(predictions),
            "num_gt": len(gt_list),
        })
    
    # 计算全局指标
    overall_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for img_metrics in metrics_by_image:
        m = img_metrics["metrics_iou05"]
        overall_metrics["overall"]["tp"] += m["tp"]
        overall_metrics["overall"]["fp"] += m["fp"]
        overall_metrics["overall"]["fn"] += m["fn"]
    
    overall_result = calculate_metrics(
        [None] * overall_metrics["overall"]["tp"],
        [None] * overall_metrics["overall"]["fp"],
        [None] * overall_metrics["overall"]["fn"],
    )
    
    # 计算逐类别指标
    by_class_result = {}
    for class_name in CLASSES:
        stats = metrics_by_class[class_name]
        by_class_result[class_name] = calculate_metrics(
            [None] * stats["tp"],
            [None] * stats["fp"],
            [None] * stats["fn"],
        )
    
    # 保存结果
    metrics_overall_file = os.path.join(output_dir, f"{model_name}_metrics_overall.json")
    with open(metrics_overall_file, 'w', encoding='utf-8') as f:
        json.dump(overall_result, f, ensure_ascii=False, indent=2)
    
    metrics_by_class_file = os.path.join(output_dir, f"{model_name}_metrics_by_class.json")
    with open(metrics_by_class_file, 'w', encoding='utf-8') as f:
        json.dump(by_class_result, f, ensure_ascii=False, indent=2)
    
    metrics_by_image_file = os.path.join(output_dir, f"{model_name}_metrics_by_image.json")
    with open(metrics_by_image_file, 'w', encoding='utf-8') as f:
        json.dump(metrics_by_image, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 评估完成")
    print(f"  总体 Precision: {overall_result['precision']}")
    print(f"  总体 Recall: {overall_result['recall']}")
    print(f"  总体 F1: {overall_result['f1']}")
    
    print(f"\n【逐类别指标 (IoU@0.5)】")
    for class_name in CLASSES:
        m = by_class_result[class_name]
        print(f"  {class_name}: Precision={m['precision']}, Recall={m['recall']}, F1={m['f1']}")
    
    return overall_result, by_class_result, metrics_by_image


def main():
    print("=" * 70)
    print("  评估脚本")
    print("=" * 70)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "evaluate_output")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n输出目录: {output_dir}")
    
    inference_output_dir = os.path.join(script_dir, "inference_output")
    gt_json = "/root/autodl-tmp/qwen_vl/finetune_data_v2/test_split.json"
    
    # 评估原始模型
    original_pred_file = os.path.join(inference_output_dir, "original_predictions.json")
    original_overall, original_by_class, original_by_image = evaluate_model(
        original_pred_file, gt_json, output_dir, "original"
    )
    
    if original_overall is None:
        print("❌ 原始模型评估失败")
        return
    
    # 评估微调模型
    finetuned_pred_file = os.path.join(inference_output_dir, "finetuned_predictions.json")
    finetuned_overall, finetuned_by_class, finetuned_by_image = evaluate_model(
        finetuned_pred_file, gt_json, output_dir, "finetuned"
    )
    
    if finetuned_overall is None:
        print("❌ 微调模型评估失败")
        return
    
    # 对比
    print(f"\n{'='*70}")
    print(f"  性能对比")
    print(f"{'='*70}")
    
    print(f"\n【总体性能】")
    print(f"  {'指标':<15} {'原始模型':<15} {'微调模型':<15} {'改进':<15}")
    print(f"  {'-'*60}")
    
    comparison_report = f"{'='*70}\n"
    comparison_report += f"  性能对比报告\n"
    comparison_report += f"{'='*70}\n\n"
    
    comparison_report += f"【总体性能 (IoU@0.5)】\n"
    comparison_report += f"  {'指标':<15} {'原始模型':<15} {'微调模型':<15} {'改进':<15}\n"
    comparison_report += f"  {'-'*60}\n"
    
    for metric in ["precision", "recall", "f1"]:
        orig_val = original_overall[metric]
        fine_val = finetuned_overall[metric]
        improve = (fine_val - orig_val) * 100
        improve_str = f"{improve:+.2f}%" if improve != 0 else "0%"
        print(f"  {metric:<15} {orig_val:<15.4f} {fine_val:<15.4f} {improve_str:<15}")
        comparison_report += f"  {metric:<15} {orig_val:<15.4f} {fine_val:<15.4f} {improve_str:<15}\n"
    
    print(f"\n【逐类别性能】")
    comparison_report += f"\n【逐类别性能 (IoU@0.5)】\n"
    
    for class_name in CLASSES:
        print(f"\n  {class_name}:")
        comparison_report += f"\n  {class_name}:\n"
        
        orig = original_by_class[class_name]
        fine = finetuned_by_class[class_name]
        
        for metric in ["precision", "recall", "f1"]:
            orig_val = orig[metric]
            fine_val = fine[metric]
            improve = (fine_val - orig_val) * 100
            improve_str = f"{improve:+.2f}%" if improve != 0 else "0%"
            print(f"    {metric:<12} {orig_val:<12.4f} {fine_val:<12.4f} {improve_str}")
            comparison_report += f"    {metric:<12} {orig_val:<12.4f} {fine_val:<12.4f} {improve_str}\n"
    
    # 【水中人员专项分析】
    print(f"\n【水中人员专项分析】")
    comparison_report += f"\n【水中人员专项分析】\n"
    
    person_class_name = "水中人员"
    orig_person = original_by_class[person_class_name]
    fine_person = finetuned_by_class[person_class_name]
    
    print(f"\n  {person_class_name}:")
    comparison_report += f"\n  {person_class_name}:\n"
    
    for metric in ["precision", "recall", "f1"]:
        orig_val = orig_person[metric]
        fine_val = fine_person[metric]
        improve = (fine_val - orig_val) * 100
        improve_str = f"{improve:+.2f}%" if improve != 0 else "0%"
        status = "✅" if improve >= 0 else "⚠️"
        print(f"    {status} {metric:<12} {orig_val:<12.4f} {fine_val:<12.4f} {improve_str}")
        comparison_report += f"    {status} {metric:<12} {orig_val:<12.4f} {fine_val:<12.4f} {improve_str}\n"
    
    # 保存对比报告
    report_file = os.path.join(output_dir, "comparison_report.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(comparison_report)
    
    print(f"\n{'='*70}")
    print(f"✅ 对比报告已保存: {report_file}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()