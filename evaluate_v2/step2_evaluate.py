"""
==========================================================================
Step 2: 评估指标计算（多阈值 + 分类别 + mAP）
==========================================================================

输入:
  output/ground_truth.json
  output/inference_base.json
  output/inference_v2.json

输出:
  output/eval_results.json

所有坐标均为归一化 0-1000，直接比较，不做任何坐标转换。
==========================================================================
"""

import json
import os
from collections import defaultdict

# ============================================================
# 配置
# ============================================================
OUTPUT_DIR = '/root/autodl-tmp/qwen_vl/evaluate_v2/output'

GT_PATH = os.path.join(OUTPUT_DIR, 'ground_truth.json')
BASE_PATH = os.path.join(OUTPUT_DIR, 'inference_base.json')
V2_PATH = os.path.join(OUTPUT_DIR, 'inference_v2.json')
RESULT_PATH = os.path.join(OUTPUT_DIR, 'eval_results.json')

ALL_CATEGORIES = ['水中人员', '船只', '水上摩托', '救生设备', '浮标']
IOU_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8]


# ============================================================
# IoU 计算
# ============================================================

def calc_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def match_at_iou(gts, preds, iou_th):
    """贪心匹配"""
    matched_gt = set()
    tp, fp = 0, 0
    ious = []
    cls_correct = 0

    for pred in preds:
        best_iou, best_idx = 0, -1
        for i, gt in enumerate(gts):
            if i in matched_gt:
                continue
            iou = calc_iou(pred['bbox'], gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        if best_iou >= iou_th and best_idx >= 0:
            tp += 1
            matched_gt.add(best_idx)
            ious.append(best_iou)
            if pred['label'] == gts[best_idx]['label']:
                cls_correct += 1
        else:
            fp += 1

    fn = len(gts) - len(matched_gt)
    return tp, fp, fn, ious, cls_correct


# ============================================================
# 评估单个模型
# ============================================================

def evaluate_model(predictions, gt_data, model_name):
    results = {}

    for iou_th in IOU_THRESHOLDS:
        total_tp, total_fp, total_fn = 0, 0, 0
        all_ious = []
        total_cls_correct = 0
        cat_tp = defaultdict(int)
        cat_gt = defaultdict(int)
        cat_fp = defaultdict(int)

        # 每张图的 precision/recall（用于计算 per-image 指标）
        per_image_p = []
        per_image_r = []

        for item in predictions:
            img_name = item['image_name']
            gts = gt_data.get(img_name, [])
            preds = item['preds']

            tp, fp, fn, ious, cls_c = match_at_iou(gts, preds, iou_th)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            all_ious.extend(ious)
            total_cls_correct += cls_c

            # per-image
            img_p = tp / max(tp + fp, 1)
            img_r = tp / max(tp + fn, 1)
            per_image_p.append(img_p)
            per_image_r.append(img_r)

            # 按类别统计 GT 总数
            for gt in gts:
                cat_gt[gt['label']] += 1

            # 按类别统计 TP（带匹配）
            matched_set = set()
            for pred in preds:
                best_iou, best_idx = 0, -1
                for i, gt in enumerate(gts):
                    if i in matched_set:
                        continue
                    iou = calc_iou(pred['bbox'], gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = i

                if best_iou >= iou_th and best_idx >= 0:
                    cat_tp[gts[best_idx]['label']] += 1
                    matched_set.add(best_idx)
                else:
                    cat_fp[pred['label']] += 1

        # 全局指标
        p = total_tp / max(total_tp + total_fp, 1)
        r = total_tp / max(total_tp + total_fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-8)
        avg_iou = sum(all_ious) / max(len(all_ious), 1)
        cls_acc = total_cls_correct / max(total_tp, 1)

        # per-image 平均
        mean_img_p = sum(per_image_p) / max(len(per_image_p), 1)
        mean_img_r = sum(per_image_r) / max(len(per_image_r), 1)

        # 按类别
        cat_metrics = {}
        for cat in ALL_CATEGORIES:
            ct = cat_tp.get(cat, 0)
            cg = cat_gt.get(cat, 0)
            cf = cat_fp.get(cat, 0)
            cat_r = ct / max(cg, 1)
            cat_p = ct / max(ct + cf, 1)
            cat_f1 = 2 * cat_p * cat_r / max(cat_p + cat_r, 1e-8)
            cat_metrics[cat] = {
                'tp': ct, 'gt': cg, 'fp': cf,
                'precision': round(cat_p, 4),
                'recall': round(cat_r, 4),
                'f1': round(cat_f1, 4),
            }

        # mAP-like: 各类别 recall 的平均
        recalls = [cat_metrics[c]['recall'] for c in ALL_CATEGORIES if cat_gt.get(c, 0) > 0]
        mean_recall = sum(recalls) / max(len(recalls), 1)

        results[f"{iou_th:.2f}"] = {
            'tp': total_tp, 'fp': total_fp, 'fn': total_fn,
            'precision': round(p, 4),
            'recall': round(r, 4),
            'f1': round(f1, 4),
            'avg_iou': round(avg_iou, 4),
            'cls_accuracy': round(cls_acc, 4),
            'mean_img_precision': round(mean_img_p, 4),
            'mean_img_recall': round(mean_img_r, 4),
            'mean_category_recall': round(mean_recall, 4),
            'category': cat_metrics,
        }

    # 速度
    times = [item.get('time', 0) for item in predictions if item.get('time', 0) > 0]
    speed = {
        'avg_per_image': round(sum(times) / max(len(times), 1), 3),
        'total': round(sum(times), 2),
        'throughput': round(len(predictions) / max(sum(times), 0.01), 2),
    }

    return results, speed


# ============================================================
# 打印对比
# ============================================================

def print_comparison(base_r, v2_r, base_s, v2_s, iou_th):
    b = base_r.get(iou_th, {})
    v = v2_r.get(iou_th, {})

    print(f"\n  {'指标':<14} {'原始模型':>10} {'V2 微调':>10} {'提升':>10}")
    print(f"  {'─' * 44}")

    for name, key in [('精确率','precision'), ('召回率','recall'), ('F1','f1'),
                       ('平均IoU','avg_iou'), ('分类准确率','cls_accuracy'),
                       ('类别平均recall','mean_category_recall')]:
        bv = b.get(key, 0)
        vv = v.get(key, 0)
        print(f"  {name:<14} {bv*100:>8.1f}% {vv*100:>8.1f}% {(vv-bv)*100:>+8.1f}%")

    print(f"  {'TP/FP/FN':<14} {b.get('tp',0):>3}/{b.get('fp',0):>3}/{b.get('fn',0):>3}  "
          f"   {v.get('tp',0):>3}/{v.get('fp',0):>3}/{v.get('fn',0):>3}")
    print(f"  {'推理速度':<14} {base_s['throughput']:>6.1f}张/s {v2_s['throughput']:>6.1f}张/s")

    print(f"\n  按类别 (IoU@{iou_th}):")
    print(f"  {'类别':<10} {'GT':>4} {'原始P':>7} {'原始R':>7} {'V2_P':>7} {'V2_R':>7} {'R提升':>7}")
    print(f"  {'─' * 52}")
    for cat in ALL_CATEGORIES:
        bc = b.get('category', {}).get(cat, {})
        vc = v.get('category', {}).get(cat, {})
        bg = bc.get('gt', 0)
        if bg == 0 and vc.get('gt', 0) == 0:
            continue
        bp = bc.get('precision', 0)
        br = bc.get('recall', 0)
        vp = vc.get('precision', 0)
        vr = vc.get('recall', 0)
        print(f"  {cat:<10} {bg:>4} {bp*100:>6.1f}% {br*100:>6.1f}% {vp*100:>6.1f}% {vr*100:>6.1f}% {(vr-br)*100:>+6.1f}%")


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 60)
    print("  Step 2: 评估指标计算")
    print("=" * 60)

    print("\n  读取数据...")
    with open(GT_PATH, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    with open(BASE_PATH, 'r', encoding='utf-8') as f:
        base_preds = json.load(f)
    with open(V2_PATH, 'r', encoding='utf-8') as f:
        v2_preds = json.load(f)

    print(f"  GT 图片数: {len(gt_data)}")
    total_gt = sum(len(v) for v in gt_data.values())
    print(f"  GT 目标总数: {total_gt}")
    print(f"  原始模型预测: {len(base_preds)} 张")
    print(f"  V2 模型预测: {len(v2_preds)} 张")

    # 坐标系验证
    print("\n  坐标系验证:")
    for item in base_preds[:2]:
        img = item['image_name']
        gts = gt_data.get(img, [])
        preds = item['preds']
        if gts and preds:
            iou = calc_iou(gts[0]['bbox'], preds[0]['bbox'])
            print(f"    {img}: GT[0]={gts[0]['bbox']} Pred[0]={preds[0]['bbox']} IoU={iou:.4f}")
            if iou < 0.01:
                print(f"    ⚠️ IoU 极低！可能坐标系不匹配！")

    # 计算
    print("\n  计算指标...")
    base_results, base_speed = evaluate_model(base_preds, gt_data, "原始模型")
    v2_results, v2_speed = evaluate_model(v2_preds, gt_data, "V2 微调")

    # 打印多个阈值
    for th in ["0.30", "0.50", "0.70"]:
        print(f"\n{'='*60}")
        print(f"  IoU 阈值 = {th}")
        print(f"{'='*60}")
        print_comparison(base_results, v2_results, base_speed, v2_speed, th)

    # 保存
    eval_output = {
        'config': {
            'num_test_images': len(base_preds),
            'total_gt_targets': total_gt,
            'iou_thresholds': IOU_THRESHOLDS,
            'coordinate_system': 'normalized_0_1000',
            'note': 'GT来自convert.py的bbox_to_normalized, 模型输出也是0-1000, 不做任何转换',
        },
        'base_model': {'results': base_results, 'speed': base_speed},
        'v2_model': {'results': v2_results, 'speed': v2_speed},
    }

    with open(RESULT_PATH, 'w', encoding='utf-8') as f:
        json.dump(eval_output, f, ensure_ascii=False, indent=2)

    print(f"\n\n  ✅ 结果保存到: {RESULT_PATH}")
    print(f"  运行 step3_visualize.py 生成图表。")


if __name__ == "__main__":
    main()