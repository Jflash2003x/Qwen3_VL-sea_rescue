"""
==========================================================================
可视化脚本：绘制性能对比图表（CPU 版本 + 中文字体正确处理）
==========================================================================

关键：
  1. 必须在导入 pyplot 前设置 matplotlib.use('Agg')
  2. 用 fc-list 找系统中文字体
  3. 用 fm.fontManager.addfont() 加载字体
  4. 用 fontproperties= 参数传递到所有文本函数
"""

import os
import json
import cv2
import numpy as np
from PIL import Image
import subprocess

# ===== 必须在导入 pyplot 前设置 =====
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as patches

# ============================================================
# 字体配置（关键！）
# ============================================================

FONT_PROP = None

def setup_font():
    """加载中文字体"""
    global FONT_PROP
    
    print("  正在加载中文字体...")
    
    # 方案 1: 尝试常见的系统字体路径
    font_candidates = [
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    
    for font_path in font_candidates:
        if os.path.exists(font_path):
            try:
                fm.fontManager.addfont(font_path)
                FONT_PROP = fm.FontProperties(fname=font_path)
                print(f"    ✅ 已加载: {font_path}")
                return
            except Exception as e:
                print(f"    ⚠️ 加载失败: {e}")
                continue
    
    # 方案 2: 用 fc-list 查找系统中文字体
    try:
        result = subprocess.run(
            ['fc-list', ':lang=zh', '-f', '%{file}\n'],
            capture_output=True, text=True, timeout=5
        )
        lines = [l.strip() for l in result.stdout.strip().split('\n') if l.strip()]
        if lines:
            for font_path in lines:
                if os.path.exists(font_path):
                    try:
                        fm.fontManager.addfont(font_path)
                        FONT_PROP = fm.FontProperties(fname=font_path)
                        print(f"    ✅ 已加载: {font_path}")
                        return
                    except:
                        continue
    except Exception as e:
        print(f"    ⚠️ fc-list 查询失败: {e}")
    
    # 方案 3: 降级处理 - 不使用中文，用数字/符号替代
    print("    ⚠️ 未找到中文字体，将使用英文标签")
    try:
        FONT_PROP = fm.FontProperties()
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
    except:
        pass


def fkw():
    """返回字体参数，方便在 matplotlib 函数中使用"""
    return {'fontproperties': FONT_PROP} if FONT_PROP else {}


# ============================================================
# 工具函数
# ============================================================

def load_json(filepath):
    """加载 JSON 文件"""
    if not os.path.exists(filepath):
        print(f"❌ 文件不存在: {filepath}")
        return None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def calc_iou(box1, box2):
    """计算 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    a2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    return inter / max(a1 + a2 - inter, 1e-8)


def count_tp(gts, preds, iou_th=0.5):
    """计算 TP 数量"""
    matched = set()
    tp = 0
    for pred in preds:
        best_iou, best_idx = 0, -1
        for i, gt in enumerate(gts):
            if i in matched:
                continue
            iou = calc_iou(pred['bbox'], gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        if best_iou >= iou_th and best_idx >= 0:
            tp += 1
            matched.add(best_idx)
    return tp


# ============================================================
# 绘图函数
# ============================================================

def plot_overall_comparison(original_metrics, finetuned_metrics, output_dir):
    """绘制全局指标对比柱状图"""
    metrics = ["precision", "recall", "f1"]
    metric_labels_en = ["Precision", "Recall", "F1"]
    metric_labels_ch = ["精确率", "召回率", "F1分数"]
    
    # 选择标签（如果有中文字体就用中文，否则用英文）
    labels = metric_labels_ch if FONT_PROP else metric_labels_en
    
    orig_vals = [original_metrics.get(m, 0) for m in metrics]
    fine_vals = [finetuned_metrics.get(m, 0) for m in metrics]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, orig_vals, width, label='Original Model', color='#1f77b4', edgecolor='white')
    bars2 = ax.bar(x + width/2, fine_vals, width, label='Finetuned Model', color='#ff7f0e', edgecolor='white')
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.4f}',
               ha='center', va='bottom', fontsize=10, **fkw())
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.4f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold', **fkw())
    
    ax.set_ylabel('Score', fontsize=12, **fkw())
    title_text = "Global Metrics Comparison (IoU@0.5)" if not FONT_PROP else "全局指标对比 (IoU@0.5)"
    ax.set_title(title_text, fontsize=14, fontweight='bold', **fkw())
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, **fkw())
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, "overall_metrics_comparison.png")
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ {os.path.basename(output_file)}")


def plot_by_class_metrics(original_by_class, finetuned_by_class, output_dir):
    """绘制按类别的指标对比"""
    classes = list(original_by_class.keys())
    if not classes:
        return
    
    # Precision
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, metric in enumerate(['precision', 'recall', 'f1']):
        ax = axes[idx]
        orig_vals = [original_by_class[c].get(metric, 0) for c in classes]
        fine_vals = [finetuned_by_class[c].get(metric, 0) for c in classes]
        
        x = np.arange(len(classes))
        width = 0.35
        
        ax.bar(x - width/2, orig_vals, width, label='Original', color='#1f77b4', edgecolor='white')
        ax.bar(x + width/2, fine_vals, width, label='Finetuned', color='#ff7f0e', edgecolor='white')
        
        ax.set_ylabel('Score', fontsize=11, **fkw())
        metric_names = {'precision': 'Precision', 'recall': 'Recall', 'f1': 'F1'}
        ax.set_title(metric_names[metric], fontsize=12, fontweight='bold', **fkw())
        ax.set_xticks(x)
        ax.set_xticklabels(classes, fontsize=10, **fkw(), rotation=45, ha='right')
        ax.set_ylim([0, 1.1])
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle("Category-wise Metrics (IoU@0.5)", fontsize=14, fontweight='bold', **fkw(), y=1.00)
    plt.tight_layout()
    output_file = os.path.join(output_dir, "by_class_metrics_comparison.png")
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ {os.path.basename(output_file)}")


def plot_f1_vs_iou(original_by_image, finetuned_by_image, output_dir):
    """绘制 F1 vs IoU 曲线"""
    
    iou_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    orig_f1_by_iou = {str(iou): 0 for iou in iou_thresholds}
    fine_f1_by_iou = {str(iou): 0 for iou in iou_thresholds}
    
    count_orig = {str(iou): 0 for iou in iou_thresholds}
    count_fine = {str(iou): 0 for iou in iou_thresholds}
    
    for img_metric in original_by_image:
        for iou in iou_thresholds:
            key = f"iou_{iou}"
            if key in img_metric.get("metrics_multi_iou", {}):
                f1 = img_metric["metrics_multi_iou"][key].get("f1", 0)
                orig_f1_by_iou[str(iou)] += f1
                count_orig[str(iou)] += 1
    
    for img_metric in finetuned_by_image:
        for iou in iou_thresholds:
            key = f"iou_{iou}"
            if key in img_metric.get("metrics_multi_iou", {}):
                f1 = img_metric["metrics_multi_iou"][key].get("f1", 0)
                fine_f1_by_iou[str(iou)] += f1
                count_fine[str(iou)] += 1
    
    orig_curve = [orig_f1_by_iou[str(iou)] / max(count_orig[str(iou)], 1) for iou in iou_thresholds]
    fine_curve = [fine_f1_by_iou[str(iou)] / max(count_fine[str(iou)], 1) for iou in iou_thresholds]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(iou_thresholds, orig_curve, 'o-', linewidth=2.5, markersize=8, label='Original Model', color='#1f77b4')
    ax.plot(iou_thresholds, fine_curve, 's-', linewidth=2.5, markersize=8, label='Finetuned Model', color='#ff7f0e')
    
    # 标注关键点
    idx_05 = iou_thresholds.index(0.5)
    ax.scatter([0.5], [orig_curve[idx_05]], s=150, color='#1f77b4', zorder=5, edgecolors='white', linewidth=2)
    ax.scatter([0.5], [fine_curve[idx_05]], s=150, color='#ff7f0e', zorder=5, edgecolors='white', linewidth=2)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('IoU Threshold', fontsize=12, **fkw())
    ax.set_ylabel('F1 Score', fontsize=12, **fkw())
    ax.set_title('F1 Score vs IoU Threshold', fontsize=14, fontweight='bold', **fkw())
    ax.legend(fontsize=11, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    ax.set_xticks(iou_thresholds)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, "f1_vs_iou_curve.png")
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ {os.path.basename(output_file)}")

def plot_sample_comparisons(original_predictions, finetuned_predictions, gt_map, data_base, output_dir, num_samples=8):
    """绘制样本检测对比图"""
    
    print(f"\n  生成样本对比图 ({num_samples} 张):")
    
    if not original_predictions or not finetuned_predictions:
        print("    ⚠️ 无可用的预测数据")
        return
    
    # 简单采样：均匀选择
    total = len(original_predictions)
    step = max(1, total // num_samples)
    selected_indices = list(range(0, total, step))[:num_samples]
    
    # ===== 关键：matplotlib 颜色需要是 0-1 范围的 RGB =====
    colors = {
        'gt': '#FF0000',        # 红色
        'original': '#00FF00',  # 绿色
        'finetuned': '#0000FF', # 蓝色
    }
    
    for plot_idx, sample_idx in enumerate(selected_indices):
        if sample_idx >= len(original_predictions):
            break
        
        orig_pred = original_predictions[sample_idx]
        fine_pred = finetuned_predictions[sample_idx]
        img_path = orig_pred.get('image_path', '')
        
        image_full_path = os.path.join(data_base, img_path)
        
        if not os.path.exists(image_full_path):
            continue
        
        try:
            img = Image.open(image_full_path).convert('RGB')
        except:
            continue
        
        img_w, img_h = img.size
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 获取 GT、原始和微调的框
        gt_list = gt_map.get(img_path, [])
        orig_preds = orig_pred.get('predictions', [])
        fine_preds = fine_pred.get('predictions', [])
        
        def draw(ax, boxes, title, color):
            ax.imshow(img)
            ax.set_title(title, fontsize=11, fontweight='bold', **fkw())
            ax.axis('off')
            
            for box in boxes:
                bbox = box['bbox_2d']
                x1 = bbox[0] / 1000 * img_w
                y1 = bbox[1] / 1000 * img_h
                x2 = bbox[2] / 1000 * img_w
                y2 = bbox[3] / 1000 * img_h
                
                # ===== 使用 hex 颜色字符串 =====
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=2, edgecolor=color, facecolor='none'
                )
                ax.add_patch(rect)
        
        draw(axes[0], gt_list, f'Ground Truth ({len(gt_list)} targets)', colors['gt'])
        draw(axes[1], orig_preds, f'Original ({len(orig_preds)} preds)', colors['original'])
        draw(axes[2], fine_preds, f'Finetuned ({len(fine_preds)} preds)', colors['finetuned'])
        
        plt.suptitle(f'Sample {plot_idx+1}/{num_samples}: {os.path.basename(img_path)}', 
                    fontsize=12, fontweight='bold', **fkw())
        plt.tight_layout()
        
        output_file = os.path.join(output_dir, f"sample_comparison_{plot_idx+1:02d}.png")
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"    ✅ sample_comparison_{plot_idx+1:02d}.png")


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 70)
    print("  可视化脚本（CPU 模式 + 中文字体正确处理）")
    print("=" * 70)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "visualize_output")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n输出目录: {output_dir}\n")
    
    # 设置字体
    setup_font()
    
    # 加载数据
    print("\n加载数据...")
    evaluate_output_dir = os.path.join(script_dir, "evaluate_output")
    inference_output_dir = os.path.join(script_dir, "inference_output")
    
    original_overall = load_json(os.path.join(evaluate_output_dir, "original_metrics_overall.json"))
    finetuned_overall = load_json(os.path.join(evaluate_output_dir, "finetuned_metrics_overall.json"))
    original_by_class = load_json(os.path.join(evaluate_output_dir, "original_metrics_by_class.json"))
    finetuned_by_class = load_json(os.path.join(evaluate_output_dir, "finetuned_metrics_by_class.json"))
    original_by_image = load_json(os.path.join(evaluate_output_dir, "original_metrics_by_image.json"))
    finetuned_by_image = load_json(os.path.join(evaluate_output_dir, "finetuned_metrics_by_image.json"))
    original_predictions = load_json(os.path.join(inference_output_dir, "original_predictions.json"))
    finetuned_predictions = load_json(os.path.join(inference_output_dir, "finetuned_predictions.json"))
    
    gt_json = "/root/autodl-tmp/qwen_vl/finetune_data_v2/test_split.json"
    with open(gt_json, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    
    gt_map = {}
    for sample in gt_data:
        img_path = sample["messages"][0]["content"][0]["image"]
        gt_text = sample["messages"][1]["content"]
        try:
            gt_map[img_path] = json.loads(gt_text)
        except:
            gt_map[img_path] = []
    
    data_base = "/root/autodl-tmp/qwen_vl/finetune_data_v2"
    
    if not all([original_overall, finetuned_overall]):
        print("❌ 评估结果加载失败")
        return
    
    print("✅ 数据加载完成\n")
    
    # 生成图表
    print("生成图表:\n")
    plot_overall_comparison(original_overall, finetuned_overall, output_dir)
    plot_by_class_metrics(original_by_class, finetuned_by_class, output_dir)
    plot_f1_vs_iou(original_by_image, finetuned_by_image, output_dir)
    plot_sample_comparisons(original_predictions, finetuned_predictions, gt_map, data_base, output_dir, num_samples=8)
    
    print(f"\n{'='*70}")
    print(f"✅ 所有可视化已生成！")
    print(f"输出目录: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()