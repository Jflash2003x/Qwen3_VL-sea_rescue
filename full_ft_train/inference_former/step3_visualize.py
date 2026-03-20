"""
==========================================================================
Step 3: 生成可视化图表
==========================================================================

输入:
  output/eval_results.json
  output/ground_truth.json
  output/inference_base.json
  output/inference_v2.json
  output/image_list.json

输出:
  output/charts/bar_overall.png          - 总体指标对比柱状图
  output/charts/bar_category_recall.png  - 分类别召回率
  output/charts/bar_category_f1.png      - 分类别 F1
  output/charts/line_f1_vs_iou.png       - F1 随 IoU 变化曲线
  output/charts/bar_improvement.png      - 提升幅度水平条形图
  output/charts/detection_*.png          - 检测对比图（GT/Base/V2）

画图坐标转换：归一化 / 1000 × 图片宽高 = 像素位置
==========================================================================
"""

import json
import os
import subprocess

import random
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as patches

# ============================================================
# 配置
# ============================================================
OUTPUT_DIR = '/root/autodl-tmp/qwen_vl/full_ft_train/inference_former/inference_output'
CHARTS_DIR = os.path.join(OUTPUT_DIR, 'charts')

EVAL_PATH = os.path.join(OUTPUT_DIR, 'eval_results.json')
GT_PATH = os.path.join(OUTPUT_DIR, 'ground_truth.json')
BASE_PATH = os.path.join(OUTPUT_DIR, 'inference_base.json')
V2_PATH = os.path.join(OUTPUT_DIR, 'inference_v2.json')
IMG_LIST_PATH = os.path.join(OUTPUT_DIR, 'image_list.json')

ALL_CATEGORIES = ['水中人员', '船只', '水上摩托', '救生设备', '浮标']
MAIN_IOU = "0.50"

C_BASE = '#5B9BD5'
C_V2 = '#ED7D31'

CAT_COLORS = {
    '水中人员': '#FF4444', '船只': '#4444FF', '水上摩托': '#44BB44',
    '救生设备': '#FF8800', '浮标': '#AA44AA',
}

# ============================================================
# 字体
# ============================================================
FONT_PROP = None

def setup_font():
    global FONT_PROP
    for p in [
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]:
        if os.path.exists(p):
            fm.fontManager.addfont(p)
            fp = fm.FontProperties(fname=p)
            plt.rcParams['font.sans-serif'] = [fp.get_name()]
            plt.rcParams['axes.unicode_minus'] = False
            FONT_PROP = fp
            print(f"  字体: {fp.get_name()}")
            return
    try:
        result = subprocess.run(['fc-list', ':lang=zh', '-f', '%{file}\n'],
                                capture_output=True, text=True, timeout=5)
        lines = [l.strip() for l in result.stdout.strip().split('\n') if l.strip()]
        if lines and os.path.exists(lines[0]):
            fm.fontManager.addfont(lines[0])
            fp = fm.FontProperties(fname=lines[0])
            plt.rcParams['font.sans-serif'] = [fp.get_name()]
            plt.rcParams['axes.unicode_minus'] = False
            FONT_PROP = fp
            print(f"  字体: {fp.get_name()}")
            return
    except:
        pass
    print("  ⚠️ 未找到中文字体")


def fkw():
    return {'fontproperties': FONT_PROP} if FONT_PROP else {}


# ============================================================
# IoU（供检测对比图用）
# ============================================================

def calc_iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    a2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    return inter / max(a1 + a2 - inter, 1e-8)


def count_tp(gts, preds, iou_th=0.5):
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
# 图表
# ============================================================

def plot_overall(b, v, path):
    metrics = ['precision', 'recall', 'f1', 'avg_iou', 'cls_accuracy']
    labels = ['精确率', '召回率', 'F1', '平均IoU', '分类准确率']
    bv = [b.get(m, 0) for m in metrics]
    vv = [v.get(m, 0) for m in metrics]
    x = np.arange(len(labels)); w = 0.35
    fig, ax = plt.subplots(figsize=(13, 6))
    b1 = ax.bar(x - w/2, bv, w, label='原始模型', color=C_BASE, edgecolor='white')
    b2 = ax.bar(x + w/2, vv, w, label='全参数微调', color=C_V2, edgecolor='white')
    for bar in b1:
        ax.annotate(f'{bar.get_height():.3f}', xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points", ha='center', fontsize=10)
    for bar in b2:
        ax.annotate(f'{bar.get_height():.3f}', xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points", ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel('数值', fontsize=12, **fkw())
    ax.set_title(f'总体指标对比 (IoU@{MAIN_IOU})', fontsize=15, fontweight='bold', **fkw())
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11, **fkw())
    ax.legend(prop=FONT_PROP, fontsize=11); ax.set_ylim(0, 1.18); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"  ✅ {os.path.basename(path)}")


def plot_category_recall(b_cat, v_cat, path):
    cats = [c for c in ALL_CATEGORIES if b_cat.get(c, {}).get('gt', 0) > 0 or v_cat.get(c, {}).get('gt', 0) > 0]
    if not cats: return
    bv = [b_cat.get(c, {}).get('recall', 0) for c in cats]
    vv = [v_cat.get(c, {}).get('recall', 0) for c in cats]
    x = np.arange(len(cats)); w = 0.35
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - w/2, bv, w, label='原始模型', color=C_BASE, edgecolor='white')
    ax.bar(x + w/2, vv, w, label='全参数微调', color=C_V2, edgecolor='white')
    for i, (bval, vval) in enumerate(zip(bv, vv)):
        ax.annotate(f'{bval:.2f}', xy=(i-w/2, bval), xytext=(0,3), textcoords="offset points", ha='center', fontsize=9)
        ax.annotate(f'{vval:.2f}', xy=(i+w/2, vval), xytext=(0,3), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')
    ax.set_title(f'按类别召回率 (IoU@{MAIN_IOU})', fontsize=15, fontweight='bold', **fkw())
    ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=11, **fkw())
    ax.legend(prop=FONT_PROP, fontsize=11); ax.set_ylim(0, 1.18); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"  ✅ {os.path.basename(path)}")

def plot_category_precision(b_cat, v_cat, path):
    cats = [c for c in ALL_CATEGORIES if b_cat.get(c, {}).get('gt', 0) > 0 or v_cat.get(c, {}).get('gt', 0) > 0]
    if not cats: return
    bv = [b_cat.get(c, {}).get('precision', 0) for c in cats]
    vv = [v_cat.get(c, {}).get('precision', 0) for c in cats]
    x = np.arange(len(cats)); w = 0.35
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - w/2, bv, w, label='原始模型', color=C_BASE, edgecolor='white')
    ax.bar(x + w/2, vv, w, label='全参数微调', color=C_V2, edgecolor='white')
    for i, (bval, vval) in enumerate(zip(bv, vv)):
        ax.annotate(f'{bval:.2f}', xy=(i-w/2, bval), xytext=(0,3), textcoords="offset points", ha='center', fontsize=9)
        ax.annotate(f'{vval:.2f}', xy=(i+w/2, vval), xytext=(0,3), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')
    ax.set_title(f'按类别精确率 (IoU@{MAIN_IOU})', fontsize=15, fontweight='bold', **fkw())
    ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=11, **fkw())
    ax.legend(prop=FONT_PROP, fontsize=11); ax.set_ylim(0, 1.18); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"  ✅ {os.path.basename(path)}")

def plot_category_f1(b_cat, v_cat, path):
    cats = [c for c in ALL_CATEGORIES if b_cat.get(c, {}).get('gt', 0) > 0 or v_cat.get(c, {}).get('gt', 0) > 0]
    if not cats: return
    bv = [b_cat.get(c, {}).get('f1', 0) for c in cats]
    vv = [v_cat.get(c, {}).get('f1', 0) for c in cats]
    x = np.arange(len(cats)); w = 0.35
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - w/2, bv, w, label='原始模型', color=C_BASE, edgecolor='white')
    ax.bar(x + w/2, vv, w, label='全参数微调', color=C_V2, edgecolor='white')
    for i, (bval, vval) in enumerate(zip(bv, vv)):
        ax.annotate(f'{bval:.2f}', xy=(i-w/2, bval), xytext=(0,3), textcoords="offset points", ha='center', fontsize=9)
        ax.annotate(f'{vval:.2f}', xy=(i+w/2, vval), xytext=(0,3), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')
    ax.set_title(f'按类别 F1 (IoU@{MAIN_IOU})', fontsize=15, fontweight='bold', **fkw())
    ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=11, **fkw())
    ax.legend(prop=FONT_PROP, fontsize=11); ax.set_ylim(0, 1.18); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"  ✅ {os.path.basename(path)}")


def plot_f1_curve(base_results, v2_results, path):
    ious = sorted([float(k) for k in v2_results.keys()])
    bf1 = [base_results.get(f"{t:.2f}", {}).get('f1', 0) for t in ious]
    vf1 = [v2_results.get(f"{t:.2f}", {}).get('f1', 0) for t in ious]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(ious, bf1, 'o-', color=C_BASE, linewidth=2.5, markersize=7, label='原始模型')
    ax.plot(ious, vf1, 's-', color=C_V2, linewidth=2.5, markersize=7, label='全参数微调')
    ax.fill_between(ious, bf1, vf1, alpha=0.12, color=C_V2)
    for t, b, v in zip(ious, bf1, vf1):
        if t in [0.3, 0.5, 0.7]:
            ax.annotate(f'{b:.2f}', xy=(t, b), xytext=(-15, -18), textcoords="offset points", fontsize=8, color=C_BASE)
            ax.annotate(f'{v:.2f}', xy=(t, v), xytext=(5, 8), textcoords="offset points", fontsize=8, color=C_V2, fontweight='bold')
    ax.set_xlabel('IoU 阈值', fontsize=12, **fkw()); ax.set_ylabel('F1', fontsize=12, **fkw())
    ax.set_title('F1 随 IoU 阈值变化', fontsize=15, fontweight='bold', **fkw())
    ax.legend(prop=FONT_PROP, fontsize=11); ax.grid(alpha=0.3); ax.set_ylim(0, 1.05)
    plt.tight_layout(); plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"  ✅ {os.path.basename(path)}")


def plot_improvement(b, v, path):
    metrics = ['precision', 'recall', 'f1', 'avg_iou', 'cls_accuracy', 'mean_category_recall']
    labels = ['精确率', '召回率', 'F1', '平均IoU', '分类准确率', '类别平均R']
    diffs = [(v.get(m, 0) - b.get(m, 0)) * 100 for m in metrics]
    colors = ['#27AE60' if d >= 0 else '#E74C3C' for d in diffs]
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(labels, diffs, color=colors, height=0.55, edgecolor='white')
    for bar, d in zip(bars, diffs):
        offset = 1.5 if d >= 0 else -1.5
        ax.text(bar.get_width() + offset, bar.get_y() + bar.get_height()/2,
                f'{d:+.1f}%', va='center', fontsize=12, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_xlabel('提升幅度（百分点）', fontsize=12, **fkw())
    ax.set_title(f'全参数微调 vs 原始模型 提升 (IoU@{MAIN_IOU})', fontsize=15, fontweight='bold', **fkw())
    ax.set_yticklabels(labels, fontsize=11, **fkw()); ax.grid(axis='x', alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"  ✅ {os.path.basename(path)}")


def plot_detections(gt_data, base_preds, v2_preds, img_list, charts_dir, n=8):
    """检测对比图：GT / Base / V2 三列"""
    print("\n  检测对比图:")

    base_map = {p['image_name']: p['preds'] for p in base_preds}
    v2_map = {p['image_name']: p['preds'] for p in v2_preds}
    img_path_map = {item['image_name']: item['image_path'] for item in img_list}

    # 选最有代表性的图
    candidates = []
    for img_name, gts in gt_data.items():
        bp = base_map.get(img_name, [])
        vp = v2_map.get(img_name, [])
        b_tp = count_tp(gts, bp)
        v_tp = count_tp(gts, vp)
        candidates.append((img_name, gts, bp, vp, v_tp - b_tp, len(gts), b_tp, v_tp))

        # 只保留V2提升TP在4到6之间的
    filtered = [item for item in candidates if 4 <= item[4] <= 6]
    random.shuffle(filtered)
    selected = filtered[:n]

    for idx, (img_name, gts, bp, vp, diff, ng, b_tp, v_tp) in enumerate(selected):
        img_path = img_path_map.get(img_name)
        if not img_path or not os.path.exists(img_path):
            continue
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            continue
        img_w, img_h = img.size

        fig, axes = plt.subplots(1, 3, figsize=(21, 7))

        def draw(ax, boxes, title,show_label=True):
            ax.imshow(img)
            ax.set_title(title, fontsize=11, fontweight='bold', **fkw())
            ax.axis('off')
            for b in boxes:
                px1 = b['bbox'][0] / 1000 * img_w
                py1 = b['bbox'][1] / 1000 * img_h
                px2 = b['bbox'][2] / 1000 * img_w
                py2 = b['bbox'][3] / 1000 * img_h
                c = CAT_COLORS.get(b['label'], '#FFFFFF')
                rect = patches.Rectangle(
                    (px1, py1), px2-px1, py2-py1,
                    linewidth=2, edgecolor=c, facecolor='none'
                )
                ax.add_patch(rect)
                if show_label:
                    ax.text(px1, max(py1-4, 0), b['label'], fontsize=7,
                            color='white', backgroundcolor=c,
                            verticalalignment='bottom', **fkw())

        draw(axes[0], gts, f'GT ({len(gts)}个目标)', show_label=False)
        draw(axes[1], bp, f'原始模型 ({len(bp)}个预测) TP={b_tp}', show_label=True)
        draw(axes[2], vp, f'全参数微调 ({len(vp)}个预测) TP={v_tp}', show_label=True)


        fig.suptitle(f'{img_name}  (V2提升: +{diff} TP)', fontsize=13, fontweight='bold')
        plt.tight_layout()
        out_name = f'detection_{idx+1:02d}_{img_name.split(".")[0]}.png'
        plt.savefig(os.path.join(charts_dir, out_name), dpi=200, bbox_inches='tight')
        plt.close()
        print(f"    ✅ {out_name}")


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 60)
    print("  Step 3: 生成可视化图表")
    print("=" * 60)

    os.makedirs(CHARTS_DIR, exist_ok=True)
    setup_font()

    # 读取
    print("\n  读取数据...")
    with open(EVAL_PATH, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
    with open(GT_PATH, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    with open(BASE_PATH, 'r', encoding='utf-8') as f:
        base_preds = json.load(f)
    with open(V2_PATH, 'r', encoding='utf-8') as f:
        v2_preds = json.load(f)

    img_list = []
    if os.path.exists(IMG_LIST_PATH):
        with open(IMG_LIST_PATH, 'r', encoding='utf-8') as f:
            img_list = json.load(f)
    else:
        # 兜底
        val_img_dir = '/root/autodl-tmp/qwen_vl/finetune_data/val'
        img_list = [{'image_name': k, 'image_path': os.path.join(val_img_dir, k)} for k in gt_data.keys()]

    base_results = eval_data['base_model']['results']
    v2_results = eval_data['v2_model']['results']
    b = base_results.get(MAIN_IOU, {})
    v = v2_results.get(MAIN_IOU, {})

    print(f"  测试图片数: {eval_data.get('config', {}).get('num_test_images', '?')}")

    # 生成图表
    print("\n  生成图表:")

    plot_overall(b, v, os.path.join(CHARTS_DIR, 'bar_overall.png'))

    plot_category_precision(
    b.get('category', {}), v.get('category', {}),
    os.path.join(CHARTS_DIR, 'bar_category_precision.png'))

    plot_category_recall(
        b.get('category', {}), v.get('category', {}),
        os.path.join(CHARTS_DIR, 'bar_category_recall.png'))

    plot_category_f1(
        b.get('category', {}), v.get('category', {}),
        os.path.join(CHARTS_DIR, 'bar_category_f1.png'))

    plot_f1_curve(base_results, v2_results, os.path.join(CHARTS_DIR, 'line_f1_vs_iou.png'))

    plot_improvement(b, v, os.path.join(CHARTS_DIR, 'bar_improvement.png'))

    plot_detections(gt_data, base_preds, v2_preds, img_list, CHARTS_DIR, n=8)

    print(f"\n{'='*60}")
    print(f"  ✅ 全部图表已生成！")
    print(f"  输出: {CHARTS_DIR}")

    # 列出所有生成的文件
    files = sorted(os.listdir(CHARTS_DIR))
    print(f"  共 {len(files)} 个文件:")
    for f in files:
        size = os.path.getsize(os.path.join(CHARTS_DIR, f)) / 1024
        print(f"    {f} ({size:.0f}KB)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()