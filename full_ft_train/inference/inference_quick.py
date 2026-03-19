"""
==========================================================================
快速推理对比脚本：微调前后模型输出格式一致性测试 (v2 - 修复版)
==========================================================================

修复内容：
  1. ✅ 处理 Markdown 代码块包装 (```json ... ```)
  2. ✅ 支持中英文标签混用
  3. ✅ 输出路径改为相对目录 (inference/)

功能：
  1. 加载原始模型和微调后的模型
  2. 在相同测试集上进行推理
  3. 对比两个模型的输出格式（结构、坐标格式、键值顺序）
  4. 验证坐标是否都归一化到 0-1000
  5. 生成详细的对比报告

输出：
  inference_results.json      → 所有推理结果（方便后续分析）
  inference_report.txt        → 对比报告（人类可读）
  inference_comparison.csv    → 统计对比表
"""

import os
import sys
import json
import csv
import torch
from datetime import datetime
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ╔══════════════════════════════════════════════════════════════╗
# ║  ⬇⬇⬇ 配置路径 ⬇⬇⬇                                         ║
# ╚══════════════════════════════════════════════════════════════╝

CONFIG = {
    # --- 模型路径 ---
    "original_model_path": "/root/autodl-tmp/qwen_vl/models/Qwen/Qwen3-VL-8B-Instruct",
    "finetuned_model_path": "/root/autodl-tmp/qwen_vl/full_ft_train/full_ft_output/best_model",
    
    # --- 数据路径 ---
    "data_base": "/root/autodl-tmp/qwen_vl/finetune_data_v2",
    "test_file": "test_split.json",
    
    # --- 推理配置 ---
    "num_test_samples": 5,  # 测试几个样本
    "max_new_tokens": 1024,
    
    # --- 输出路径 (相对于脚本所在目录) ---
    "output_dir": "./quick_output",  # 当前目录 (inference/)
}

# 用户 Prompt（与训练数据一致）
USER_PROMPT = "请检测这张无人机航拍图中水面上的所有目标，返回每个目标的类别和边界框坐标。"


# ╔══════════════════════════════════════════════════════════════╗
# ║  ⬆⬆⬆ 以下代码不需要改 ⬆⬆⬆                                  ║
# ╚══════════════════════════════════════════════════════════════╝


def clean_markdown_json(text):
    """
    清理 Markdown 代码块包装
    输入:  "```json\n[...]\n```"
    输出:  "[...]"
    """
    text = text.strip()
    
    # 移除前面的 ```json 或 ```
    if text.startswith("```json"):
        text = text[7:]  # 移除 "```json"
    elif text.startswith("```"):
        text = text[3:]  # 移除 "```"
    
    # 移除后面的 ```
    if text.endswith("```"):
        text = text[:-3]
    
    # 再次 strip 空白和换行
    return text.strip()


class OutputAnalyzer:
    """分析模型输出的格式"""
    
    @staticmethod
    def parse_output(output_text):
        """尝试解析模型输出为 JSON"""
        output_text = output_text.strip()
        
        # 第一步：清理 Markdown 代码块
        output_text = clean_markdown_json(output_text)
        
        # 第二步：尝试解析 JSON
        try:
            data = json.loads(output_text)
            return data, "success"
        except json.JSONDecodeError as e:
            return None, f"JSONDecodeError: {str(e)}"
    
    @staticmethod
    def validate_structure(data):
        """验证输出结构"""
        if not isinstance(data, list):
            return False, "输出不是列表"
        
        if len(data) == 0:
            return True, "空列表"
        
        # 检查第一个元素
        first_item = data[0]
        if not isinstance(first_item, dict):
            return False, "列表元素不是字典"
        
        required_keys = {"bbox_2d", "label"}
        actual_keys = set(first_item.keys())
        
        if not required_keys.issubset(actual_keys):
            return False, f"缺少必需的键。期望: {required_keys}, 实际: {actual_keys}"
        
        # 检查 bbox_2d 格式
        bbox_2d = first_item.get("bbox_2d")
        if not isinstance(bbox_2d, list) or len(bbox_2d) != 4:
            return False, f"bbox_2d 格式错误，应为长度为4的列表，实际: {type(bbox_2d)}"
        
        # 检查坐标是否在 0-1000 范围内
        for coord in bbox_2d:
            if not isinstance(coord, (int, float)):
                return False, f"坐标不是数字: {coord}"
            if coord < 0 or coord > 1000:
                return False, f"坐标超出 0-1000 范围: {coord}"
        
        return True, "结构正确"
    
    @staticmethod
    def get_key_order(data):
        """获取 JSON 键的顺序"""
        if not isinstance(data, list) or len(data) == 0:
            return None
        return list(data[0].keys())
    
    @staticmethod
    def compare_outputs(original_output, finetuned_output):
        """对比两个模型的输出"""
        result = {
            "original": {
                "raw": original_output,
                "cleaned": clean_markdown_json(original_output),
                "parsed": None,
                "parse_status": None,
                "structure_valid": None,
                "structure_msg": None,
                "key_order": None,
                "num_targets": None,
            },
            "finetuned": {
                "raw": finetuned_output,
                "cleaned": clean_markdown_json(finetuned_output),
                "parsed": None,
                "parse_status": None,
                "structure_valid": None,
                "structure_msg": None,
                "key_order": None,
                "num_targets": None,
            },
            "comparison": {
                "parse_both_success": False,
                "structure_both_valid": False,
                "key_order_same": False,
                "num_targets_same": False,
                "coordinates_match_tolerance_10": True,  # 允许 ±10 的误差
                "all_consistent": False,
            }
        }
        
        # 解析原始模型输出
        parsed_orig, status_orig = OutputAnalyzer.parse_output(original_output)
        result["original"]["parsed"] = parsed_orig
        result["original"]["parse_status"] = status_orig
        
        if parsed_orig is not None:
            valid, msg = OutputAnalyzer.validate_structure(parsed_orig)
            result["original"]["structure_valid"] = valid
            result["original"]["structure_msg"] = msg
            result["original"]["key_order"] = OutputAnalyzer.get_key_order(parsed_orig)
            result["original"]["num_targets"] = len(parsed_orig)
        
        # 解析微调模型输出
        parsed_fine, status_fine = OutputAnalyzer.parse_output(finetuned_output)
        result["finetuned"]["parsed"] = parsed_fine
        result["finetuned"]["parse_status"] = status_fine
        
        if parsed_fine is not None:
            valid, msg = OutputAnalyzer.validate_structure(parsed_fine)
            result["finetuned"]["structure_valid"] = valid
            result["finetuned"]["structure_msg"] = msg
            result["finetuned"]["key_order"] = OutputAnalyzer.get_key_order(parsed_fine)
            result["finetuned"]["num_targets"] = len(parsed_fine)
        
        # 对比
        result["comparison"]["parse_both_success"] = (
            result["original"]["parse_status"] == "success" and
            result["finetuned"]["parse_status"] == "success"
        )
        
        result["comparison"]["structure_both_valid"] = (
            result["original"]["structure_valid"] is True and
            result["finetuned"]["structure_valid"] is True
        )
        
        result["comparison"]["key_order_same"] = (
            result["original"]["key_order"] == result["finetuned"]["key_order"]
        )
        
        result["comparison"]["num_targets_same"] = (
            result["original"]["num_targets"] == result["finetuned"]["num_targets"]
        )
        
        # 对比坐标（允许一定偏差）
        if (parsed_orig is not None and parsed_fine is not None and 
            result["original"]["num_targets"] == result["finetuned"]["num_targets"]):
            coord_tolerance = 10
            coords_match = True
            for i, (target_orig, target_fine) in enumerate(zip(parsed_orig, parsed_fine)):
                bbox_orig = target_orig.get("bbox_2d", [])
                bbox_fine = target_fine.get("bbox_2d", [])
                
                if len(bbox_orig) != len(bbox_fine):
                    coords_match = False
                    break
                
                for coord_orig, coord_fine in zip(bbox_orig, bbox_fine):
                    if abs(coord_orig - coord_fine) > coord_tolerance:
                        coords_match = False
                        break
            
            result["comparison"]["coordinates_match_tolerance_10"] = coords_match
        
        # 综合判断
        result["comparison"]["all_consistent"] = (
            result["comparison"]["parse_both_success"] and
            result["comparison"]["structure_both_valid"] and
            result["comparison"]["key_order_same"] and
            result["comparison"]["num_targets_same"]
        )
        
        return result


def run_inference(model, processor, image_path, device):
    """运行推理"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": USER_PROMPT},
            ],
        },
    ]
    
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)
    
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=CONFIG["max_new_tokens"])
    
    output_ids = output_ids[:, inputs.input_ids.shape[1]:]
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    return response


def main():
    print("=" * 70)
    print("  微调前后模型输出格式对比工具 (v2 - 修复版)")
    print("=" * 70)
    
    cfg = CONFIG
    
    # 获取输出目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, cfg["output_dir"])
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n输出目录: {output_dir}")
    
    # 检查文件存在
    test_json = os.path.join(cfg["data_base"], cfg["test_file"])
    if not os.path.exists(test_json):
        print(f"❌ 测试数据不存在: {test_json}")
        sys.exit(1)
    
    if not os.path.exists(cfg["original_model_path"]):
        print(f"❌ 原始模型不存在: {cfg['original_model_path']}")
        sys.exit(1)
    
    if not os.path.exists(cfg["finetuned_model_path"]):
        print(f"❌ 微调模型不存在: {cfg['finetuned_model_path']}")
        sys.exit(1)
    
    # 加载模型
    print(f"\n[1/3] 加载模型...")
    print(f"  原始模型:  {cfg['original_model_path']}")
    original_model = Qwen3VLForConditionalGeneration.from_pretrained(
        cfg["original_model_path"],
        torch_dtype=torch.float16,
        device_map="auto",
    )
    original_processor = AutoProcessor.from_pretrained(cfg["original_model_path"])
    original_device = next(original_model.parameters()).device
    print(f"  ✅ 加载完成 (device: {original_device})")
    
    print(f"  微调模型:   {cfg['finetuned_model_path']}")
    finetuned_model = Qwen3VLForConditionalGeneration.from_pretrained(
        cfg["finetuned_model_path"],
        torch_dtype=torch.float16,
        device_map="auto",
    )
    finetuned_processor = AutoProcessor.from_pretrained(cfg["finetuned_model_path"])
    finetuned_device = next(finetuned_model.parameters()).device
    print(f"  ✅ 加载完成 (device: {finetuned_device})")
    
    # 读取测试数据
    print(f"\n[2/3] 读取测试数据...")
    with open(test_json, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # 选择不同图片的测试样本
    tested_images = set()
    test_samples = []
    for sample in test_data:
        img_path = sample["messages"][0]["content"][0]["image"]
        if img_path not in tested_images and len(test_samples) < cfg["num_test_samples"]:
            tested_images.add(img_path)
            test_samples.append(sample)
    
    print(f"  ✅ 选择了 {len(test_samples)} 个测试样本")
    
    # 推理对比
    print(f"\n[3/3] 推理对比...")
    print("=" * 70)
    
    all_results = []
    summary_stats = {
        "total_samples": len(test_samples),
        "parse_both_success": 0,
        "structure_both_valid": 0,
        "key_order_same": 0,
        "num_targets_same": 0,
        "coordinates_match": 0,
        "all_consistent": 0,
    }
    
    for i, sample in enumerate(test_samples):
        print(f"\n【样本 {i+1}/{len(test_samples)}】")
        
        img_relative = sample["messages"][0]["content"][0]["image"]
        img_absolute = os.path.join(cfg["data_base"], img_relative)
        ground_truth = sample["messages"][1]["content"] if len(sample["messages"]) > 1 else None
        
        print(f"  图片: {img_relative}")
        
        if not os.path.exists(img_absolute):
            print(f"  ⚠️ 图片不存在: {img_absolute}")
            continue
        
        # 原始模型推理
        print(f"  推理中 (原始模型)...", end="", flush=True)
        original_output = run_inference(original_model, original_processor, img_absolute, original_device)
        print(" ✓")
        
        # 微调模型推理
        print(f"  推理中 (微调模型)...", end="", flush=True)
        finetuned_output = run_inference(finetuned_model, finetuned_processor, img_absolute, finetuned_device)
        print(" ✓")
        
        # 对比分析
        comparison = OutputAnalyzer.compare_outputs(original_output, finetuned_output)
        comparison["sample_index"] = i
        comparison["image_path"] = img_relative
        comparison["ground_truth"] = ground_truth
        all_results.append(comparison)
        
        # 打印对比结果
        print(f"\n  【原始模型输出】")
        print(f"    解析状态:    {comparison['original']['parse_status']}")
        print(f"    结构有效:    {comparison['original']['structure_valid']}")
        print(f"    键顺序:      {comparison['original']['key_order']}")
        print(f"    目标数:      {comparison['original']['num_targets']}")
        if comparison['original']['parsed'] and comparison['original']['num_targets'] > 0:
            print(f"    第一个目标:  {comparison['original']['parsed'][0]}")
        
        print(f"\n  【微调模型输出】")
        print(f"    解析状态:    {comparison['finetuned']['parse_status']}")
        print(f"    结构有效:    {comparison['finetuned']['structure_valid']}")
        print(f"    键顺序:      {comparison['finetuned']['key_order']}")
        print(f"    目标数:      {comparison['finetuned']['num_targets']}")
        if comparison['finetuned']['parsed'] and comparison['finetuned']['num_targets'] > 0:
            print(f"    第一个目标:  {comparison['finetuned']['parsed'][0]}")
        
        print(f"\n  【对比结果】")
        print(f"    两者都解析成功:       {comparison['comparison']['parse_both_success']}")
        print(f"    两者结构都有效:       {comparison['comparison']['structure_both_valid']}")
        print(f"    键顺序相同:           {comparison['comparison']['key_order_same']}")
        print(f"    目标数相同:           {comparison['comparison']['num_targets_same']}")
        print(f"    坐标匹配 (±10):       {comparison['comparison']['coordinates_match_tolerance_10']}")
        print(f"    🎯 全部一致:          {comparison['comparison']['all_consistent']}")
        
        # 更新统计
        if comparison['comparison']['parse_both_success']:
            summary_stats['parse_both_success'] += 1
        if comparison['comparison']['structure_both_valid']:
            summary_stats['structure_both_valid'] += 1
        if comparison['comparison']['key_order_same']:
            summary_stats['key_order_same'] += 1
        if comparison['comparison']['num_targets_same']:
            summary_stats['num_targets_same'] += 1
        if comparison['comparison']['coordinates_match_tolerance_10']:
            summary_stats['coordinates_match'] += 1
        if comparison['comparison']['all_consistent']:
            summary_stats['all_consistent'] += 1
    
    # 保存结果
    print("\n" + "=" * 70)
    print("  保存结果...")
    
    # 保存原始数据
    results_json = os.path.join(output_dir, "inference_results.json")
    with open(results_json, 'w', encoding='utf-8') as f:
        # 清理 torch tensor 后再保存
        cleaned_results = []
        for result in all_results:
            cleaned_result = {
                "sample_index": result["sample_index"],
                "image_path": result["image_path"],
                "ground_truth": result["ground_truth"],
                "original": {
                    "raw": result["original"]["raw"],
                    "cleaned": result["original"]["cleaned"],
                    "parse_status": result["original"]["parse_status"],
                    "structure_valid": result["original"]["structure_valid"],
                    "structure_msg": result["original"]["structure_msg"],
                    "key_order": result["original"]["key_order"],
                    "num_targets": result["original"]["num_targets"],
                },
                "finetuned": {
                    "raw": result["finetuned"]["raw"],
                    "cleaned": result["finetuned"]["cleaned"],
                    "parse_status": result["finetuned"]["parse_status"],
                    "structure_valid": result["finetuned"]["structure_valid"],
                    "structure_msg": result["finetuned"]["structure_msg"],
                    "key_order": result["finetuned"]["key_order"],
                    "num_targets": result["finetuned"]["num_targets"],
                },
                "comparison": result["comparison"],
            }
            cleaned_results.append(cleaned_result)
        
        json.dump(cleaned_results, f, ensure_ascii=False, indent=2)
    print(f"  ✅ {results_json}")
    
    # 保存统计表
    csv_path = os.path.join(output_dir, "inference_comparison.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["指标", "通过", "失败", "通过率"])
        
        metrics = [
            ("解析成功", summary_stats['parse_both_success'], 
             summary_stats['total_samples'] - summary_stats['parse_both_success']),
            ("结构有效", summary_stats['structure_both_valid'],
             summary_stats['total_samples'] - summary_stats['structure_both_valid']),
            ("键顺序一致", summary_stats['key_order_same'],
             summary_stats['total_samples'] - summary_stats['key_order_same']),
            ("目标数一致", summary_stats['num_targets_same'],
             summary_stats['total_samples'] - summary_stats['num_targets_same']),
            ("坐标匹配", summary_stats['coordinates_match'],
             summary_stats['total_samples'] - summary_stats['coordinates_match']),
            ("全部一致", summary_stats['all_consistent'],
             summary_stats['total_samples'] - summary_stats['all_consistent']),
        ]
        
        for name, passed, failed in metrics:
            rate = (passed / summary_stats['total_samples'] * 100) if summary_stats['total_samples'] > 0 else 0
            writer.writerow([name, passed, failed, f"{rate:.1f}%"])
    
    print(f"  ✅ {csv_path}")
    
    # 保存文本报告
    report_path = os.path.join(output_dir, "inference_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("  微调前后模型输出格式对比报告\n")
        f.write("=" * 70 + "\n")
        f.write(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"原始模型: {cfg['original_model_path']}\n")
        f.write(f"微调模型: {cfg['finetuned_model_path']}\n")
        f.write(f"测试样本: {summary_stats['total_samples']}\n")
        
        f.write(f"\n{'─' * 70}\n")
        f.write("📊 统计汇总\n")
        f.write(f"{'─' * 70}\n")
        
        for name, passed, failed in metrics:
            rate = (passed / summary_stats['total_samples'] * 100) if summary_stats['total_samples'] > 0 else 0
            status = "✅" if passed == summary_stats['total_samples'] else "⚠️"
            f.write(f"{status} {name:<15} {passed}/{summary_stats['total_samples']} ({rate:>5.1f}%)\n")
        
        f.write(f"\n{'─' * 70}\n")
        f.write("💡 详细分析\n")
        f.write(f"{'─' * 70}\n")
        
        for result in all_results:
            f.write(f"\n【样本 {result['sample_index']+1}】{result['image_path']}\n")
            f.write(f"  原始模型解析: {result['original']['parse_status']}\n")
            f.write(f"  微调模型解析: {result['finetuned']['parse_status']}\n")
            f.write(f"  键顺序一致:   {result['comparison']['key_order_same']}\n")
            f.write(f"  目标数一致:   {result['comparison']['num_targets_same']}\n")
            f.write(f"  坐标匹配:     {result['comparison']['coordinates_match_tolerance_10']}\n")
            f.write(f"  综合评估:     {'✅ 完全一致' if result['comparison']['all_consistent'] else '⚠️ 存在差异'}\n")
    
    print(f"  ✅ {report_path}")
    
    # 打印总结
    print("\n" + "=" * 70)
    print("  📋 对比总结")
    print("=" * 70)
    
    for name, passed, failed in metrics:
        rate = (passed / summary_stats['total_samples'] * 100) if summary_stats['total_samples'] > 0 else 0
        status = "✅" if passed == summary_stats['total_samples'] else "⚠️"
        print(f"{status} {name:<15} {passed}/{summary_stats['total_samples']} ({rate:>5.1f}%)")
    
    print(f"\n{'=' * 70}")
    if summary_stats['all_consistent'] == summary_stats['total_samples']:
        print("🎉 完美！微调前后模型输出格式完全一致！")
    else:
        consistent_rate = (summary_stats['all_consistent'] / summary_stats['total_samples'] * 100)
        print(f"📊 一致性: {summary_stats['all_consistent']}/{summary_stats['total_samples']} ({consistent_rate:.1f}%)")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)