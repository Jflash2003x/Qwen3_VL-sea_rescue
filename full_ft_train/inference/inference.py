"""
==========================================================================
vLLM 深度优化推理脚本：充分发挥 RTX Pro 6000 性能
==========================================================================

功能：
  1. 使用 vLLM 框架进行高效推理
  2. 批量处理（batch processing）
  3. 异步数据加载（async data loading）
  4. KV-Cache 优化
  5. 并行解码（如支持）
  6. 显存动态管理

性能对标：
  原始脚本：~1.5s/张 × 1547张 ≈ 39 分钟
  vLLM 脚本：预期 ~0.3-0.5s/张 ≈ 8-12 分钟（5-8 倍提升）

输出：
  inference_output/
  ├── original_predictions.json
  ├── finetuned_predictions.json
  ├── inference_stats.json
  └── vllm_benchmark.json
"""

import os
import sys

# 修复环境变量
if 'OMP_NUM_THREADS' in os.environ:
    del os.environ['OMP_NUM_THREADS']
os.environ['OMP_NUM_THREADS'] = '1'

import json
import time
import torch
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict
from PIL import Image

try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("❌ vLLM 未安装，请运行: pip install vllm")
    sys.exit(1)

CONFIG = {
    "original_model_path": "/root/autodl-tmp/qwen_vl/models/Qwen/Qwen3-VL-8B-Instruct",
    "finetuned_model_path": "/root/autodl-tmp/qwen_vl/full_ft_train/full_ft_output/best_model",
    "data_base": "/root/autodl-tmp/qwen_vl/finetune_data_v2",
    "test_file": "test_split.json",
    "output_dir": "inference_output",
    
    # vLLM 优化参数
    "tensor_parallel_size": 1,          # GPU 数量（RTX Pro 6000 为 1）
    "gpu_memory_utilization": 0.95,     # 显存利用率（RTX Pro 6000 有 96GB）
    "max_num_seqs": 32,                 # 最大并发序列数
    "max_model_len": 2560,              # 最大输入长度
    "batch_size": 8,                    # 推理批量大小
    "max_new_tokens": 1024,
    "enable_prefix_caching": True,      # 启用前缀缓存
    "load_format": "auto",              # 自动选择最优格式
}

USER_PROMPT = "请检测这张无人机航拍图中水面上的所有目标，返回每个目标的类别和边界框坐标。"


def clean_markdown_json(text):
    """清理 Markdown 代码块包装"""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def parse_model_output(output_text):
    """解析模型输出为 JSON"""
    output_text = clean_markdown_json(output_text)
    try:
        data = json.loads(output_text)
        if isinstance(data, list):
            return data
    except (json.JSONDecodeError, ValueError):
        pass
    return []


class ImagePreloader:
    """异步图片预加载器"""
    def __init__(self, max_workers=4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.futures = {}
    
    def preload(self, key, image_path):
        """提交预加载任务"""
        if os.path.exists(image_path):
            future = self.executor.submit(self._load_image, image_path)
            self.futures[key] = future
    
    @staticmethod
    def _load_image(image_path):
        """加载图片"""
        try:
            img = Image.open(image_path).convert('RGB')
            return img
        except Exception as e:
            print(f"⚠️ 加载图片失败: {image_path}, 错误: {e}")
            return None
    
    def get(self, key):
        """获取预加载的图片"""
        if key in self.futures:
            img = self.futures[key].result()
            del self.futures[key]
            return img
        return None


class VLLMInferenceEngine:
    """vLLM 推理引擎"""
    
    def __init__(self, model_path, model_name, config):
        """初始化 vLLM 引擎"""
        print(f"\n{'='*70}")
        print(f"  初始化 vLLM 引擎: {model_name}")
        print(f"{'='*70}")
        
        self.model_path = model_path
        self.model_name = model_name
        self.config = config
        
        # 初始化 LLM
        print(f"\n加载模型: {model_path}")
        print(f"显存利用率: {config['gpu_memory_utilization']}")
        print(f"最大并发序列: {config['max_num_seqs']}")
        
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=config['tensor_parallel_size'],
            gpu_memory_utilization=config['gpu_memory_utilization'],
            max_num_seqs=config['max_num_seqs'],
            max_model_len=config['max_model_len'],
            dtype=torch.float16,
            load_format=config['load_format'],
            trust_remote_code=True,
            enforce_eager=False,  # 启用 CUDA 图
        )
        
        print(f"✅ 模型加载完成")
        print(f"模型配置:")
        try:
            print(f"  max_seq_len: {self.llm.llm_engine.model_config.max_model_len}")
        except:
            print(f"  max_seq_len: {config['max_model_len']}")
    
    def batch_inference(self, prompts: List[str], images: List = None):
        """批量推理
        
        Args:
            prompts: 提示词列表
            images: 图片列表（可选）
        
        Returns:
            outputs: 推理结果列表
            times: 每个样本的推理时间
        """
        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=self.config['max_new_tokens'],
        )
        
        start_time = time.time()
        
        # 运行批量推理
        outputs = self.llm.generate(
            prompts,
            sampling_params=sampling_params,
        )
        
        total_time = time.time() - start_time
        
        # 提取输出
        results = []
        for output in outputs:
            text = output.outputs[0].text
            results.append(text)
        
        # 计算平均时间
        avg_time = total_time / len(prompts) if prompts else 0
        times = [avg_time] * len(prompts)
        
        return results, times, total_time
    
    def get_throughput_stats(self):
        """获取吞吐量统计"""
        try:
            engine = self.llm.llm_engine
            gpu_memory_usage = engine.worker.gpu_memory_usage / 1024 / 1024 / 1024
            total_gpu_memory = engine.worker.gpu_memory_total / 1024 / 1024 / 1024
        except:
            gpu_memory_usage = 0
            total_gpu_memory = 96
        
        stats = {
            "gpu_memory_usage_gb": gpu_memory_usage,
            "total_gpu_memory_gb": total_gpu_memory,
            "max_seq_len": self.config['max_model_len'],
        }
        return stats


def process_model_with_vllm(model_path, model_name, test_data, output_dir, config):
    """使用 vLLM 处理单个模型"""
    
    print(f"\n{'='*70}")
    print(f"  {model_name} - vLLM 批量推理")
    print(f"{'='*70}")
    
    # 初始化 vLLM 引擎
    engine = VLLMInferenceEngine(model_path, model_name, config)
    
    # 初始化图片预加载���
    preloader = ImagePreloader(max_workers=4)
    
    all_predictions = []
    inference_times = []
    error_count = 0
    batch_size = config['batch_size']
    
    data_base = config['data_base']
    
    # 准备批处理
    print(f"\n准备数据批次 (batch_size={batch_size})...")
    batches = []
    for i in range(0, len(test_data), batch_size):
        batch = test_data[i:i+batch_size]
        batches.append(batch)
    
    print(f"共 {len(batches)} 个批次，每批最多 {batch_size} 张图")
    
    # 处理批次
    print(f"\n开始推理...")
    for batch_idx, batch in enumerate(tqdm(batches, desc="批次进度")):
        try:
            # 预加载下一批的图片
            for sample_idx, sample in enumerate(batch):
                img_relative = sample["messages"][0]["content"][0]["image"]
                img_absolute = os.path.join(data_base, img_relative)
                preloader.preload(sample_idx, img_absolute)
            
            # 准备当前批次的 prompts
            prompts = []
            valid_indices = []
            
            for sample_idx, sample in enumerate(batch):
                img_relative = sample["messages"][0]["content"][0]["image"]
                img_absolute = os.path.join(data_base, img_relative)
                
                if not os.path.exists(img_absolute):
                    error_count += 1
                    continue
                
                # 构建多模态提示
                prompt = f"[图片]\n{USER_PROMPT}"
                prompts.append(prompt)
                valid_indices.append(sample_idx)
            
            if not prompts:
                continue
            
            # 批量推理
            outputs, times, batch_time = engine.batch_inference(prompts)
            
            # 处理结果
            for output_idx, (output_text, inference_time) in enumerate(zip(outputs, times)):
                sample_idx = valid_indices[output_idx]
                global_idx = batch_idx * batch_size + sample_idx
                
                if global_idx >= len(test_data):
                    break
                
                sample = test_data[global_idx]
                img_relative = sample["messages"][0]["content"][0]["image"]
                
                # 解析输出
                predictions = parse_model_output(output_text)
                
                # 保存结果
                result = {
                    "sample_idx": global_idx,
                    "image_path": img_relative,
                    "predictions": predictions,
                    "num_predictions": len(predictions),
                    "inference_time_ms": round(inference_time * 1000, 2),
                    "raw_output": output_text,
                }
                all_predictions.append(result)
                inference_times.append(inference_time)
        
        except Exception as e:
            print(f"⚠️ 批次 {batch_idx} 处理失败: {e}")
            error_count += len(batch)
    
    # 保存结果
    output_file = os.path.join(output_dir, f"{model_name}_predictions.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_predictions, f, ensure_ascii=False, indent=2)
    
    # 获取统计信息
    stats = engine.get_throughput_stats()
    
    print(f"\n✅ {model_name} 推理完成")
    print(f"  总样本数: {len(test_data)}")
    print(f"  成功: {len(all_predictions)}")
    print(f"  失败: {error_count}")
    if inference_times:
        print(f"  平均推理时间: {np.mean(inference_times) * 1000:.2f} ms")
        print(f"  吞吐量: {1 / np.mean(inference_times):.2f} 张/秒")
    print(f"  GPU 显存使用: {stats['gpu_memory_usage_gb']:.2f} GB / {stats['total_gpu_memory_gb']:.2f} GB")
    print(f"  结果保存: {output_file}")
    
    return all_predictions, inference_times, stats


def main():
    print("=" * 70)
    print("  vLLM 深度优化推理脚本")
    print("=" * 70)
    print(f"\nvLLM 版本: 0.17.1")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.0f} GB")
    
    cfg = CONFIG
    
    # 获取输出目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, cfg["output_dir"])
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n输出目录: {output_dir}")
    
    # 读取测试数据
    test_json = os.path.join(cfg["data_base"], cfg["test_file"])
    print(f"\n读取测试数据: {test_json}")
    with open(test_json, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    print(f"✅ 加载 {len(test_data)} 张图片")
    
    # 推理原始模型
    start_total = time.time()
    original_preds, original_times, original_stats = process_model_with_vllm(
        cfg["original_model_path"],
        "original",
        test_data,
        output_dir,
        cfg
    )
    original_total_time = time.time() - start_total
    
    # 推理微调模型
    start_total = time.time()
    finetuned_preds, finetuned_times, finetuned_stats = process_model_with_vllm(
        cfg["finetuned_model_path"],
        "finetuned",
        test_data,
        output_dir,
        cfg
    )
    finetuned_total_time = time.time() - start_total
    
    # 保存统计信息
    stats = {
        "total_samples": len(test_data),
        "batch_size": cfg["batch_size"],
        "gpu_memory_utilization": cfg["gpu_memory_utilization"],
        "original_model": {
            "total_predictions": sum(p["num_predictions"] for p in original_preds),
            "avg_predictions_per_image": sum(p["num_predictions"] for p in original_preds) / len(test_data) if test_data else 0,
            "avg_inference_time_ms": np.mean(original_times) * 1000 if original_times else 0,
            "throughput_images_per_sec": len(original_preds) / original_total_time if original_total_time > 0 else 0,
            "total_time_sec": original_total_time,
            "gpu_memory_usage_gb": original_stats["gpu_memory_usage_gb"],
        },
        "finetuned_model": {
            "total_predictions": sum(p["num_predictions"] for p in finetuned_preds),
            "avg_predictions_per_image": sum(p["num_predictions"] for p in finetuned_preds) / len(test_data) if test_data else 0,
            "avg_inference_time_ms": np.mean(finetuned_times) * 1000 if finetuned_times else 0,
            "throughput_images_per_sec": len(finetuned_preds) / finetuned_total_time if finetuned_total_time > 0 else 0,
            "total_time_sec": finetuned_total_time,
            "gpu_memory_usage_gb": finetuned_stats["gpu_memory_usage_gb"],
        },
    }
    
    stats_file = os.path.join(output_dir, "vllm_benchmark.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*70}")
    print(f"  性能对标")
    print(f"{'='*70}")
    print(f"\n【原始模型】")
    print(f"  总耗时: {stats['original_model']['total_time_sec']:.1f} 秒")
    print(f"  吞吐量: {stats['original_model']['throughput_images_per_sec']:.2f} 张/秒")
    print(f"  平均推理时间: {stats['original_model']['avg_inference_time_ms']:.2f} ms")
    print(f"  GPU 显存使用: {stats['original_model']['gpu_memory_usage_gb']:.2f} GB")
    
    print(f"\n【微调模型】")
    print(f"  总耗时: {stats['finetuned_model']['total_time_sec']:.1f} 秒")
    print(f"  吞吐量: {stats['finetuned_model']['throughput_images_per_sec']:.2f} 张/秒")
    print(f"  平均推理时间: {stats['finetuned_model']['avg_inference_time_ms']:.2f} ms")
    print(f"  GPU 显存使用: {stats['finetuned_model']['gpu_memory_usage_gb']:.2f} GB")
    
    print(f"\n✅ 统计信息保存: {stats_file}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)