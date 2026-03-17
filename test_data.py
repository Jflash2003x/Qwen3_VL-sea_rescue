import json
from PIL import Image
import os

BASE = '/root/autodl-tmp/qwen_vl/finetune_data'

# 1. 读取 JSON
print('=== 测试1：读取JSON ===')
with open(os.path.join(BASE, 'train.json'), 'r') as f:
    train_data = json.load(f)
with open(os.path.join(BASE, 'val.json'), 'r') as f:
    val_data = json.load(f)
print(f'训练样本: {len(train_data)} 条')
print(f'验证样本: {len(val_data)} 条')

# 2. 检查前5条样本的图片是否能打开
print('\n=== 测试2：图片是否能正确读取 ===')
for i in range(5):
    sample = train_data[i]
    img_path_raw = sample['messages'][0]['content'][0]['image']
    img_path = os.path.join(BASE, img_path_raw)

    if os.path.exists(img_path):
        img = Image.open(img_path)
        print(f'  样本{i+1}: {img_path_raw} -> {img.size} ✅')
    else:
        print(f'  样本{i+1}: {img_path_raw} -> 文件不存在 ❌')

# 3. 全量检查图片路径
print('\n=== 测试3：全量检查图片路径 ===')
missing = 0
for sample in train_data:
    img_path_raw = sample['messages'][0]['content'][0]['image']
    img_path = os.path.join(BASE, img_path_raw)
    if not os.path.exists(img_path):
        missing += 1
        if missing <= 3:
            print(f'  缺失: {img_path_raw}')

print(f'训练集: 缺失 {missing}/{len(train_data)} 张')

missing_val = 0
for sample in val_data:
    img_path_raw = sample['messages'][0]['content'][0]['image']
    img_path = os.path.join(BASE, img_path_raw)
    if not os.path.exists(img_path):
        missing_val += 1
        if missing_val <= 3:
            print(f'  缺失: {img_path_raw}')

print(f'验证集: 缺失 {missing_val}/{len(val_data)} 张')

# 4. 检查数据格式
print('\n=== 测试4：数据格式检查 ===')
sample = train_data[0]
print(f'消息数: {len(sample["messages"])}')
print(f'用户角色: {sample["messages"][0]["role"]}')
print(f'助手角色: {sample["messages"][1]["role"]}')
print(f'内容类型: {[c["type"] for c in sample["messages"][0]["content"]]}')

if missing == 0 and missing_val == 0:
    print('\n✅ 全部测试通过！可以开始微调训练！')
else:
    print('\n⚠️ 有缺失文件，需要修复后再训练！')