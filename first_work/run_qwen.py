import os
os.environ['HF_ENDPOINT'] = 'https://huggingface.co'  # 使用官方源

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import time
from pathlib import Path
import subprocess
import sys
from tqdm import *
import random
def install_package(package):
    """安装包"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def load_model():
    """加载Qwen2-0.5B-Instruct模型"""
    print("正在加载Qwen2-0.5B-Instruct模型...")
    
    model_name = "Qwen/Qwen2-0.5B-Instruct"
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, force_download=True)
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None, 
        force_download=True
    )
    
    if not torch.cuda.is_available():
        model = model.to("cpu")
    
    model.eval()
    print("模型加载完成")
    return model, tokenizer

def create_test_data():
    """从ModelScope加载WMT中英机器翻译测试集"""
    print("正在从ModelScope加载WMT中英机器翻译测试集...")
    
    from modelscope.msdatasets import MsDataset
    
    dataset = MsDataset.load('iic/WMT-Chinese-to-English-Machine-Translation-newstest', subset_name='default', split='test')
    
    # 转换为所需格式
    all_data = []
    for i, item in enumerate(dataset):
        all_data.append({
            "id": str(i + 1),
            "chinese": item["0"],
            "english": item["1"]
        })
    
    print(f"总共加载了 {len(all_data)} 条WMT测试数据")
    
    # 随机选择70条样本
    max_samples = 70
    if len(all_data) >= max_samples:
        # 设置随机种子以确保结果可重现（可选）
        random.seed(42)
        test_data = random.sample(all_data, max_samples)
        print(f"随机选择了 {max_samples} 条测试样本")
    else:
        test_data = all_data
        print(f"数据总量不足 {max_samples} 条，使用全部 {len(all_data)} 条数据")
    
    # 重新分配ID（可选，保持ID的连续性）
    for i, item in enumerate(test_data):
        item["id"] = str(i + 1)
    
    # 保存测试数据
    with open("test_data.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    print(f"成功准备 {len(test_data)} 条测试数据")
    return test_data
        

def translate_text(model, tokenizer, chinese_text):
    """翻译单个文本"""
    prompt = f"请将以下中文翻译成英文：\n\n中文：{chinese_text}\n\n英文："
    # print(prompt)
    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    # print(inputs)

    # 确保输入张量在正确的设备上
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # 生成翻译
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            # 最大单词数
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    # 解码输出
    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

    # 清理输出，只保留第一个句子的翻译
    if '\n' in generated_text:
        generated_text = generated_text.split('\n')[0].strip()
    return generated_text

def run_inference(model, tokenizer, test_data):
    """运行推理"""
    print("开始翻译推理...")
    results = []
    
    for i, item in tqdm(enumerate(test_data)):
        print(f"翻译进度: {i+1}/{len(test_data)}")
        
        chinese_text = item["chinese"]
        start_time = time.time()
        
        # 翻译
        translation = translate_text(model, tokenizer, chinese_text)
        
        end_time = time.time()
        inference_time = end_time - start_time
        
        result = {
            "id": item["id"],
            "chinese": chinese_text,
            "english_reference": item["english"],
            "english_translation": translation,
            "inference_time": inference_time
        }
        
        results.append(result)
        
        # 打印结果
        print(f"中文: {chinese_text}")
        print(f"参考翻译: {item['english']}")
        print(f"模型翻译: {translation}")
        print(f"推理时间: {inference_time:.2f}秒")
        print("-" * 50)
    
    # 保存结果
    with open("translation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"翻译完成，结果已保存到 translation_results.json")
    return results

def evaluate_with_bleu(results):
    """使用BLEU评估翻译效果"""
    print("开始BLEU评估...")
    
    try:
        # 尝试导入sacrebleu
        import sacrebleu
    except ImportError:
        print("正在安装sacrebleu...")
        install_package("sacrebleu")
        import sacrebleu
    
    # 准备数据
    references = []
    candidates = []
    
    for result in results:
        references.append(result["english_reference"])
        candidates.append(result["english_translation"])
    
    # 计算BLEU分数
    bleu = sacrebleu.corpus_bleu(candidates, [references])
    
    # 计算平均推理时间
    avg_time = sum(r["inference_time"] for r in results) / len(results)
    
    # 打印评估结果
    print("\n" + "="*60)
    print("评估结果:")
    print("="*60)
    print(f"BLEU分数: {bleu.score:.4f}")
    print(f"BLEU精确度 (1-gram): {bleu.precisions[0]:.4f}")
    print(f"BLEU精确度 (2-gram): {bleu.precisions[1]:.4f}")
    print(f"BLEU精确度 (3-gram): {bleu.precisions[2]:.4f}")
    print(f"BLEU精确度 (4-gram): {bleu.precisions[3]:.4f}")
    print(f"长度惩罚: {bleu.bp:.4f}")
    print(f"平均推理时间: {avg_time:.4f}秒")
    print(f"评估样本数: {len(results)}")
    print("="*60)
    
    # 保存评估结果
    evaluation_results = {
        "bleu_score": bleu.score,
        "bleu_precisions": bleu.precisions,
        "bleu_bp": bleu.bp,
        "avg_inference_time": avg_time,
        "num_samples": len(results)
    }
    
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
    
    print("评估结果已保存到 evaluation_results.json")

def main():
    """主函数"""
    print("Qwen2-0.5B-Instruct模型本地推理与效果评估")
    print("="*60)
    
    # 检查CUDA
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    try:
        # 1. 创建测试数据
        test_data = create_test_data()
        
        # 2. 加载模型
        model, tokenizer = load_model()
        
        # 3. 运行推理
        results = run_inference(model, tokenizer, test_data)
        
        # 4. 评估效果
        evaluate_with_bleu(results)
        
        print("\n程序执行完成！")
        
    except Exception as e:
        print(f"运行出错: {str(e)}")
        raise

if __name__ == "__main__":
    main()
