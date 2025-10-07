#!/usr/bin/env python3
"""
实验1：不同Prompt和参数对文本生成的影响
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
import json
import warnings
warnings.filterwarnings("ignore")


class PromptExperiment:
    def __init__(self, model_name="Qwen/Qwen2-0.5B-Instruct"):
        """初始化模型"""
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.to(self.device)
        print(f"模型 {model_name} 加载完成")

    def generate_text(self, prompt, max_length=100, temperature=0.7, top_p=0.9, do_sample=True):
        """文本生成函数"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text[len(prompt):].strip()

    def run(self):
        """运行Prompt和参数对比实验"""
        print("\n" + "="*60)
        print("实验1：不同Prompt和参数对比")
        print("="*60)

        # 定义两组参数
        param_sets = {
            "保守参数": {"temperature": 0.3, "top_p": 0.6, "max_length": 80},
            "创意参数": {"temperature": 1.3, "top_p": 0.95, "max_length": 100}
        }

        # 定义两组prompt
        prompts = {
            "情感描写": "那天傍晚的天空格外安静，她一个人走在回家的路上，心里想着",
            "科技畅想": "如果未来的城市全部由人工智能管理，那么人类的生活将会"
        }

        results = {}

        for prompt_name, prompt in prompts.items():
            print(f"\n--- {prompt_name} ---")
            results[prompt_name] = {}

            for param_name, params in param_sets.items():
                print(f"\n{param_name}: {params}")
                generated = self.generate_text(prompt, **params)
                print(f"生成结果: {generated}")

                results[prompt_name][param_name] = {
                    "prompt": prompt,
                    "parameters": params,
                    "generated_text": generated
                }

        # 保存结果
        with open("experiment_1_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print("\n实验1完成！结果已保存为 experiment_1_results.json")

        # 简短分析
        print("\n=== 简要分析 ===")
        print("高温度参数（1.3）→ 文本更有创意，内容丰富但偶尔跳跃；低温度（0.3）→ 文本更稳重，逻辑清晰但较保守。")
        print("Prompt类型影响生成风格：")
        print(" - 情感描写类 → 更注重心理和场景细节；")
        print(" - 科技畅想类 → 更注重逻辑推理和未来场景描述。")


if __name__ == "__main__":
    exp = PromptExperiment()
    exp.run()
