#!/usr/bin/env python3
"""
实验2：模型微调与微调前后效果对比
"""

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from datetime import datetime
import json
import warnings
warnings.filterwarnings("ignore")


class FineTuneExperiment:
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

    def prepare_dataset(self):
        """准备微调数据"""
        training_data = [
            "夜晚的风轻轻吹过城市的屋顶，路灯下的影子被拉得很长。",
            "她在雨中等了一整夜，只为再见他一面。",
            "有时候，沉默比言语更能表达心里的波澜。",
            "那只流浪猫每天都来窗边，像是在等一个旧友。",
            "记忆是一场梦，醒来后什么都抓不住。",
            "星光洒在湖面上，像碎成无数秘密的眼泪。",
            "他说，‘有一天我们会在春天重逢。’ 但春天过去了，又一年。",
            "火车缓缓驶离站台，她看着窗外的风景一点点远去。",
            "幸福从不是喧嚣，而是那一刻你忽然觉得世界都安静了。",
            "风吹起她的发丝，也吹散了那些不敢说出口的话。"
        ]
        return Dataset.from_dict({"text": training_data})

    def finetune(self, dataset, num_epochs=5):
        """执行微调实验"""
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], truncation=True, max_length=128)

        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        tokenized_dataset = tokenized_dataset.remove_columns(["text"])

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

        training_args = TrainingArguments(
            output_dir="./finetuned_model",
            num_train_epochs=num_epochs,
            per_device_train_batch_size=2,
            warmup_steps=50,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=5,
            save_total_limit=2,
            evaluation_strategy="no",
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )

        # 微调前对比
        print("\n=== 微调前生成示例 ===")
        #  prompts = ["人工智能的未来发展", "机器学习在医疗领域的应用"]
        prompts = ["黄昏的海边，她一个人走着，忽然想起了", "如果梦境能寄信给现实，那么我想告诉你"]
        before = {}
        for p in prompts:
            out = self.generate(p)
            print(f"Prompt: {p}\n输出: {out}\n")
            before[p] = out

        print("开始微调...")
        trainer.train()
        trainer.save_model()
        print("微调完成 ✅ 模型已保存到 ./finetuned_model")

        # 微调后对比
        print("\n=== 微调后生成示例 ===")
        after = {}
        for p in prompts:
            out = self.generate(p)
            print(f"Prompt: {p}\n输出: {out}\n")
            after[p] = out

        results = {
            "before_finetune": before,
            "after_finetune": after,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        with open("experiment_2_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print("实验2完成！结果已保存为 experiment_2_results.json")

    def generate(self, prompt, max_length=80, temperature=0.7):
        """文本生成辅助函数"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()


if __name__ == "__main__":
    exp = FineTuneExperiment()
    dataset = exp.prepare_dataset()
    exp.finetune(dataset)
