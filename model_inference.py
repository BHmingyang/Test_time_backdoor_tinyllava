import os
import json
import torch
from torch.utils.data import DataLoader
from PIL import Image
from torch.utils.data import Dataset
from transformers import GenerationConfig
from tinyllava.model.load_model import load_pretrained_model
from tinyllava.utils.constants import DEFAULT_IMAGE_TOKEN
from tinyllava.data import ImagePreprocess
from utils import get_prompt, get_transform_processor
from tinyllava.utils import *
from tinyllava.data import *
from tinyllava.model import *
import logging
os.environ['HF_HOME'] = '/mnt/data1/zhy_dataset/model'

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('AnyDoor/eval_log.txt', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EvalDataset(Dataset):
    """ 评估专用数据集,🈯只包含原始输入 """
    def __init__(self, data_name, data_file, text_processor, image_processor, height=384, width=384, is_constraint=False):
        self.data_name = data_name
        if self.data_name == 'svit':
            vis_root = './AnyDoor/data/svit/raw/'
        elif self.data_name == 'dalle3':
            vis_root = './AnyDoor/data/dalle3'

        if is_constraint:
            self.constraint = 'Answer the queslion using a single wordphrase.'
        else:
            self.constraint = ''

        self.data = json.load(open(data_file, 'r')) 
        self.vis_root = vis_root
        self.text_processor = text_processor
        self.image_processor = image_processor
        # self.transform_processor, self.normalize = get_transform_processor(height, width)
        
        # self.eos_token = self.vlm_processor.tokenizer.eos_token


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        item = self.data[list(self.data.keys())[index]]
        img_id = item['image']
        image_path = os.path.join(self.vis_root, img_id)
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.image_processor(image).to(torch.bfloat16)

        # 原始问题处理
        qs_ori = item['text_input']
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs_ori
        msg = Message()
        msg.add_message(qs)
        # print(prompt)
        # print("messages: ",msg.messages)
        result = self.text_processor(msg.messages, mode='eval')
        # print("result:", result)
        input_ids = result['input_ids']

        # 记录input_ids到日志文件
        logger.info(f"样本索引: {index}, 图像ID: {img_id}")
        logger.info(f"问题: {qs_ori}")
        logger.info("-" * 50)



        return input_ids, image_tensor, image.size, img_id, qs_ori
    
def collate_fn(batch):
    input_ids, image_tensors, image_sizes, img_ids, text_inputs = zip(*batch)
    return {
        "input_ids": torch.stack(input_ids, dim=0),
        "images": torch.stack(image_tensors, dim=0),
        "image_sizes": image_sizes,
        "image_id": img_ids,  # 新增图像ID字段
        "text_input": text_inputs  # 新增文本输入字段
    }

def main():
    # 配置参数
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # Set the device for PyTorch
    torch.cuda.set_device(device)
    model_path = 'Zhang199/TinyLLaVA-Qwen2-0.5B-SigLIP'
    eval_set = "/home/buaadl/zhy/TinyLLaVA_Factory/AnyDoor/s_datasets/dalle3_eval_set_llava.json"
    output_path = "/home/buaadl/zhy/TinyLLaVA_Factory/AnyDoor/s_datasets/dalle3_eval_set_tinyllava.json"
    
    # 加载模型
    model, tokenizer, image_processor, _ = load_pretrained_model(
        model_path,
        load_8bit=False,
        load_4bit=False,
        torch_dtype=torch.bfloat16,
        device_map={"": device} 
    )
    model.to(device)
    data_args = model.config
    text_processor = TextPreprocess(tokenizer, 'phi')
    image_processor = ImagePreprocess(image_processor, data_args)
    eval_dataset = EvalDataset(
        data_name='dalle3',
        data_file=eval_set,
        image_processor=image_processor,
        text_processor=text_processor
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=1,
        collate_fn=collate_fn,
        shuffle=False
    )
    from accelerate import Accelerator
    accelerator = Accelerator(
        mixed_precision='bf16',
        device_placement=True  # 确保设备放置正确
    )

    # 生成配置
    generation_config = GenerationConfig(
                            max_new_tokens=512,
                            do_sample=False,  # 改为False表示不使用采样
                            num_beams=1,      # 设置为1表示使用贪心搜索
                            temperature=1.0,  # 温度设为1.0
                            top_p=1.0,        # top_p设为1.0
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            bos_token_id=tokenizer.bos_token_id 
                        )

    results = {}
    model.eval()
    with torch.no_grad():
        for batch in eval_dataloader:
            clean_outputs = model.generate(
                inputs=batch["input_ids"].to(accelerator.device),
                images=batch["images"].to(accelerator.device),
                image_sizes=384,
                generation_config=generation_config
            )
            answer = [
                text.split('ASSISTANT:')[-1].strip() 
                for text in tokenizer.batch_decode(clean_outputs, skip_special_tokens=True)
                ][0]
            # print("answer:", answer)
            
            # 保留原始数据结构
            new_item = {
                "image": batch["image_id"][0],
                "text_input": batch["text_input"][0],
                "answer_tinyllava": answer  # 保持字段名兼容
            }
            # print("new_item:", new_item)
            results[batch["image_id"][0]] = new_item
            

    # 保存结果
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {output_path}")

if __name__ == "__main__":
    main()