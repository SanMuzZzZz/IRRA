from prettytable import PrettyTable
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import numpy as np
import time
import os.path as op

from datasets import build_dataloader
from processor.processor import do_inference
from utils.checkpoint import Checkpointer
from utils.logger import setup_logger
from model import build_model
from utils.metrics import Evaluator
import argparse
#from utils.iotools import load_train_configs
from utils.options import get_args


if __name__ == '__main__':
    # --- 修改开始 ---
    # parser = argparse.ArgumentParser(description="IRRA Test")
    # parser.add_argument("--config_file", default='logs/CUHK-PEDES/iira/configs.yaml')
    # args = parser.parse_args()
    # args = load_train_configs(args.config_file)
    # 使用 get_args() 来解析所有命令行参数，包括我们新加的
    args = get_args() # <<<--- 使用 get_args() 获取所有参数
    # get_args 内部已经处理了 --config_file (如果需要的话，取决于 options.py 的实现)
    # 并且包含了 --perturb_type 等新参数
    # --- 修改结束 ---

    # 确认 args.output_dir 存在 (get_args 应该已经定义了 output_dir)
    # 如果 get_args 没有定义 output_dir，可能需要从 config_file 加载或手动设置
    if not hasattr(args, 'output_dir') or not args.output_dir:
         # 尝试从 config 文件推断 output_dir，或者设置一个默认值
         # 这里简单地报错退出，提示需要配置 output_dir
         print("Error: 'output_dir' not found in args. Please ensure it's defined in utils/options.py or loaded correctly.")
         exit(1)
    if not os.path.exists(args.output_dir):
        print(f"Warning: Output directory {args.output_dir} does not exist. Checkpoint loading might fail.")
        # os.makedirs(args.output_dir) # 或者如果需要，创建它

    args.training = False # 强制设为 False，因为这是测试脚本
    logger = setup_logger('IRRA', save_dir=args.output_dir, if_train=args.training)
    logger.info("Running Test Script") # 添加日志信息
    logger.info(str(args).replace(',', '\n')) # 打印所有使用的参数
    device = "cuda" if torch.cuda.is_available() else "cpu" # 自动检测设备

    # 现在 args 对象包含了 perturb_type 等信息，可以正确传递给 build_dataloader
    test_img_loader, test_txt_loader, num_classes = build_dataloader(args)
    model = build_model(args, num_classes=num_classes)
    checkpointer = Checkpointer(model)

    # 确认模型加载路径正确
    checkpoint_path = op.join(args.output_dir, 'best.pth')
    if not op.exists(checkpoint_path):
         print(f"Error: Checkpoint file not found at {checkpoint_path}")
         print("Please ensure 'output_dir' in your arguments points to the correct experiment log folder containing 'best.pth'.")
         exit(1)

    checkpointer.load(f=checkpoint_path)
    model.to(device)
    do_inference(model, test_img_loader, test_txt_loader)