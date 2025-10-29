import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import numpy as np
import os.path as op
import torch.nn.functional as F
from datasets import build_dataloader
from utils.checkpoint import Checkpointer
from model import build_model
from utils.metrics import Evaluator
from utils.iotools import load_train_configs
import random
import matplotlib.pyplot as plt
from PIL import Image
from datasets.cuhkpedes import CUHKPEDES
import argparse


def find_query_by_content(captions, target_text):
    """根据文本内容查找查询索引"""
    target_text_lower = target_text.lower()
    matches = []
    for idx, caption in enumerate(captions):
        if target_text_lower in caption.lower():
            matches.append((idx, caption))

    return matches


def get_one_query_caption_and_result_by_id(idx, indices, qids, gids, captions, img_paths, image_pids):
    """获取指定索引的查询结果"""
    # 确保查询ID是标量
    if torch.is_tensor(qids[idx]):
        query_id = qids[idx].item()
    else:
        query_id = qids[idx]

    query_caption = captions[idx]

    # 将 indices 移到 CPU
    indices_cpu = indices.cpu()
    image_indices = indices_cpu[idx].tolist()

    # 获取图像路径
    image_paths = [img_paths[j] for j in image_indices]

    # 获取图像对应的行人ID
    if torch.is_tensor(gids):
        gids_cpu = gids.cpu()
        image_ids = [gids_cpu[j].item() for j in image_indices]
    else:
        image_ids = [gids[j] for j in image_indices]

    # 查找真实匹配的图像：找到与查询文本对应行人ID的第一张图像
    gt_image_path = None
    for i, pid in enumerate(image_pids):
        if pid == query_id:
            gt_image_path = img_paths[i]
            break

    # 如果没找到，使用检索结果的第一个匹配图像
    if gt_image_path is None:
        for i, pid in enumerate(image_ids):
            if pid == query_id:
                gt_image_path = image_paths[i]
                break

    # 如果还是没找到，使用第一张图像
    if gt_image_path is None:
        gt_image_path = img_paths[0] if img_paths else "no_image_available.jpg"

    return query_id, image_ids, query_caption, image_paths, gt_image_path


def plot_retrieval_images(query_id, image_ids, query_caption, image_paths, gt_img_path, fname=None):
    """绘制检索结果图像"""
    print(f"检索结果各行人ID: {image_ids}")
    print(f"各行人图像路径:")
    for i, (path, pid) in enumerate(zip(image_paths, image_ids), 1):
        print(f"{i}. PID: {pid}, Path: {path}")

    # 创建图像
    fig = plt.figure(figsize=(16, 4))
    col = len(image_paths)

    # 真实匹配图像
    plt.subplot(1, col + 1, 1)
    try:
        if os.path.exists(gt_img_path):
            img = Image.open(gt_img_path)
            img = img.resize((128, 256))
            plt.imshow(img)
            plt.title("Ground Truth", fontsize=10)
        else:
            plt.text(0.5, 0.5, "GT Image\nNot Found", ha='center', va='center', fontsize=10)
    except Exception as e:
        plt.text(0.5, 0.5, f"GT Error", ha='center', va='center', fontsize=8)
    plt.xticks([])
    plt.yticks([])

    # 检索结果图像
    for i in range(col):
        plt.subplot(1, col + 1, i + 2)
        try:
            if i < len(image_paths) and os.path.exists(image_paths[i]):
                img = Image.open(image_paths[i])
                ax = plt.gca()

                # 设置边框颜色
                if i < len(image_ids) and image_ids[i] == query_id:
                    # 正确匹配：绿色边框
                    border_color = 'green'
                    border_width = 3
                    match_text = "Match"
                else:
                    # 错误匹配：红色边框
                    border_color = 'red'
                    border_width = 1.5
                    match_text = "No Match"

                for spine in ax.spines.values():
                    spine.set_color(border_color)
                    spine.set_linewidth(border_width)

                img = img.resize((128, 256))
                plt.imshow(img)
                plt.title(f"Rank {i + 1}\n{match_text}", fontsize=10)
            else:
                plt.text(0.5, 0.5, f"Rank {i + 1}\nImage Missing", ha='center', va='center', fontsize=10)
        except Exception as e:
            plt.text(0.5, 0.5, f"Rank {i + 1}\nLoad Error", ha='center', va='center', fontsize=8)
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()

    plt.show(block=True)

    # 阻塞显示
    plt.show(block=True)


def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='IRRA 检索可视化')
    parser.add_argument('--config_file', required=True, help='配置文件路径')
    parser.add_argument('--query_index', type=int, help='查询索引')
    parser.add_argument('--query_text', type=str, help='查询文本内容（部分匹配）')
    parser.add_argument('--dataset_split', type=str, default='test', choices=['train', 'test', 'val'],
                        help='数据集划分: train, test, val (默认: test)')
    args_cmd = parser.parse_args()

    # 加载配置
    args = load_train_configs(args_cmd.config_file)
    args.batch_size = 1024
    args.training = False
    device = "cuda"

    # 数据加载和模型初始化
    try:
        test_img_loader, test_txt_loader, num_classes = build_dataloader(args)
    except ValueError:
        test_img_loader, test_txt_loader = build_dataloader(args)
        num_classes = None

    model = build_model(args, num_classes=num_classes)
    checkpointer = Checkpointer(model)
    checkpointer.load(f=op.join(args.output_dir, 'best.pth'))
    model.to(device)

    evaluator = Evaluator(test_img_loader, test_txt_loader)

    # 特征提取
    qfeats, gfeats, qids, gids = evaluator._compute_embedding(model.eval())
    qfeats = F.normalize(qfeats, p=2, dim=1)
    gfeats = F.normalize(gfeats, p=2, dim=1)

    similarity = qfeats @ gfeats.t()
    _, indices = torch.topk(similarity, k=10, dim=1, largest=True, sorted=True)

    # 加载数据集 - 根据参数选择不同的划分
    dataset = CUHKPEDES(root='./data')

    if args_cmd.dataset_split == 'train':
        dataset_split = dataset.train
        print("使用训练集")
    elif args_cmd.dataset_split == 'test':
        dataset_split = dataset.test
        print("使用测试集")
    else:  # val
        dataset_split = dataset.val
        print("使用验证集")

    img_paths = dataset_split['img_paths']
    captions = dataset_split['captions']
    image_pids = dataset_split['image_pids']

    print(f"数据集信息: {args_cmd.dataset_split} 集")
    print(f"图像数量: {len(img_paths)}")
    print(f"文本数量: {len(captions)}")
    print(f"行人ID数量: {len(set(image_pids))}")

    # 确定目标查询索引
    target_indices = []

    if args_cmd.query_index is not None:
        # 使用指定的索引
        target_indices = [args_cmd.query_index]
        print(f"使用指定索引: {args_cmd.query_index}")

    elif args_cmd.query_text is not None:
        # 根据文本内容搜索
        matches = find_query_by_content(captions, args_cmd.query_text)
        if matches:
            print(f"找到 {len(matches)} 个匹配的文本索引:")
            for idx, caption in matches:
                print(f"  文本索引{idx}: {caption[:80]}...")
            target_indices = [idx for idx, _ in matches]
        else:
            print(f"未找到包含文本 '{args_cmd.query_text}' 的查询")
            return
    else:
        # 默认使用数据集的第一个文本进行查询
        target_indices = [0]
        print("默认使用数据集的第一个文本进行查询")

    # 处理每个目标查询
    for target_index in target_indices:
        print(f"\n" + "=" * 60)
        print(f"处理查询索引: {target_index}")

        if target_index >= len(captions):
            print(f"错误: 索引 {target_index} 超出范围 (最大索引: {len(captions) - 1})")
            continue

        query_id, image_ids, query_caption, image_paths, gt_img_path = get_one_query_caption_and_result_by_id(
            target_index, indices, qids, gids, captions, img_paths, image_pids)

        print(f"查询文本: {query_caption}")
        print(f"查询文本对应的行人ID: {query_id}")

        plot_retrieval_images(
            query_id, image_ids, query_caption, image_paths, gt_img_path
        )

    print(f"\n所有查询处理完成！")


if __name__ == '__main__':
    main()