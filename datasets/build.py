import logging
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from datasets.sampler import RandomIdentitySampler
from datasets.sampler_ddp import RandomIdentitySampler_DDP
from torch.utils.data.distributed import DistributedSampler

from utils.comm import get_world_size

from .bases import ImageDataset, TextDataset, ImageTextDataset, ImageTextMLMDataset,PerturbedImageDataset

from .cuhkpedes import CUHKPEDES
from .icfgpedes import ICFGPEDES
from .rstpreid import RSTPReid

__factory = {'CUHK-PEDES': CUHKPEDES, 'ICFG-PEDES': ICFGPEDES, 'RSTPReid': RSTPReid}


def build_transforms(img_size=(384, 128), aug=False, is_train=True):
    height, width = img_size

    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    # --- 修改测试集的 Transform ---
    if not is_train:
        # 基础变换：Resize + ToTensor (不再包含 Normalize)
        base_transform = T.Compose([
            T.Resize((height, width)),
            T.ToTensor(), # 输出 Tensor [0, 1]
        ])
        # 单独创建 Normalize transform
        normalize_transform = T.Normalize(mean=mean, std=std)
        # 返回两个 transform
        return base_transform, normalize_transform # <<<--- 修改返回值

    # --- 训练集的 Transform 保持不变 (通常包含 Normalize) ---
    if aug:
        train_transform = T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            # T.RandomErasing(scale=(0.02, 0.4), value=mean), # RandomErasing 在 Normalize 后可能效果不佳？
        ])
    else:
        train_transform = T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    return train_transform # 训练集返回单个包含 Normalize 的 transform

def collate(batch):
    keys = set([key for b in batch for key in b.keys()])
    # turn list of dicts data structure to dict of lists data structure
    dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

    batch_tensor_dict = {}
    for k, v in dict_batch.items():
        if isinstance(v[0], int):
            batch_tensor_dict.update({k: torch.tensor(v)})
        elif torch.is_tensor(v[0]):
             batch_tensor_dict.update({k: torch.stack(v)})
        else:
            raise TypeError(f"Unexpect data type: {type(v[0])} in a batch.")

    return batch_tensor_dict

def build_dataloader(args, tranforms=None): # 函数签名不变
    logger = logging.getLogger("IRRA.dataset")
    num_workers = args.num_workers
    dataset = __factory[args.dataset_name](root=args.root_dir)
    num_classes = len(dataset.train_id_container)

    if args.training:
        # --- 训练 DataLoader 部分保持不变 ---
        train_transforms = build_transforms(img_size=args.img_size, aug=args.img_aug, is_train=True)
        # val_transforms 通常用于训练过程中的验证，这里也保持原样（包含Normalize）
        # 如果训练中的验证也需要支持扰动，则需要类似测试集的修改
        _val_base_transforms, _val_normalize_transform = build_transforms(img_size=args.img_size, is_train=False) # 获取测试/验证的transforms

        # ... (创建 train_set, train_loader 的逻辑不变) ...

        # --- 验证 DataLoader (如果使用，通常不加扰动) ---
        # 假设验证集使用原始 ImageDataset 和包含 Normalize 的 transform
        # 注意：build_transforms 返回两个值了，需要处理
        val_img_set = ImageDataset(dataset.val['image_pids'], dataset.val['img_paths'], T.Compose([_val_base_transforms, _val_normalize_transform])) # 组合起来
        val_txt_set = TextDataset(dataset.val['caption_pids'], dataset.val['captions'], text_length=args.text_length)

        val_img_loader = DataLoader(val_img_set, batch_size=args.test_batch_size, shuffle=False, num_workers=num_workers) # 使用 test_batch_size
        val_txt_loader = DataLoader(val_txt_set, batch_size=args.test_batch_size, shuffle=False, num_workers=num_workers)

        return train_loader, val_img_loader, val_txt_loader, num_classes

    else:
        # --- 测试 DataLoader 部分修改 ---
        # 获取基础变换 (Resize, ToTensor) 和 Normalize 变换
        test_base_transforms, test_normalize_transform = build_transforms(img_size=args.img_size, is_train=False)

        ds = dataset.test

        # 1. 默认使用全部数据
        image_pids=ds['image_pids']
        img_paths=ds['img_paths']
        caption_pids=ds['caption_pids']
        captions=ds['captions']
        
        # 2. 检查是否需要截取子集 (减小分母)
        N = args.test_query_subset
        if hasattr(args, 'test_query_subset') and N > 0:
            if N > len(captions):
                logger.warning(f"--test_query_subset ({N}) is larger than total queries ({len(captions)}). Using all queries.")
            else:
                logger.info(f"Reducing test query set (denominator) to first {N} queries.")
                # 仅截取查询列表 (caption_pids 和 captions)
                caption_pids = caption_pids[:N]
                captions = captions[:N]
                # 注意：图像库 (image_pids, img_paths) 保持完整，以确保检索难度不变

        # --- 使用 PerturbedImageDataset ---
        logger.info(f"Building test image loader with perturbation type: {args.perturb_type}")
        test_img_set = PerturbedImageDataset(
            image_pids=ds['image_pids'],
            img_paths=ds['img_paths'],
            transform=test_base_transforms,             # 传递基础变换
            normalize_transform=test_normalize_transform, # 传递 Normalize 变换
            perturbation_type=args.perturb_type,        # 传递扰动类型
            attack_csv_path=args.attack_csv,            # 传递 CSV 路径
            defend_csv_path=args.defend_csv,            # 传递 CSV 路径
            clean_ttc_csv_path=args.clean_ttc_csv,
            epsilon=args.perturb_epsilon,               # 传递 Epsilon
            dataset_folder_name=args.dataset_folder_name # 传递数据集文件夹名
        )
        # ---

        test_txt_set = TextDataset(caption_pids, captions, text_length=args.text_length)

        test_img_loader = DataLoader(test_img_set, batch_size=args.test_batch_size, shuffle=False, num_workers=num_workers)
        test_txt_loader = DataLoader(test_txt_set, batch_size=args.test_batch_size, shuffle=False, num_workers=num_workers)

        return test_img_loader, test_txt_loader, num_classes
