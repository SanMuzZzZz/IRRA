import os
from typing import List
from torch.utils.data import Dataset
import os.path as osp
import logging
import torch
from utils.iotools import read_image
from utils.simple_tokenizer import SimpleTokenizer
from prettytable import PrettyTable
import random
import regex as re
import copy
import os.path as op # 确保导入了 op
from PIL import Image # 确保导入了 Image
import pandas as pd # 导入 pandas
import torchvision.transforms as T # 导入 T
from utils.iotools import read_image # 导入 read_image (IRRA项目中已有)


class BaseDataset(object):
    """
    Base class of text to image reid dataset
    """
    logger = logging.getLogger("IRRA.dataset")

    def show_dataset_info(self):
        num_train_pids, num_train_imgs, num_train_captions = len(
            self.train_id_container), len(self.train_annos), len(self.train)
        num_test_pids, num_test_imgs, num_test_captions = len(
            self.test_id_container), len(self.test_annos), len(
                self.test['captions'])
        num_val_pids, num_val_imgs, num_val_captions = len(
            self.val_id_container), len(self.val_annos), len(
                self.val['captions'])

        # TODO use prettytable print comand line table

        self.logger.info(f"{self.__class__.__name__} Dataset statistics:")
        table = PrettyTable(['subset', 'ids', 'images', 'captions'])
        table.add_row(
            ['train', num_train_pids, num_train_imgs, num_train_captions])
        table.add_row(
            ['test', num_test_pids, num_test_imgs, num_test_captions])
        table.add_row(['val', num_val_pids, num_val_imgs, num_val_captions])
        self.logger.info('\n' + str(table))


def tokenize(caption: str, tokenizer, text_length=77, truncate=True) -> torch.LongTensor:
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]

    result = torch.zeros(text_length, dtype=torch.long)
    if len(tokens) > text_length:
        if truncate:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        else:
            raise RuntimeError(
                f"Input {caption} is too long for context length {text_length}"
            )
    result[:len(tokens)] = torch.tensor(tokens)
    return result

def load_perturbation_map(csv_path):
    # Expand user path (~) if present
    csv_path = op.expanduser(csv_path)
    if not op.exists(csv_path):
        print(f"Warning: Perturbation CSV file not found at {csv_path}. Returning empty map.")
        return {}
    try:
        df = pd.read_csv(csv_path)
        perturbation_map = {}
        missing_count = 0
        required_columns = ['file_path', 'mask_path']
        if not all(col in df.columns for col in required_columns):
             print(f"Error: CSV {csv_path} needs columns: {required_columns}")
             return {}

        for index, row in df.iterrows():
            relative_key = row['file_path'].replace('\\', '/')
            # Expand user path for the perturbation file as well
            perturbation_abs_path = op.expanduser(row['mask_path'])
            if op.exists(perturbation_abs_path):
                perturbation_map[relative_key] = perturbation_abs_path
            else:
                missing_count += 1
        print(f"Loaded {len(perturbation_map)} perturbation mappings from {csv_path}.")
        if missing_count > 0:
            print(f"Warning: Checked {len(df)} rows, {missing_count} referenced perturbation files not found.")
        return perturbation_map
    except Exception as e:
        print(f"Error loading perturbation CSV {csv_path}: {e}")
        return {}

def get_relative_path_key(abs_path, dataset_folder_name):
    try:
        # Expand user path for robustness if needed, though img_paths are likely absolute already
        abs_path = op.expanduser(abs_path)
        key_part = dataset_folder_name + os.path.sep
        parts = abs_path.split(os.path.sep)

        try:
            base_index = parts.index(dataset_folder_name)
            # Adjust index based on whether 'imgs' exists and matches CSV format
            start_index = base_index + 1
            # Assuming CSV file_path starts after CUHK-PEDES/ (e.g., 'imgs/Market/file.jpg' or 'Market/file.jpg')
            # Let's try to match the CSV format directly by taking everything after dataset_folder_name
            relative_path = os.path.join(*parts[start_index:])
            return relative_path.replace('\\', '/')
        except ValueError:
            # Fallback if dataset_folder_name not in path
            print(f"Warning: '{dataset_folder_name}' not in path '{abs_path}'. Using fallback.")
            parts_fallback = abs_path.replace('\\', '/').split('/')
            if len(parts_fallback) >= 2 and parts_fallback[-2] in ['Market', 'test_query', 'train_query', 'imgs']:
                 # Match common CSV patterns like 'Market/file.jpg' or 'test_query/file.jpg'
                 return '/'.join(parts_fallback[-2:])
            elif len(parts_fallback) >= 3:
                 return '/'.join(parts_fallback[-3:]) # e.g., imgs/cam/file.jpg
            else:
                 return parts_fallback[-1]
    except Exception as e:
        print(f"Error extracting relative key from {abs_path}: {e}")
        return None


# === 新的 Dataset 类 ===
class PerturbedImageDataset(Dataset):
    """
    Dataset that loads original images and optionally adds perturbation
    before applying normalization. Returns PID and processed image Tensor.
    """
    def __init__(self, image_pids, img_paths, transform=None, # Basic transforms (Resize, ToTensor)
                 normalize_transform=None, # Normalization transform
                 perturbation_type='none',
                 attack_csv_path=None,
                 defend_csv_path=None,
                 epsilon=16.0,
                 dataset_folder_name='CUHK-PEDES'):

        self.image_pids = image_pids
        self.img_paths = img_paths
        self.transform = transform # Should include Resize, ToTensor
        self.normalize_transform = normalize_transform # Separate Normalize transform
        self.perturbation_type = perturbation_type
        self.epsilon = epsilon
        self.dataset_folder_name = dataset_folder_name

        self.attack_map = {}
        self.defend_map = {}
        if self.perturbation_type == 'attack' and attack_csv_path:
            self.attack_map = load_perturbation_map(attack_csv_path)
            if not self.attack_map: print("Warning: Attack map is empty.")
        elif self.perturbation_type == 'defend' and defend_csv_path:
            self.defend_map = load_perturbation_map(defend_csv_path)
            if not self.defend_map: print("Warning: Defend map is empty.")

        self.pil_to_tensor = T.ToTensor() # Local instance

    def __len__(self):
        return len(self.image_pids)

    def __getitem__(self, index):
        pid = self.image_pids[index]
        img_path = self.img_paths[index]

        try:
            # 1. Load original image (PIL)
            img_original_pil = read_image(img_path)

            # --- 修改开始 ---
            # 2. 应用基础变换 (Resize + ToTensor)
            #    self.transform 预期是 build_transforms 返回的 base_transform
            if self.transform is not None:
                # 应用包含 Resize 和 ToTensor 的变换
                img_tensor = self.transform(img_original_pil) # <<<--- 应用 transform
                # 现在 img_tensor 应该是 [0, 1] 范围且尺寸固定
            else:
                # 如果没有 transform，至少需要 ToTensor
                # 但这种情况可能意味着 Resize 也缺失了，最好确保 transform 被正确传递
                print(f"Warning: No base transform provided for PerturbedImageDataset. Applying ToTensor only.")
                img_tensor = self.pil_to_tensor(img_original_pil) # 尺寸可能不一致！
            # --- 修改结束 ---

            # 3. (Convert original image to Tensor [0, 1] - 这步已合并到上面的 transform 中)

            # 4. Add perturbation if needed
            if self.perturbation_type != 'none':
                relative_key = get_relative_path_key(img_path, self.dataset_folder_name)
                perturbation_path = None
                if relative_key:
                    if self.perturbation_type == 'attack':
                        perturbation_path = self.attack_map.get(relative_key)
                    elif self.perturbation_type == 'defend':
                        perturbation_path = self.defend_map.get(relative_key)

                if perturbation_path and op.exists(perturbation_path):
                    try:
                        perturbation_pil = read_image(perturbation_path)
                        
                        # --- 修改：也对扰动图应用基础变换 ---
                        if self.transform is not None:
                            # 对扰动图应用同样的 Resize + ToTensor
                            perturbation_tensor = self.transform(perturbation_pil) # <<<--- 应用 transform
                        else:
                            # Fallback if no transform
                            perturbation_tensor = self.pil_to_tensor(perturbation_pil)
                            # 手动 resize (确保尺寸一致) - 但最好依赖 transform
                            if perturbation_tensor.shape != img_tensor.shape:
                                 target_shape = img_tensor.shape[1:] # (H, W)
                                 print(f"Warning: Manually resizing perturbation tensor for {img_path} to {target_shape}")
                                 perturbation_tensor = T.functional.resize(perturbation_tensor, target_shape, antialias=True)
                        # --- 修改结束 ---

                        # (不再需要手动检查和 resize perturbation_tensor)
                        # if perturbation_tensor.shape != img_tensor.shape: ...

                        # Calculate delta and add
                        delta_tensor = (perturbation_tensor - 0.5) * (self.epsilon / 128.0)
                        perturbed_tensor_unclamped = img_tensor + delta_tensor
                        img_tensor = torch.clamp(perturbed_tensor_unclamped, 0.0, 1.0) # Update img_tensor

                    except Exception as e:
                        print(f"Warning: Error applying perturbation {perturbation_path} for {img_path}: {e}. Using original.")

            # 5. Apply Normalization (applied last)
            if self.normalize_transform is not None:
                img_tensor = self.normalize_transform(img_tensor)

            return pid, img_tensor

        except Exception as e:
            print(f"Error loading/processing image at index {index}, path {img_path}: {e}")
            # Return placeholder data or skip? Returning None might break DataLoader.
            # Let's return PID and a placeholder tensor (e.g., zeros)
            # Find expected shape from transform or args if possible, otherwise guess C,H,W

            # --- 修改这里的尺寸以匹配 args.img_size (默认 384, 128) ---
            # 您可以硬编码为 options.py 中的默认值，或者尝试从 args 获取
            # 假设默认尺寸为 H=384, W=128
            height, width = 384, 128 # <<<--- 确保这里的 H, W 与 build_transforms 一致
            placeholder_tensor = torch.zeros((3, height, width))
            # --- 修改结束 ---

            # Apply normalization to placeholder if possible
            if self.normalize_transform:
                 placeholder_tensor = self.normalize_transform(placeholder_tensor)
            return pid, placeholder_tensor

class ImageTextDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)

        tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            'caption_ids': tokens,
        }

        return ret


class ImageDataset(Dataset):
    def __init__(self, image_pids, img_paths, transform=None):
        self.image_pids = image_pids
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_pids)

    def __getitem__(self, index):
        pid, img_path = self.image_pids[index], self.img_paths[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return pid, img


class TextDataset(Dataset):
    def __init__(self,
                 caption_pids,
                 captions,
                 text_length: int = 77,
                 truncate: bool = True):
        self.caption_pids = caption_pids
        self.captions = captions
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.caption_pids)

    def __getitem__(self, index):
        pid, caption_text = self.caption_pids[index], self.captions[index] # 使用 caption_text 避免混淆

        # 将 caption 文本 tokenize
        tokens = tokenize(caption_text, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        # --- 新增：计算实际长度 (cap_len) ---
        # 找到第一个 padding token (0) 或 EOT token 的索引
        eot_token = self.tokenizer.encoder["<|endoftext|>"]
        try:
            # 找到 SOT 之后第一个 padding 或 EOT
            non_padding_indices = torch.where(tokens[1:] > 0)[0] # 排除 SOT
            if len(non_padding_indices) > 0:
                 # 加 1 是因为索引从0开始, 再加 1 是因为我们排除了 SOT
                 cap_len = int(non_padding_indices.max()) + 2
                 # 确保 cap_len 不超过最大长度
                 cap_len = min(cap_len, self.text_length)
                 # 处理特殊情况: 只有 SOT 和 EOT
                 if tokens[1] == eot_token and cap_len == 2:
                     pass # 正确长度是 2
                 # 检查最后一个 token 是否是 EOT (如果文本被截断，可能不是)
                 elif cap_len == self.text_length and tokens[cap_len - 1] != eot_token:
                     pass # 截断时长度也是对的
            else:
                 # 只有 SOT (以及可能的 EOT 或 padding)
                 cap_len = 2 if tokens[1] == eot_token else 1
        except Exception:
             # 备用逻辑: 找到最后一个非零 token
             non_zero_indices = torch.where(tokens > 0)[0]
             cap_len = int(non_zero_indices.max()) + 1 if len(non_zero_indices) > 0 else 1
             cap_len = min(cap_len, self.text_length) # 确保不超过最大长度

        return pid, tokens


class ImageTextMLMDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate

        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        
        caption_tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        mlm_tokens, mlm_labels = self._build_random_masked_tokens_and_labels(caption_tokens.cpu().numpy())

        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            'caption_ids': caption_tokens,
            'mlm_ids': mlm_tokens,
            'mlm_labels': mlm_labels
        }

        return ret

    def _build_random_masked_tokens_and_labels(self, tokens):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of int, tokenized sentence.
        :return: (list of int, list of int), masked tokens and related labels for MLM prediction
        """
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder)-3)) # 1 ~ 49405
        
        labels = []
        for i, token in enumerate(tokens):
            if 0 < token < 49405:
                prob = random.random()
                # mask token with 15% probability
                if prob < 0.15:
                    prob /= 0.15

                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        tokens[i] = mask

                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        tokens[i] = random.choice(token_range)

                    # -> rest 10% randomly keep current token

                    # append current token to output (we will predict these later)
                    labels.append(token)
                else:
                    # no masking token (will be ignored by loss function later)
                    labels.append(0)
            else:
                labels.append(0)
        
        if all(l == 0 for l in labels):
            # at least mask 1
            labels[1] = tokens[1]
            tokens[1] = mask

        return torch.tensor(tokens), torch.tensor(labels)