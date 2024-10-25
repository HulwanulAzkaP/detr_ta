import argparse
import sys
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from roboflow import Roboflow
import torchvision
from PIL import Image
from util.misc import NestedTensor
from typing import List

# Impor modul-modul DETR
sys.path.append(os.path.abspath("path/to/detr"))
from models import build_model
from models.matcher import build_matcher
from models.detr import SetCriterion
from util.misc import NestedTensor, get_world_size, is_dist_avail_and_initialized
from datasets.coco import make_coco_transforms
from util import box_ops
from util.misc import accuracy, get_world_size, interpolate, is_dist_avail_and_initialized

# Konfigurasi argumen
parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--lr_backbone', default=1e-5, type=float)
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--epochs', default=300, type=int)
parser.add_argument('--lr_drop', default=200, type=int)
parser.add_argument('--clip_max_norm', default=0.1, type=float,
                    help='gradient clipping max norm')
parser.add_argument('--bbox_loss_coef', default=5, type=float)
parser.add_argument('--giou_loss_coef', default=2, type=float)
parser.add_argument('--eos_coef', default=0.1, type=float,
                    help="Relative classification weight of the no-object class")
parser.add_argument('--dataset_file', default='coco')
parser.add_argument('--coco_path', type=str)
parser.add_argument('--masks', action='store_true',
                    help="Train segmentation head if the flag is provided")
parser.add_argument('--aux_loss', action='store_true',
                    help="Disables auxiliary decoding losses (loss at each layer)")
parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                    help='device to use for training / testing')
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--resume', default='', help='resume from checkpoint')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='start epoch')
parser.add_argument('--eval', action='store_true')
parser.add_argument('--num_workers', default=2, type=int)
parser.add_argument('--num_queries', default=100, type=int,
                    help="Number of query slots")
parser.add_argument('--dec_layers', default=6, type=int,
                    help="Number of decoding layers in the transformer")
parser.add_argument('--enc_layers', default=6, type=int,
                    help="Number of encoding layers in the transformer")
parser.add_argument('--dim_feedforward', default=2048, type=int,
                    help="Intermediate size of the feedforward layers in the transformer blocks")
parser.add_argument('--hidden_dim', default=256, type=int,
                    help="Size of the embeddings (dimension of the transformer)")
parser.add_argument('--dropout', default=0.1, type=float,
                    help="Dropout applied in the transformer")
parser.add_argument('--nheads', default=8, type=int,
                    help="Number of attention heads inside the transformer's attentions")
parser.add_argument('--pre_norm', action='store_true')
parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                    help="Type of positional embedding to use on top of the image features")
parser.add_argument('--backbone', default='resnet50', type=str,
                    help="Name of the convolutional backbone to use")
parser.add_argument('--dilation', action='store_true',
                    help="If true, we replace stride with dilation in the last convolutional block (DC5)")
parser.add_argument('--train_backbone', action='store_true',
                    help="If true, we train the backbone")
parser.add_argument('--return_interm_layers', action='store_true',
                    help="If true, return intermediate layers of the backbone")
parser.add_argument('--set_cost_class', default=1, type=float,
                    help="Class coefficient in the matching cost")
parser.add_argument('--set_cost_bbox', default=5, type=float,
                    help="L1 box coefficient in the matching cost")
parser.add_argument('--set_cost_giou', default=2, type=float,
                    help="giou box coefficient in the matching cost")
parser.add_argument('--num_classes', default=2, type=int)

args = parser.parse_args()

rf = Roboflow(api_key="umCDBYYeGbFwUd9x2KRY")
project = rf.workspace("fire-qdubq").project("detr_api")
dataset = project.version(1).download("coco")


class RoboflowCOCODataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, transforms, split='train'):
        self.dataset_path = dataset_path
        self.transforms = transforms
        self.split = split
        
        annotation_file = os.path.join(dataset_path, split, "_annotations.coco.json")
        with open(annotation_file, "r") as f:
            self.coco_data = json.load(f)
        
        self.image_ids = [img['id'] for img in self.coco_data['images']]
        self.id_to_filename = {img['id']: img['file_name'] for img in self.coco_data['images']}
        self.id_to_size = {img['id']: (img['width'], img['height']) for img in self.coco_data['images']}
        
        self.annotations = {}
        for ann in self.coco_data['annotations']:
            if ann['image_id'] not in self.annotations:
                self.annotations[ann['image_id']] = []
            self.annotations[ann['image_id']].append(ann)
        
        # Buat pemetaan nama kelas ke ID
        self.class_mapping = {'api': 0, 'asap': 1}
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_filename = self.id_to_filename[img_id]
        img_path = os.path.join(self.dataset_path, self.split, img_filename)
        
        img = Image.open(img_path).convert("RGB")
        
        anns = self.annotations.get(img_id, [])
        boxes = []
        labels = []
        iscrowd = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x+w, y+h])
            category_name = next(cat['name'] for cat in self.coco_data['categories'] if cat['id'] == ann['category_id'])
            labels.append(self.class_mapping[category_name])
            iscrowd.append(ann.get('iscrowd', 0))
        
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
            "orig_size": torch.tensor(self.id_to_size[img_id]),
            "size": torch.tensor([img.height, img.width]),
            "iscrowd": torch.tensor(iscrowd, dtype=torch.int64)
        }
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch):
    model.train()
    criterion.train()
    
    for samples, targets in data_loader:
        samples = nested_tensor_from_tensor_list(samples).to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        print("Target keys:", targets[0].keys())
        print("Target labels shape:", targets[0]['labels'].shape)
        print("Target boxes shape:", targets[0]['boxes'].shape)

        outputs = model(samples)
        
        print(f"Output pred_logits shape: {outputs['pred_logits'].shape}")
        print(f"Output pred_boxes shape: {outputs['pred_boxes'].shape}")
        
        try:
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            print(f"Epoch: {epoch}, Loss: {losses.item()}")
        except Exception as e:
            print(f"Error in criterion: {e}")
            print(f"src_logits shape: {outputs['pred_logits'].transpose(1, 2).shape}")
            print(f"target_classes shape: {targets[0]['labels'].shape}")
            print(f"empty_weight shape: {criterion.empty_weight.shape}")
            raise e

        break  # Hanya proses batch pertama untuk debugging


@torch.no_grad()
def evaluate(model, criterion, data_loader, device):
    model.eval()
    criterion.eval()

    total_loss = 0
    num_batches = 0

    for samples, targets in data_loader:
        samples = nested_tensor_from_tensor_list(samples).to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        total_loss += losses.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    print(f"Validation Loss: {avg_loss}")
    return avg_loss


def main():
    # Inisialisasi dataset dan dataloader
    transform = make_coco_transforms(image_set='train')
    train_dataset = RoboflowCOCODataset(dataset.location, transforms=transform, split='train')
    val_dataset = RoboflowCOCODataset(dataset.location, transforms=transform, split='valid')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    # Inisialisasi model
    num_classes = 2  # api dan asap
    args.num_classes = num_classes

    # Buat model
    model, criterion, postprocessors = build_model(args)
    
    # Ubah lapisan terakhir model untuk menghasilkan 3 kelas (2 + latar belakang)
    if hasattr(model, 'class_embed'):
        model.class_embed = nn.Linear(model.class_embed.in_features, num_classes + 1)
    else:
        print("Warning: model doesn't have class_embed attribute. Make sure the model is configured correctly.")
    
    # Pindahkan model dan criterion ke device yang sesuai
    device = torch.device(args.device)
    model.to(device)
    criterion.to(device)

    # Sesuaikan empty_weight
    if hasattr(criterion, 'empty_weight'):
        print(f"Original empty_weight shape: {criterion.empty_weight.shape}")
        new_empty_weight = torch.ones(num_classes + 1, device=device)  # +1 untuk kelas no-object
        new_empty_weight[-1] = args.eos_coef
        criterion.empty_weight = new_empty_weight
        print(f"New empty_weight shape: {criterion.empty_weight.shape}")
        print(f"New empty_weight values: {criterion.empty_weight}")

    # Ganti SetCriterion
    if isinstance(criterion, SetCriterion):
        print("Replacing SetCriterion")
        new_criterion = SetCriterion(
            num_classes,
            matcher=criterion.matcher,
            weight_dict=criterion.weight_dict,
            eos_coef=args.eos_coef,
            losses=criterion.losses
        )
        criterion = new_criterion
    else:
        print(f"Criterion is not SetCriterion, it's {type(criterion)}")

    # Pindahkan criterion ke device yang sesuai
    criterion.to(device)

    # Periksa empty_weight lagi
    if hasattr(criterion, 'empty_weight'):
        print(f"Final empty_weight shape: {criterion.empty_weight.shape}")
        print(f"Final empty_weight values: {criterion.empty_weight}")

    # Inisialisasi optimizer
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    # Loop pelatihan
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        train_one_epoch(model, criterion, train_loader, optimizer, device, epoch)
        evaluate(model, criterion, val_loader, device)

    # Simpan model
    torch.save(model.state_dict(), 'detr_roboflow_coco_model.pth')


def nested_tensor_from_tensor_list(tensor_list):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)

def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


if __name__ == '__main__':
    main()



