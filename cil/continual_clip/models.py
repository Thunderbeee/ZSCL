from omegaconf import DictConfig
from tqdm import tqdm
import torch.nn.functional as F

import clip
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .utils import get_class_ids_per_task, get_class_names, batch, merge_we, wise_we, moving_avg, l2_loss, virtual_vocab, distillation
import copy

from .cc import conceptual_captions

from . import utils
import os
import random

from .dynamic_dataset import DynamicDataset


class ClassIncremental(nn.Module):
    def __init__(self, cfg, device, jit=False):
        super().__init__()
        self.prompt_template = cfg.prompt_template
        self.device = device
        self.classes_names = None
        self.model, self.transforms, _ = clip.load(cfg.model_name, device=device, jit=jit)
        self.ref_model = None
        self.class_ids_per_task = list(get_class_ids_per_task(cfg))
        self.current_class_names = []
        self.text_tokens = None
        self.dynamic_dataset = DynamicDataset(cfg)
    
    def forward(self, image):
        with torch.no_grad():
            logits_per_image, _ = self.model(image, self.text_tokens)
            probs = logits_per_image.softmax(dim=-1)
        return probs

    def adaptation(self, task_id, cfg, train_dataset, train_classes_names):
        self.current_class_names += get_class_names(self.classes_names, self.class_ids_per_task[task_id])
        self.text_tokens = clip.tokenize(
            [self.prompt_template.format(c) for c in self.current_class_names]
        ).to(self.device)

        if cfg.method != "zeroshot":
            self.train(task_id, cfg, train_dataset, train_classes_names)
    
    def train(self, task_id, cfg, train_dataset, train_classes_names):
        ### laoding dataset
        train_loader = DataLoader(train_dataset[task_id:task_id + 1], 
                                    batch_size=cfg.batch_size, 
                                    shuffle=True, num_workers=8)

        train_iter = iter(train_loader)

        if cfg.method == "iCaRL":
            self.dynamic_dataset.update(train_dataset[task_id:task_id + 1], self.model) 

        ### hardcoding 
        EPOCH = 1
        num_batches = len(train_loader)
        total_iterations = EPOCH * num_batches

        ### whole-model
        exclude_params_name = ["logit_scale"]
        params = [
            v for k, v in self.model.named_parameters() if k not in exclude_params_name
        ]
        logit_scale = self.model.logit_scale

        # optimizer
        optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = utils.cosine_lr(
            optimizer, cfg.lr, 30, total_iterations
        )

        # move model to device
        self.model = self.model.cuda()
        devices = list(range(torch.cuda.device_count()))
        print("Using devices", devices)
        if cfg.we is not None and cfg.method == "ZSCL":
            print("Averaging training")
            we_model = copy.deepcopy(self.model)
            we_model.cuda()
            we_n = 0


        if cfg.l2 > 0:
            print("L2 norm")
            l2_model = copy.deepcopy(self.model)
            l2_model.cuda()
        # text
        classnames = get_class_names(self.classes_names, self.class_ids_per_task[task_id])
        texts = [self.prompt_template.format(c) for c in classnames]
        
        if cfg.ce_method == "ext":
            _, _, test_preprocess = clip.load(cfg.model_name, jit=False)
            ref_sentences = conceptual_captions(
                test_preprocess,
                location="./Continual-CLIP/data",
                batch_size=cfg.batch_size,
            )
            if cfg.ref_sentences == "conceptual_captions":
                tmp_texts = ref_sentences.train_dataset.captions
            texts_ = texts
        elif cfg.ce_method == "previous":
            previous_class_names = self.current_class_names[:-len(self.class_ids_per_task[0])]
            tmp_texts = [self.prompt_template.format(c) for c in previous_class_names]
            texts.extend(tmp_texts)
            texts = clip.tokenize(texts).to(self.device)
        else:
            texts = clip.tokenize(texts).to(self.device)

        # method
        if cfg.method in ["ZSCL", "lwf", "lwfvr", "iCaRL"]:
            # (Ref Model) get reference model
            if cfg.method in ["ZSCL"]:
                print("[Method] ZSCL")
                print("[ref_model] Zero-shot")
                self.ref_model, _, test_preprocess = clip.load(cfg.model_name, jit=False)
            else:
                print("[Method] LwF/LwF-vr/iCaRL")
                print(f"[ref_model] last-task Model")
                self.ref_model, _, test_preprocess = clip.load(cfg.model_name, jit=False)
                for param_q, param_k in zip(self.ref_model.parameters(), self.model.parameters()):
                    param_q.data = param_k.data 

            self.ref_model = self.ref_model.cuda()          
            self.ref_model.eval()

            # (Ref Dataset) get reference dataset
            if cfg.method in ["ZSCL"]:
                assert cfg.ref_dataset == "conceptual_captions"
                print(f"[Ref Dataset] conceptual_captions")
                ref_dataset = conceptual_captions(
                    test_preprocess,
                    location="./Continual-CLIP/data", 
                    batch_size=cfg.batch_size,
                )
            elif cfg.method in ["iCaRL"]:
                ref_dataset = self.dynamic_dataset.get()
            else: ### lwf, lwf-vr
                assert cfg.ref_dataset is None
                print(f"[Ref Dataset] lwf: current_task")
                ref_dataset = DataLoader(train_dataset[task_id:task_id + 1], 
                                    batch_size=cfg.batch_size, 
                                    shuffle=True, num_workers=8)
                
            if cfg.method in ["iCaRL"]:
                ref_iter = batch(ref_dataset)
            else:
                ref_iter = iter(ref_dataset)

            # (Ref Text) get reference text
            assert cfg.ref_sentences == "conceptual_captions" or cfg.ref_sentences == "random" or cfg.ref_sentences is None

            if cfg.ref_sentences == "random": ### lwf-vr
                ref_texts = virtual_vocab()
                ref_texts = ref_texts.cuda()
                print("[Ref Sentences] Random Sentences")
            elif cfg.ref_sentences is not None:
                print(f"[Ref Sentences] {cfg.ref_sentences}")
                # please change to your location directory
                ref_sentences = conceptual_captions(
                    test_preprocess,
                    location="./Continual-CLIP/data", 
                    batch_size=cfg.batch_size,
                )
                if cfg.ref_sentences == "conceptual_captions":
                    ref_texts = ref_sentences.train_dataset.captions
                    ref_texts = clip.tokenize(ref_texts).cuda()
                else:
                    ref_template = ref_sentences.template
                    ref_texts = [ref_template(x) for x in ref_sentences.classnames]
                    ref_texts = clip.tokenize(ref_texts).cuda()
            else: ### lwf
                print(f"[Ref Sentences] lwf: {cfg.ref_dataset}")
                ref_classnames = get_class_names(self.classes_names, self.class_ids_per_task[task_id])
                ref_texts = clip.tokenize(
                    [self.prompt_template.format(c) for c in ref_classnames]
                ).to(self.device)
        
        # start training
        self.model.train()
        for iteration in tqdm(range(total_iterations + 1)):
            scheduler(iteration)
            try:
                inputs, targets, task_ids = next(train_iter)
            except:
                train_iter = iter(train_loader)
                inputs, targets, task_ids = next(train_iter)
            
            if cfg.dataset == "tinyimagenet" and task_id != 0:
                shift =  100 + (task_id-1) * cfg.increment
                targets -= shift
            elif cfg.dataset == "imagenet100" and task_id != 0:
                shift =  cfg.initial_increment + (task_id-1) * cfg.increment
                targets -= shift
            else:
                shift = task_id * cfg.increment
                targets -= shift
            
            inputs, targets = inputs.cuda(), targets.cuda()
            
            if cfg.ce_method == "ext":
                new_texts = copy.deepcopy(texts_)
                tmp_texts_ = random.sample(tmp_texts, 100)
                new_texts.extend(tmp_texts_)
                texts = clip.tokenize(new_texts).to(self.device)
            
            logits_per_image, _ = self.model(inputs, texts)
            loss = F.cross_entropy(logits_per_image, targets, label_smoothing=cfg.ls)


            if cfg.l2 > 0:
                loss_l2 = l2_loss(self.model, l2_model)
                loss += cfg.l2 * loss_l2

            if cfg.method in ["ZSCL", "lwf", "lwfvr", "iCaRL"]:
                if cfg.method in ["ZSCL"]:
                    try:
                        ref_images, ref_labels = next(ref_iter)
                    except:
                        ref_iter = iter(ref_dataset.train_loader)
                        ref_images, ref_labels = next(ref_iter)
                    ref_images, ref_labels = ref_images.cuda(), ref_labels.cuda()
                elif cfg.method in ["iCaRL"]:
                    try:
                        ref_images = next(ref_iter)
                    except:
                        ref_iter = batch(ref_dataset)
                        ref_images = next(ref_iter)
                    ref_images = ref_images.cuda()
                else: ### lwf, lwf-vr
                    try:
                        ref_images, ref_labels, task_ids = next(ref_iter)
                    except:
                        # print("Regenerate Ref_Dataset")
                        ref_iter = iter(ref_dataset)
                        ref_images, ref_labels, task_ids = next(ref_iter)
                    ref_images, ref_labels = ref_images.cuda(), ref_labels.cuda()

                with torch.no_grad():
                    # -- get ref text embedding --
                    ref_embeddings = self.ref_model(None, ref_texts)
                    ref_embeddings = ref_embeddings / ref_embeddings.norm(
                        dim=-1, keepdim=True
                    )
                    # -- get ref image embedding --
                    ref_out = self.ref_model(ref_images, None)
                    ref_out = ref_out / ref_out.norm(dim=-1, keepdim=True)
                # -- get image embedding --
                ref_out_current = self.model(ref_images, None)
                ref_out_current = ref_out_current / ref_out_current.norm(
                    dim=-1, keepdim=True
                )
                # -- image_loss --
                logits_current = logit_scale.exp() * ref_out_current @ ref_embeddings.t()
                logits_ref = logit_scale.exp() * ref_out @ ref_embeddings.t()
                loss_ZSCL = distillation(logits_ref, logits_current, T=2)
                # -- text_loss --
                logits_current_2 = logits_current.t()
                logits_ref_2 = logits_ref.t()
                loss_ZSCL_2 = distillation(logits_ref_2, logits_current_2, T=2)
                # -- final loss --
                loss = loss + 5 * loss_ZSCL + 5 * loss_ZSCL_2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if cfg.we is not None and cfg.method == "ZSCL" and iteration % cfg.avg_freq == 0:
                we_n += 1
                merge_we(self.model, we_model, we_n)

        if cfg.we is not None and cfg.method == "ZSCL":
            for param_q, param_k in zip(self.model.parameters(), we_model.parameters()):
                param_q.data = param_k.data 

        self.model.eval()

        

class DomainIncremental(nn.Module):
    pass


class TaskAgnostic(nn.Module):
    pass



def load_model(cfg: DictConfig, device: torch.device) -> nn.Module:
    r"""Load a CLIP model in different continual scenarios.
    
    Arguments:
        cfg (DictConfig): Experiment configurations.
        device (torch.device): Device to train (or) evaluate the model on.
        
    Returns:
        nn.Module: Return scenario specific CLIP model.
    """
    if cfg.scenario == "class":
        return ClassIncremental(cfg, device)
    elif cfg.scenario == "domain":
        return DomainIncremental(cfg, device)
    elif cfg.scenario == "task-aganostic":
        return TaskAgnostic(cfg, device)
    else:
        raise ValueError(f"""
            `{cfg.scenarios}` is not a valid scenario, 
            Please choose from ['class', "domain', 'task-agnostic']
        """)
    
