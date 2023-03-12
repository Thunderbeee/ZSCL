import copy
import os
from random import random

import clip
import torch

from . import utils
from .args import parse_arguments
from .models import evaluate, evaluate_fc, evaluate_wise_ft, finetune, finetune_fc, finetune_icarl
from .models.modeling import create_image_classifier


def merge(model_0, model_1, alpha=0.95):
    key_name = [k for k, v in model_0.named_parameters()]
    for i, (param_q, param_k) in enumerate(zip(model_0.parameters(), model_1.parameters())):
        param_k.data = param_k.data * alpha + param_q.data * (1 - alpha)
    return model_1


def main(args):
    print(args)
    utils.seed_all(args.seed)

    if "fc" in args.train_mode:
        assert args.train_mode in ["image-fc", "image-fc-fixed"]
        if args.eval_only:
            model = create_image_classifier(
                args, initialize=args.fc_init, setnone=args.fc_setnone
            )
            if args.load:
                utils.torch_load(model, args.load)
            elif args.save:
                checkpoint_pth = os.path.join(
                    args.save, f"zeroshot_{args.train_dataset}.pth"
                )
                utils.torch_load(model, checkpoint_pth)
            evaluate_fc(model, args)
        else:
            model = finetune_fc(args)
    else:
        assert args.train_mode in ["whole", "text", "image"]
        # assert args.method in ["finetune"]
        if args.eval_only:
            model, _, val_preprocess = clip.load(args.model, jit=False)
            if args.load:
                if args.wise_ft:
                    print("Use wise-ft.")
                    model_0 = copy.deepcopy(model)
                utils.torch_load(model, args.load)
                if args.wise_ft:
                    model = merge(model_0, model, alpha=args.alpha)
            elif args.save:
                checkpoint_pth = os.path.join(
                    args.save, f"clip_zeroshot_{args.train_dataset}.pth"
                )
                utils.torch_save(checkpoint_pth, model)
            evaluate(model, args, val_preprocess)
        elif args.method in ["icarl"]:
            model = finetune_icarl(args) 
        else:
            model = finetune(args)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
