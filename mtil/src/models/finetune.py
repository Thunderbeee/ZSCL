import copy
import os

import clip.clip as clip
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .. import datasets, templates, utils
from .evaluation import evaluate, zeroshot_classifier
from .helpers import get_datasets_text, merge_we, wise_we, moving_avg, l2_loss, virtual_vocab, distillation

def finetune(args):
    model, train_preprocess, val_preprocess = clip.load(args.model, jit=False)
    if args.load is not None:
        utils.torch_load(model, args.load)

    if args.we_wise or (args.wise_merge and args.wise_ft_model != "zeroshot"):
        print("Using WiSE-FT with Loaded Model")
        model_fix, train_preprocess, val_preprocess = clip.load(args.model, jit=False)
        if args.load is not None:
            utils.torch_load(model_fix, args.load)

    if args.we or args.moving_avg or args.we_wise:
        print("Averaging training")
        if args.moving_avg and args.mv_avg_model == "zeroshot": # mv+zeroshot
            we_model, _, _ =  clip.load(args.model, jit=False)
            we_model.cuda()
            we_n = 0
        else: #we; mv+m; mv+t; we_wise
            we_model = copy.deepcopy(model)
            we_model.cuda()
            we_n = 0
    if args.l2 > 0:
        print("L2 norm")
        l2_model = copy.deepcopy(model)
        l2_model.cuda()

    # prepare dataset
    dataset_class = getattr(datasets, args.train_dataset)
    dataset = dataset_class(
        train_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
        batch_size_eval=args.batch_size_eval,
    )

    # prepare template
    if args.template is not None:
        template = getattr(templates, args.template)[0]
    else:
        template = dataset.template

    # number of iterations
    num_batches = len(dataset.train_loader)
    if args.epochs is not None:
        total_iterations = args.epochs * num_batches
    else:
        total_iterations = args.iterations
    if args.eval_every_epoch:
        eval_iterations = num_batches
    else:
        eval_iterations = args.eval_interval
    loss_interval = args.loss_interval
    print("Iterations per epoch:", num_batches)
    print("Total iterations:", total_iterations)

    # get params
    if args.train_mode == "text":
        print("[Training mode] Text Encoder")
        visual_params_name = [k for k, v in model.visual.named_parameters()]
        exclude_params_name = visual_params_name + ["logit_scale"]
        params = [
            v for k, v in model.named_parameters() if k not in exclude_params_name
        ]
    elif args.train_mode == "image":
        print("[Training mode] Image Encoder")
        params = model.visual.parameters()
    else:
        assert args.train_mode == "whole"
        print("[Training mode] Both Encoders")
        exclude_params_name = ["logit_scale"]
        params = [
            v for k, v in model.named_parameters() if k not in exclude_params_name
        ]

    # optimizer
    optimizer = torch.optim.AdamW(
        params, lr=args.lr, weight_decay=args.wd, betas=(0.9, args.beta2)
    )
    scheduler = utils.cosine_lr(
        optimizer, args.lr, args.warmup_length, total_iterations
    )

    # move model to device
    model = model.cuda()
    logit_scale = model.logit_scale
    devices = list(range(torch.cuda.device_count()))
    print("Using devices", devices)
    model = torch.nn.DataParallel(model, device_ids=devices)

    # text
    texts = [template(x) for x in dataset.classnames]
    texts = clip.tokenize(texts).cuda()

    # Method
    if args.method == "ZSCL":
        # (Ref Model) get reference model
        print("[Method] ZSCL")
        if args.ref_model is None:
            if args.ref_wise:
                print("[ref_model] WiSE-Zero-shot")
                ref_model, _, test_preprocess = clip.load(args.model, jit=False)
                for param_q, param_k in zip(ref_model.parameters(), model.module.parameters()):
                    param_q.data = param_q.data * (1 - args.ref_wise_alpha) + param_k.data * args.ref_wise_alpha
            else:    
                print("[ref_model] Zero-shot")
                ref_model, _, test_preprocess = clip.load(args.model, jit=False)
        else:
            print(f"[ref_model] {args.ref_model}")
            ref_model, _, test_preprocess = clip.load(args.model, jit=False)
            utils.torch_load(
                ref_model, args.ref_model
            )
        ref_model = ref_model.cuda()
        ref_model = torch.nn.DataParallel(ref_model, device_ids=devices)
        ref_model.eval()

        # (Ref Dataset) get reference dataset
        ref_dataset_cls = getattr(datasets, args.ref_dataset)
        print(f"[Ref Dataset] {args.ref_dataset}")
        if args.ref_dataset in ["ImageNetSM", "ImageNetSUB"]:
            ref_dataset = ref_dataset_cls(
                test_preprocess,
                location=args.data_location,
                batch_size=args.batch_size,
                num=args.num,
            )
        else:
            ref_dataset = ref_dataset_cls(
                test_preprocess,
                location=args.data_location,
                batch_size=args.batch_size,
            )
        ref_iter = iter(ref_dataset.train_loader)

        # (Ref Text) get reference text
        if args.text_datasets is not None:
            print("[Ref Sentences] Text-Datasets")
            ref_texts = get_datasets_text(args.text_datasets, args)
        elif args.ref_sentences == "random":
            ref_texts = virtual_vocab()
            print("[Ref Sentences] Random Sentences")
        elif args.ref_sentences is not None:
            ref_sentences_cls = getattr(datasets, args.ref_sentences)
            print(f"[Ref Sentences] {args.ref_sentences}")
            ref_sentences = ref_sentences_cls(
                test_preprocess,
                location=args.data_location,
                batch_size=args.batch_size,
            )
            if args.ref_sentences == "conceptual_captions":
                # breakpoint()
                ref_texts = ref_sentences.train_dataset.captions
                ref_texts = clip.tokenize(ref_texts).cuda()

            else:
                ref_template = ref_sentences.template
                ref_texts = [ref_template(x) for x in ref_sentences.classnames]
                ref_texts = clip.tokenize(ref_texts).cuda()
        else:
            print(f"[Ref Sentences] {args.ref_dataset}")
            ref_template = ref_dataset.template
            ref_texts = [ref_template(x) for x in ref_dataset.classnames]
            ref_texts = clip.tokenize(ref_texts).cuda()
            
    if args.train_mode == "text":
        embeddings = zeroshot_classifier(dataset.classnames, dataset.templates, model)

    for iteration in tqdm(range(total_iterations + 1)):
        # evaluation
        if eval_iterations is not None and iteration % eval_iterations == 0:
            if args.we or args.we_wise:
                evaluate(we_model, args, val_preprocess)
            else:
                evaluate(model.module, args, val_preprocess)

        # training
        if iteration % num_batches == 0:
            data_iter = iter(dataset.train_loader)

        # prepare model
        model.train()
        scheduler(iteration)

        # prepare data
        if args.train_dataset == 'ImageNet':
            try:
                train_batch = next(data_iter)
            except:
                data_iter = iter(dataset.train_loader)
                train_batch = next(data_iter)
            images, labels = train_batch["images"], train_batch["labels"]
        else:
            try:
                images, labels = next(data_iter)
            except:
                data_iter = iter(dataset.train_loader)
                images, labels = next(data_iter)
        images, labels = images.cuda(), labels.cuda()

        # ce loss
        # -- get text embedding --
        if args.train_mode != "text":
            embeddings = model(None, texts)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

        # -- get image embedding --
        out = model(images, None)
        out = out / out.norm(dim=-1, keepdim=True)

        # -- cross entropy loss --
        logits_per_image = logit_scale.exp() * out @ embeddings.t()
        loss = F.cross_entropy(logits_per_image, labels, label_smoothing=args.ls)

        if args.l2 > 0:
            loss_l2 = l2_loss(model, l2_model)
            loss += args.l2 * loss_l2

        if args.method == "ZSCL":

            # FIXME: only for ImageNet
            if args.ref_dataset in ["ImageNet", "ImageNetSM", "ImageNetSUB"]:
                try:
                    ref_batch = next(ref_iter)
                except:
                    ref_iter = iter(ref_dataset.train_loader)
                    ref_batch = next(ref_iter)
                ref_images, ref_labels = ref_batch["images"], ref_batch["labels"]
            else:
                try:
                    ref_images, ref_labels = next(ref_iter)
                except:
                    ref_iter = iter(ref_dataset.train_loader)
                    ref_images, ref_labels = next(ref_iter)
            ref_images, ref_labels = ref_images.cuda(), ref_labels.cuda()
            # breakpoint()
            with torch.no_grad():
                # -- get ref text embedding --
                ref_embeddings = ref_model(None, ref_texts)
                ref_embeddings = ref_embeddings / ref_embeddings.norm(
                    dim=-1, keepdim=True
                )

                # -- get ref image embedding --
                ref_out = ref_model(ref_images, None)
                ref_out = ref_out / ref_out.norm(dim=-1, keepdim=True)

            # -- get image embedding --
            ref_out_current = model(ref_images, None)
            ref_out_current = ref_out_current / ref_out_current.norm(
                dim=-1, keepdim=True
            )

            # -- loss --
            logits_current = logit_scale.exp() * ref_out_current @ ref_embeddings.t()
            logits_ref = logit_scale.exp() * ref_out @ ref_embeddings.t()
            loss_ZSCL = distillation(logits_ref, logits_current, T=args.T)

            # feature-space mse
            if args.feature_mse:
                mse_loss = torch.nn.MSELoss()
                loss += mse_loss(ref_out, ref_out_current)

            # -- final loss --
            if args.image_loss:
                if args.weight_adjust:
                    loss = loss + 0.5 * loss_ZSCL 
                else:
                    loss = loss + 1.0 * loss_ZSCL 

            # transpose loss
            if args.text_loss:
                logits_current_2 = logits_current.t()
                logits_ref_2 = logits_ref.t()
                loss_ZSCL_2 = distillation(logits_ref_2, logits_current_2, T=args.T)
                if args.weight_adjust:
                    loss += 0.5 * loss_ZSCL_2
                else:
                    loss += loss_ZSCL_2
            
            if args.ablation_loss_2:
                logits_img_current = logit_scale.exp() * ref_out_current @ ref_out_current.t()
                logits_img_ref = logit_scale.exp() * ref_out @ ref_out.t()
                logits_img_current -= torch.diag(logits_img_current.diag() + 1e4)
                logits_img_ref -= torch.diag(logits_img_ref.diag() + 1e4)
                loss_ZSCL_3 = distillation(logits_img_ref, logits_img_current, T=args.T)
                if args.weight_adjust:
                    loss += 0.5 * loss_ZSCL_3
                else:
                    loss += loss_ZSCL_3


        # update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # we
        if (args.we or args.moving_avg or args.we_wise) and iteration % args.avg_freq == 0:
            we_n += 1
            if args.moving_avg:
                if args.mv_avg_model == "t":
                    next_we_model = copy.deepcopy(model.module)
                    moving_avg(model.module, we_model, args.mv_avg_decay)
                    we_model = next_we_model.cuda()
                else: ### args.moving_avg_model == "n" or "zeroshot"
                    moving_avg(model.module, we_model, args.mv_avg_decay)
            elif args.we:
                merge_we(model.module, we_model, we_n)
            else:
                wise_we(model.module, we_model, we_n, model_fix, args.we_wise_alpha)

        # evaluation
        if iteration % loss_interval == 0:
            print("Loss:", loss.item())
            if args.method == "ZSCL":
                print("Loss ZSCL:", loss_ZSCL.item())
            if args.l2 > 0:
                print("Loss L2:", loss_l2.item())

    if args.wise_merge:
        alpha = args.wise_ft_alpha
        if args.wise_ft_model == "zeroshot":
            wise_ft_model, _, _ =  clip.load(args.model, jit=False)
        else:
            wise_ft_model = copy.deepcopy(model_fix)
        wise_ft_model.cuda()
        for param_q, param_k in zip(model.module.parameters(), wise_ft_model.parameters()):
            param_q.data = param_q.data * alpha + param_k.data * (1 - alpha)



    # Saving model
    if args.save is not None:
        if args.we or args.we_wise:
            to_save_model = we_model
        else:
            to_save_model = model.module
        # to_save_model = model.module
        path = os.path.join(args.save, f"{args.train_dataset}.pth")
        utils.torch_save(to_save_model, path)
