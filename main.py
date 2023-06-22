from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch
import numpy as np
import random
import os
from metrics import compute_metrics, tensor_text_to_video_metrics, tensor_video_to_text_sim
import time
import argparse
from transformers import BertTokenizer
from modules.model import CLIP


from util import parallel_apply, get_logger
from dataloaders.data_dataloaders import DATALOADER_DICT


global logger


def get_args(description='CLIP4Clip on Retrieval Task'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_pretrain", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")

    parser.add_argument('--train_csv', type=str,
                        default='msrvtt_data/MSRVTT_JSFUSION_test.csv', help='')
    parser.add_argument('--val_csv', type=str,
                        default='msrvtt_data/train.csv', help='')
    parser.add_argument('--data_path', type=str,
                        default='data/caption.pickle', help='data pickle file path')
    parser.add_argument('--features_path', type=str,
                        default='data/videos_feature.pickle', help='feature path')

    parser.add_argument('--num_thread_reader', type=int, default=1, help='')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=20,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int,
                        default=256, help='batch size')
    parser.add_argument('--batch_size_val', type=int,
                        default=3500, help='batch size eval')
    parser.add_argument('--lr_decay', type=float, default=0.9,
                        help='Learning rate exp epoch decay')
    parser.add_argument('--n_display', type=int, default=100,
                        help='Information display frequence')
    parser.add_argument('--video_dim', type=int, default=1024,
                        help='video feature dimension')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_words', type=int, default=20, help='')
    parser.add_argument('--max_frames', type=int, default=100, help='')
    parser.add_argument('--feature_framerate', type=int, default=1, help='')
    parser.add_argument('--margin', type=float,
                        default=0.1, help='margin for loss')
    parser.add_argument('--hard_negative_rate', type=float,
                        default=0.5, help='rate of intra negative sample')
    parser.add_argument('--negative_weighting', type=int,
                        default=1, help='Weight the loss for intra negative')
    parser.add_argument('--n_pair', type=int, default=1,
                        help='Num of pair to output from data loader')

    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--n_gpu', type=int, default=1,
                        help="Changed in the execute process.")

    parser.add_argument("--local_rank", default=0,
                        type=int, help="distribted training")
    parser.add_argument("--rank", default=0, type=int,
                        help="distribted training")
    parser.add_argument('--coef_lr', type=float, default=1.,
                        help='coefficient for bert branch.')

    parser.add_argument('--loose_type', action='store_true',
                        help="Default using tight type for retrieval.")
    parser.add_argument('--expand_msrvtt_sentences',
                        action='store_true', help="")

    parser.add_argument('--freeze_layer_num', type=int,
                        default=0, help="Layer NO. of CLIP need to freeze.")

    parser.add_argument("--pretrained_clip_name", default="ViT-B/32",
                        type=str, help="Choose a CLIP version")

    args = parser.parse_args()

    args.output_dir = "output"
    args.batch_size_val = 64
    args.do_eval = True

    return args


def set_seed_logger(args):
    global logger
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    world_size = 20
    torch.cuda.set_device(args.local_rank)
    args.world_size = world_size
    rank = 1
    args.rank = rank

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logger = get_logger(os.path.join(args.output_dir, "log.txt"))

    if args.local_rank == 0:
        logger.info("Effective parameters:")
        for key in sorted(args.__dict__):
            logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args


def init_device(args, local_rank):
    global logger

    device = torch.device(
        'cuda' if torch.cuda.is_available() else "cpu", local_rank)

    n_gpu = 1
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu

    if args.batch_size % args.n_gpu != 0 or args.batch_size_val % args.n_gpu != 0:
        raise ValueError("Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
            args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu))

    return device, n_gpu


def init_model(args, device, n_gpu, local_rank):

    model = CLIP()

    model.to(device)

    return model


def save_model(epoch, args, model, optimizer, tr_loss, type_name=""):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(
        args.output_dir, "pytorch_model.bin.{}{}".format("" if type_name == "" else type_name+".", epoch))
    optimizer_state_file = os.path.join(
        args.output_dir, "pytorch_opt.bin.{}{}".format("" if type_name == "" else type_name+".", epoch))
    torch.save(model_to_save.state_dict(), output_model_file)
    torch.save({
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': tr_loss,
    }, optimizer_state_file)
    logger.info("Model saved to %s", output_model_file)
    logger.info("Optimizer saved to %s", optimizer_state_file)
    return output_model_file


def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer, scheduler, global_step, local_rank=0):
    global logger
    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    start_time = time.time()
    total_loss = 0

    for step, batch in enumerate(train_dataloader):
        if n_gpu == 1:
            # multi-gpu does scattering it-self
            batch = tuple(t.to(device=device, non_blocking=True)
                          for t in batch)

        input_ids, input_mask, segment_ids, video, video_mask = batch
        loss = model(input_ids, segment_ids, input_mask, video, video_mask)

        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()

        total_loss += float(loss)
        if (step + 1) % args.gradient_accumulation_steps == 0:

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if scheduler is not None:
                scheduler.step()  # Update learning rate schedule

            optimizer.step()
            optimizer.zero_grad()

            # https://github.com/openai/CLIP/issues/46
            if hasattr(model, 'module'):
                torch.clamp_(model.module.clip.logit_scale.data,
                             max=np.log(100))
            else:
                torch.clamp_(model.logit_scale.data, max=np.log(100))

            global_step += 1
            if global_step % log_step == 0 and local_rank == 0:
                logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, Time/step: %f", epoch + 1,
                            args.epochs, step + 1,
                            len(train_dataloader), "-".join([str('%.9f' % itm)
                                                             for itm in sorted(list(set(optimizer.get_lr())))]),
                            float(loss),
                            (time.time() - start_time) / (log_step * args.gradient_accumulation_steps))
                start_time = time.time()

    total_loss = total_loss / len(train_dataloader)
    return total_loss, global_step


def _run_on_single_gpu(model, batch_list_t, batch_list_v, batch_sequence_output_list, batch_visual_output_list):
    sim_matrix = []
    for idx1, b1 in enumerate(batch_list_t):
        input_mask, segment_ids, *_tmp = b1
        sequence_output = batch_sequence_output_list[idx1]
        each_row = []
        for idx2, b2 in enumerate(batch_list_v):
            video_mask, *_tmp = b2
            visual_output = batch_visual_output_list[idx2]
            b1b2_logits, *_tmp = model.get_similarity_logits(sequence_output, visual_output, input_mask, video_mask,
                                                             loose_type=model.loose_type)
            b1b2_logits = b1b2_logits.cpu().detach().numpy()
            each_row.append(b1b2_logits)
        each_row = np.concatenate(tuple(each_row), axis=-1)
        sim_matrix.append(each_row)
    return sim_matrix


def eval_epoch(args, model, test_dataloader, device, n_gpu):

    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

    # #################################################################
    # below variables are used to multi-sentences retrieval
    # multi_sentence_: important tag for eval
    # cut_off_points: used to tag the label when calculate the metric
    # sentence_num: used to cut the sentence representation
    # video_num: used to cut the video representation
    # #################################################################
    multi_sentence_ = False
    cut_off_points_, sentence_num_, video_num_ = [], -1, -1
    # if hasattr(test_dataloader.dataset, 'multi_sentence_per_video') \
    #         and test_dataloader.dataset.multi_sentence_per_video:
    #     multi_sentence_ = True
    #     cut_off_points_ = test_dataloader.dataset.cut_off_points
    #     sentence_num_ = test_dataloader.dataset.sentence_num
    #     video_num_ = test_dataloader.dataset.video_num
    #     cut_off_points_ = [itm - 1 for itm in cut_off_points_]

    # if multi_sentence_:
    #     logger.warning("Eval under the multi-sentence per video clip setting.")
    #     logger.warning("sentence num: {}, video num: {}".format(
    #         sentence_num_, video_num_))

    model.eval()
    with torch.no_grad():
        batch_list_t = []
        batch_list_v = []
        batch_sequence_output_list, batch_visual_output_list = [], []
        total_video_num = 0

        # ----------------------------
        # 1. cache the features
        # ----------------------------
        print(next(iter(test_dataloader)))
        for bid, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, video = batch

            if multi_sentence_:
                # multi-sentences retrieval means: one clip has two or more descriptions.
                b, *_t = video.shape
                sequence_output = model.get_sequence_output(
                    input_ids, segment_ids, input_mask)
                batch_sequence_output_list.append(sequence_output)
                batch_list_t.append((input_mask, segment_ids,))

                s_, e_ = total_video_num, total_video_num + b
                filter_inds = [
                    itm - s_ for itm in cut_off_points_ if itm >= s_ and itm < e_]

                if len(filter_inds) > 0:
                    video, video_mask = video[filter_inds,
                                              ...], video_mask[filter_inds, ...]
                    visual_output = model.get_visual_output(video, video_mask)
                    batch_visual_output_list.append(visual_output)
                    batch_list_v.append((video_mask,))
                total_video_num += b
            else:
                sequence_output, visual_output = model(
                    input_ids, segment_ids, input_mask, video)

                batch_sequence_output_list.append(sequence_output)
                batch_list_t.append((input_mask, segment_ids,))

                batch_visual_output_list.append(visual_output)
                batch_list_v.append((video_mask,))

            print("{}/{}\r".format(bid, len(test_dataloader)), end="")

        # ----------------------------------
        # 2. calculate the similarity
        # ----------------------------------

        sim_matrix = _run_on_single_gpu(
            model, batch_list_t, batch_list_v, batch_sequence_output_list, batch_visual_output_list)
        sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)

    logger.info("sim matrix size: {}, {}".format(
        sim_matrix.shape[0], sim_matrix.shape[1]))
    tv_metrics = compute_metrics(sim_matrix)
    vt_metrics = compute_metrics(sim_matrix.T)
    logger.info(
        '\t Length-T: {}, Length-V:{}'.format(len(sim_matrix), len(sim_matrix[0])))

    logger.info("Text-to-Video:")
    logger.info('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['MR'], tv_metrics['MeanR']))
    logger.info("Video-to-Text:")
    logger.info('\t>>>  V2T$R@1: {:.1f} - V2T$R@5: {:.1f} - V2T$R@10: {:.1f} - V2T$Median R: {:.1f} - V2T$Mean R: {:.1f}'.
                format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['MR'], vt_metrics['MeanR']))

    R1 = tv_metrics['R1']
    return R1


def main():
    global logger
    args = get_args()
    args = set_seed_logger(args)
    device, n_gpu = init_device(args, args.local_rank)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    model = init_model(args, device, n_gpu, args.local_rank)

    ## ####################################
    # freeze testing
    ## ####################################

    for param in model.text_encoder.parameters():
        param.requires_grad = False

    for param in model.video_encoder.parameters():
        param.requires_grad = False

    ## ####################################
    # dataloader loading
    ## ####################################

    test_dataloader, test_length = None, 0
    val_dataloader, val_length = DATALOADER_DICT["msrvtt"]["val"](
        args, tokenizer, subset="val")

    # report validation results if the ["test"] is None
    if test_dataloader is None:
        test_dataloader, test_length = val_dataloader, val_length

    if args.local_rank == 0:
        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", test_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d", len(test_dataloader))
        logger.info("***** Running val *****")
        logger.info("  Num examples = %d", val_length)

    ## ####################################
    # train and eval
    ## ####################################
    if args.do_train:
        train_dataloader, train_length, train_sampler = DATALOADER_DICT["msrvtt"]["train"](
            args, tokenizer)
        num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                        / args.gradient_accumulation_steps) * args.epochs

        coef_lr = args.coef_lr
        optimizer, scheduler, model = prep_optimizer(
            args, model, num_train_optimization_steps, device, n_gpu, args.local_rank, coef_lr=coef_lr)

        if args.local_rank == 0:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", train_length)
            logger.info("  Batch size = %d", args.batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps *
                        args.gradient_accumulation_steps)

        best_score = 0.00001
        best_output_model_file = "None"
        ## ##############################################################
        # resume optimizer state besides loss to continue train
        ## ##############################################################
        resumed_epoch = 0
        if args.resume_model:
            checkpoint = torch.load(args.resume_model, map_location='cpu')
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            resumed_epoch = checkpoint['epoch']+1
            resumed_loss = checkpoint['loss']

        global_step = 0
        for epoch in range(resumed_epoch, args.epochs):
            train_sampler.set_epoch(epoch)
            tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer,
                                               scheduler, global_step, local_rank=args.local_rank)
            if args.local_rank == 0:
                logger.info("Epoch %d/%s Finished, Train Loss: %f",
                            epoch + 1, args.epochs, tr_loss)

                output_model_file = save_model(
                    epoch, args, model, optimizer, tr_loss, type_name="")

                # Run on val dataset, this process is *TIME-consuming*.
                # logger.info("Eval on val dataset")
                # R1 = eval_epoch(args, model, val_dataloader, device, n_gpu)

                R1 = eval_epoch(args, model, test_dataloader, device, n_gpu)
                if best_score <= R1:
                    best_score = R1
                    best_output_model_file = output_model_file
                logger.info("The best model is: {}, the R1 is: {:.4f}".format(
                    best_output_model_file, best_score))

        # Uncomment if want to test on the best checkpoint
        # if args.local_rank == 0:
        #     model = load_model(-1, args, n_gpu, device, model_file=best_output_model_file)
        #     eval_epoch(args, model, test_dataloader, device, n_gpu)

    elif args.do_eval:
        if args.local_rank == 0:
            eval_epoch(args, model, test_dataloader, device, n_gpu)


if __name__ == "__main__":
    main()
