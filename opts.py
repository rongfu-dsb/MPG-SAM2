import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('BIFIT training and inference scripts.', add_help=False)
    parser.add_argument('--lr', default=2e-6, type=float)
    parser.add_argument('--lr_bifit', default=5e-5, type=float)
    parser.add_argument('--lr_bifit_names', default=['mask_linear','mask_product','vmtoken','mmfusion_module','objfusion_module','vtlfusion_module'], type=str, nargs='+')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lr_drop', default=[6, 8], type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    # load the pretrained weights
    parser.add_argument('--pretrained_weights', type=str, default=None,
                        help="Path to the pretrained model.")

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')  # NOTE: must be false

    # * Backbone no use
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--backbone_pretrained', default=None, type=str,
                        help="if use swin backbone and train from scratch, the path to the pretrained weights")
    parser.add_argument('--use_checkpoint', action='store_true',
                        help='whether use checkpoint for swin/video swin backbone')
    parser.add_argument('--dilation', action='store_true',  # DC5
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=4, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=4, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--query_enc_layers', default=1, type=int,
                        help="Number of query encoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_frames', default=6, type=int,
                        help="Number of clip frames for training")
    parser.add_argument('--num_queries', default=5, type=int,
                        help="Number of query slots, all frames share the same queries")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--pre_norm', action='store_true')
    # for text
    parser.add_argument('--freeze_text_encoder', action='store_true')  # default: False

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--mask_dim', default=256, type=int,
                        help="Size of the mask embeddings (dimension of the dynamic mask conv)")
    parser.add_argument('--controller_layers', default=3, type=int,
                        help="Dynamic conv layer number")
    parser.add_argument('--dynamic_mask_channels', default=8, type=int,
                        help="Dynamic conv final channel number")
    parser.add_argument('--no_rel_coord', dest='rel_coord', action='store_false',
                        help="Disables relative coordinates")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_cost_mask', default=2, type=float,
                        help="mask coefficient in the matching cost")
    parser.add_argument('--set_cost_dice', default=5, type=float,
                        help="mask coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=2, type=float)
    parser.add_argument('--dice_loss_coef', default=5, type=float)
    parser.add_argument('--sim_loss_coef', default=2, type=float)
    parser.add_argument('--yues_dice_loss_coef', default=5, type=float)
    parser.add_argument('--yues_mask_loss_coef', default=2, type=float)
    parser.add_argument('--iou_loss_coef', default=0.5, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    # ['ytvos', 'davis', 'a2d', 'jhmdb', 'refcoco', 'refcoco+', 'refcocog', 'all']
    # 'all': using the three ref datasets for pretraining
    parser.add_argument('--dataset_file', default='ytvos', help='Dataset name')
    parser.add_argument('--coco_path', type=str, default='./your_data/coco')
    parser.add_argument('--ytvos_path', type=str,
                        default='./your_data/ref-youtube-vos')
    parser.add_argument('--mevis_path',type=str,default='./your_data/mevis-release')
    parser.add_argument('--davis_path', type=str, default='./your_data/ref-davis')
    parser.add_argument('--a2d_path', type=str, default='./your_data/A2d_sentences') # no use
    parser.add_argument('--jhmdb_path', type=str,
                        default='./your_data/jhmdb_sentences') # no use
    parser.add_argument('--max_skip', default=3, type=int, help="max skip frame number")
    parser.add_argument('--max_size', default=640, type=int, help="max size for the frame")
    parser.add_argument('--binary', action='store_true')
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=8, type=int)

    # test setting
    parser.add_argument('--threshold', default=0.5, type=float)  # binary threshold for mask
    parser.add_argument('--ngpu', default=4, type=int, help='gpu number when inference for ref-ytvos and ref-davis')
    parser.add_argument('--split', default='valid', type=str, choices=['valid', 'test'])
    parser.add_argument('--visualize', action='store_true', help='whether visualize the masks during inference')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    # evf
    parser.add_argument('--vision_pretrained', default="./your_weights/sam2-hiera-large/sam2_hiera_large.pt",type=str) # sam
    parser.add_argument('--encoder_pretrained', default="./your_weights/beit3_large_patch16_224.pth", type=str) #beit
    parser.add_argument('--train_mask_decoder', action= 'store_true', default=True)
    parser.add_argument('--train_prompt_encoder', action= 'store_true', default=True)
    parser.add_argument('--version', default="./your_weights/beit3.spm",type=str) # all
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--precision", default='bf16',type=str)
    parser.add_argument("--sam_scale", default='huge', type=str)
    parser.add_argument("--mm_extractor_scale", default='large',type=str)
    parser.add_argument("--hidden_size", default=1024,type=int)
    parser.add_argument('--inference', action= 'store_true', default=False)
    parser.add_argument('--get_pretrain', action='store_true', default=False)
    parser.add_argument('--clip_length', default=2,type=int)
    return parser


