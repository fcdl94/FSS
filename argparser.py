import argparse
import task


def modify_command_options(opts):
    if opts.dataset == 'voc':
        opts.num_classes = 21

    if not opts.visualize:
        opts.sample_num = 0

    opts.no_cross_val = not opts.cross_val
    opts.pooling = round(opts.crop_size / opts.output_stride)
    opts.crop_size_test = opts.crop_size if opts.crop_size_test is None else opts.crop_size_test

    return opts


def get_argparser():
    parser = argparse.ArgumentParser()

    # Performance Options
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--random_seed", type=int, default=42,
                        help="random seed (default: 42)")
    parser.add_argument("--num_workers", type=int, default=1,
                        help='number of workers (default: 1)')
    parser.add_argument('--opt_level', type=str, choices=['O0', 'O1', 'O2', 'O3'], default='O0')

    # Dataset Options
    parser.add_argument("--data_root", type=str, default="data",
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")

    # Task Options
    parser.add_argument("--step", type=int, default=0,
                        help="Step (0 is base)")
    parser.add_argument("--task", type=str, default="15-5", choices=task.get_task_list(),
                        help="Task to be executed (default: 15-5)")
    parser.add_argument("--nshot", type=int, default=5,
                        help="If step>0, the shot to use for FSL (Def=5)")
    parser.add_argument("--ishot", type=int, default=0,
                        help="First index where to sample shots")
    parser.add_argument("--input_mix", default="novel", choices=['novel', 'both'],
                        help="Which class to use for FSL")

    # Train Options
    parser.add_argument("--epochs", type=int, default=30,
                        help="epoch number (default: 30)")

    parser.add_argument("--fix_bn", action='store_true', default=False,
                        help='fix batch normalization during training (default: False)')
    parser.add_argument("--batch_size", type=int, default=4,
                        help='batch size (default: 4)')
    parser.add_argument("--crop_size", type=int, default=512,
                        help="crop size (default: 512)")
    parser.add_argument("--crop_size_test", type=int, default=None,
                        help="test crop size (default: = --crop_size)")

    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_head", type=float, default=1,
                        help="learning rate scaler for ASPP (default: 1)")
    parser.add_argument("--lr_cls", type=float, default=1,
                        help="learning rate scaler for classifier (default: 1)")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help='momentum for SGD (default: 0.9)')
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')

    parser.add_argument("--lr_policy", type=str, default='poly',
                        choices=['poly', 'step'], help="lr schedule policy (default: poly)")
    parser.add_argument("--lr_decay_step", type=int, default=5000,
                        help="decay step for stepLR (default: 5000)")
    parser.add_argument("--lr_decay_factor", type=float, default=0.1,
                        help="decay factor for stepLR (default: 0.1)")
    parser.add_argument("--lr_power", type=float, default=0.9,
                        help="power for polyLR (default: 0.9)")

    # Logging Options
    parser.add_argument("--logdir", type=str, default='./logs',
                        help="path to Log directory (default: ./logs)")
    parser.add_argument("--name", type=str, default='Experiment',
                        help="name of the experiment - to append to log directory (default: Experiment)")
    parser.add_argument("--sample_num", type=int, default=4,
                        help='number of samples for visualization (default: 0)')
    parser.add_argument("--debug",  action='store_true', default=False,
                        help="verbose option")
    parser.add_argument("--visualize",  action='store_false', default=True,
                        help="visualization on tensorboard (def: Yes)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=1,
                        help="epoch interval for eval (default: 1)")

    # Segmentation Architecture Options
    parser.add_argument("--backbone", type=str, default='resnet101',
                        choices=['resnet50', 'resnet101'], help='backbone for the body (def: resnet50)')
    parser.add_argument("--output_stride", type=int, default=16,
                        choices=[8, 16], help='stride for the backbone (def: 16)')
    parser.add_argument("--no_pretrained", action='store_true', default=False,
                        help='Wheather to use pretrained or not (def: True)')
    parser.add_argument("--norm_act", type=str, default="iabn_sync",
                        choices=['iabn_sync', 'iabn', 'abn', 'std'], help='Which BN to use (def: abn_sync')
    parser.add_argument("--fusion-mode", metavar="NAME", type=str, choices=["mean", "voting", "max"], default="mean",
                        help="How to fuse the outputs. Options: 'mean', 'voting', 'max'")
    parser.add_argument("--relu", default=False, action='store_true',
                        help='Use this to enable last BN+ReLU on Deeplab-v3 (def. False)')

    # Test and Checkpoint options
    parser.add_argument("--test",  action='store_true', default=False,
                        help="Whether to train or test only (def: train and test)")
    parser.add_argument("--ckpt", default=None, type=str,
                        help="path to trained model. Leave it None if you want to retrain your model")
    parser.add_argument("--continue_ckpt", default=False, action='store_true',
                        help="Restart from the ckpt. Named taken automatically from method name.")
    parser.add_argument("--ckpt_interval", type=int, default=1,
                        help="epoch interval for saving model (default: 1)")
    parser.add_argument("--cross_val", action='store_true', default=False,
                        help="If validate on training or on validation (default: Train)")

    parser.add_argument("--step_ckpt", default=None, type=str,
                        help="path to trained model at previous step. Leave it None if you want to use def path")

    # Method
    parser.add_argument("--method", type=str, default='FT',
                        choices=['FT', 'SPN', 'COS', 'CFTC', 'WI', 'AMP', 'FTC', 'MIB', 'LWF', 'MIB-SPN', 'MIB-WI'],
                        help="The method you want to use.")
    parser.add_argument("--embedding", type=str, default="fastnvec", choices=['word2vec', 'fasttext', 'fastnvec'])
    parser.add_argument("--amp_alpha", type=float, default=1.,
                        help='Alpha value for the proxy adaptation.')

    return parser
