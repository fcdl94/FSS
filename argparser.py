import argparse
import task
from methods import methods


def modify_command_options(opts):
    if not opts.visualize:
        opts.sample_num = 0

    if opts.batch_size == -1:
        if opts.step == 0:
            opts.batch_size = 24
        else:
            opts.batch_size = 10

    if opts.backbone is None:
        opts.backbone = 'resnet101'

    if opts.method == "GIFS":
        opts.method = "WI"
        opts.norm_act = "iabr"
        opts.l2_loss = 0.1 if opts.l2_loss == 0 else opts.l2_loss
    elif opts.method == 'LWF':
        opts.loss_kd = 10 if opts.loss_kd == 0 else opts.loss_kd
        opts.method = "FT"
    elif opts.method == 'MIB':
        opts.mib_kd = 10 if opts.mib_kd == 0 else opts.mib_kd
        opts.mib_ce = True
        opts.init_mib = True
        opts.method = "FT"
    elif opts.method == 'ILT':
        opts.loss_kd = 10 if opts.loss_kd == 0 else opts.loss_kd
        opts.loss_de = 10 if opts.loss_de == 0 else opts.loss_de
        opts.method = "FT"
    elif opts.method == 'RT':
        opts.train_only_novel = True
        opts.method = "FT"
        opts.lr_cls = 10

    if opts.train_only_classifier or opts.train_only_novel:
        opts.freeze = True
        opts.lr_head = 0.

    opts.no_cross_val = not opts.cross_val
    opts.pooling = round(opts.crop_size / opts.output_stride)
    opts.crop_size_test = 500 if opts.dataset == "voc" else 640
    opts.test_batch_size = 1

    return opts


def get_argparser():
    parser = argparse.ArgumentParser()

    # Performance Options
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--random_seed", type=int, default=42,
                        help="random seed (default: 42)")
    parser.add_argument("--num_workers", type=int, default=2,
                        help='number of workers (default: 2)')
    parser.add_argument('--opt_level', type=str, choices=['O0', 'O1', 'O2', 'O3'], default='O0')
    parser.add_argument("--device", type=int, default=None,
                        help='Specify the device you want to use.')

    # Dataset Options
    parser.add_argument("--data_root", type=str, default="data",
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'cts', 'coco', 'coco-stuff'], help='Name of dataset')

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
    parser.add_argument("--iter", type=int, default=None,
                        help="iteration number (default: None)\n THIS OVERWRITE --EPOCHS!")

    parser.add_argument("--fix_bn", action='store_true', default=False,
                        help='fix batch normalization during training (default: False)')
    parser.add_argument("--batch_size", type=int, default=-1,
                        help='batch size (default: 24/10)')
    parser.add_argument("--crop_size", type=int, default=512,
                        help="crop size (default: 512)")
    parser.add_argument("--crop_size_test", type=int, default=None,
                        help="test crop size (default: = --crop_size)")

    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--freeze", action='store_true', default=False,
                        help="Freeze body (default: False)")
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
    parser.add_argument("--sample_num", type=int, default=0,
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
    parser.add_argument("--backbone", type=str, default=None,
                        choices=['resnet50', 'resnet101', 'resnext101'], help='backbone for the body')
    parser.add_argument("--deeplab", type=str, default="v3",
                        choices=['v3', 'v2', 'none'], help='network head')
    parser.add_argument("--output_stride", type=int, default=16,
                        choices=[8, 16], help='stride for the backbone (def: 16)')
    parser.add_argument("--no_pretrained", action='store_true', default=False,
                        help='Wheather to use pretrained or not (def: True)')
    parser.add_argument("--norm_act", type=str, default="iabn_sync",
                        # choices=['riabn_sync', 'riabn_sync2', 'iabn_sync', 'iabn', 'abn', 'rabn', 'ain'],
                        help='Which BN to use (def: iabn_sync')

    parser.add_argument("--n_feat", type=int, default=256,
                        help="Feature size (default: 256)")
    parser.add_argument("--relu", default=False, action='store_true',
                        help='Use this to enable last BN+ReLU on Deeplab-v3 (def. False)')
    parser.add_argument("--no_pooling", default=False, action='store_true',
                        help='Use this to DIS-enable Pooling in Deeplab-v3 (def. False)')
    parser.add_argument("--hnm", default=False, action='store_true',
                        help='Use this to enable Hard Negative Mining (def. False)')
    parser.add_argument("--focal", default=False, action='store_true',
                        help='Use this to enable Focal Loss (def. False)')

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
                        help="If validate on training or on validation (default: Val)")

    parser.add_argument("--step_ckpt", default=None, type=str,
                        help="path to trained model at previous step. Leave it None if you want to use def path")

    # Method
    parser.add_argument("--method", type=str, default='FT',
                        choices=methods, help="The method you want to use.")
    parser.add_argument("--embedding", type=str, default="fastnvec", choices=['word2vec', 'fasttext', 'fastnvec'])
    parser.add_argument("--amp_alpha", type=float, default=0.25,
                        help='Alpha value for the proxy adaptation.')
    parser.add_argument("--mib_ce", default=False, action='store_true',
                        help='Use the MiB classification loss (Def No)')
    parser.add_argument("--init_mib", default=False, action='store_true',
                        help='Use the MiB initialization (Def No)')
    parser.add_argument("--mib_kd", default=0, type=float,
                        help='The MiB distillation loss strength (Def 0.)')
    parser.add_argument("--loss_kd", default=0, type=float,
                        help='The distillation loss strength (Def 0.)')
    parser.add_argument("--l2_loss", default=0, type=float,
                        help='The MSE feature loss strength (Def 0.)')
    parser.add_argument("--loss_de", default=0, type=float,
                        help='The MSE-body feature loss strength (Def 0.)')
    parser.add_argument("--l1_loss", default=0, type=float,
                        help='The L1 feature loss strength (Def 0.)')
    parser.add_argument("--cos_loss", default=0, type=float,
                        help='The feature loss strength (Def 0.)')
    parser.add_argument("--kl_div", default=False, action='store_true',
                        help='Use true KL loss and not the CE loss.')
    parser.add_argument("--dist_warm_start", default=False, action='store_true',
                        help='Use warm start for distillation.')
    parser.add_argument("--born_again", default=False, action='store_true',
                        help='Use born again strategy (use --ckpt as model old).')

    parser.add_argument("--train_only_classifier", action='store_true', default=False,
                        help="Freeze body and head of network (default: False)")
    parser.add_argument("--train_only_novel", action='store_true', default=False,
                        help="Train only the classifier of current step (default: False)")
    parser.add_argument("--bn_momentum", default=None, type=float,
                        help="The BN momentum (Set to 0.1 to update of running stats of ABR.)")

    parser.add_argument("--dyn_lr", default=1., type=float,
                        help='LR for DynWI (Def 1)')
    parser.add_argument("--dyn_iter", default=1000, type=int,
                        help='Iterations for DynWI (Def 1000)')

    # to remove
    parser.add_argument("--pixel_imprinting", action='store_true', default=False,
                        help="Use only a pixel for imprinting when with WI (default: False)")
    parser.add_argument("--weight_mix", action='store_true', default=False,
                        help="When doing WI, sum to proto the mix of old weights (default: False)")
    return parser
