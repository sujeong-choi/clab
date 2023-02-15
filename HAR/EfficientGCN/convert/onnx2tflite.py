import onnx
import tensorflow as tf
from onnx_tf.backend import prepare

import os, yaml, argparse
from time import sleep

class Onnx2Tflite():
    def start(self,args):
        self.args = args

        onnx_model = onnx.load(self.args.onnx_fname)
        # onnx.checker.check_model(onnx_model)
        # convert onnx to tflite
        onnx.helper.printable_graph(onnx_model.graph)

        tf_model_path = "data/tf_model"
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(tf_model_path)
        print("success to save pb file\n")

        converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
            tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
        ]
        tflite_model = converter.convert()
        tflite_model_path = "data/tflite-model"
        with open(tflite_model_path, 'wb') as f:
            f.write(tflite_model)
        print("success to make and save tflite file\n")


def main():
    # Loading parameters
    parser = init_parser()
    args = parser.parse_args()
    args = update_parameters(parser, args)  # cmd > yaml > default

    # Waiting to run
    sleep(args.delay_hours * 3600)

    # Processing
    if  args.convertonnx:
        if os.path.exists(args.onnx_fname):
            print(f"{args.onnx_fname} 파일이 존재합니다.\n")
            o2t = Onnx2Tflite()
            o2t.start(args)
        else:
            print(f"{args.onnx_fname} 파일이 없습니다. \n")
    else:
        print(f"convertonnx 가 false입니다.  \n")


def init_parser():
    parser = argparse.ArgumentParser(description='Method for Skeleton-based Action Recognition')

    # Setting
    parser.add_argument('--config', '-c', type=str, default='', help='ID of the using config', required=True)
    parser.add_argument('--gpus', '-g', type=int, nargs='+', default=[], help='Using GPUs')
    parser.add_argument('--seed', '-s', type=int, default=1, help='Random seed')
    parser.add_argument('--pretrained_path', '-pp', type=str, default='', help='Path to pretrained models')
    parser.add_argument('--work_dir', '-w', type=str, default='', help='Work dir')
    parser.add_argument('--no_progress_bar', '-np', default=False, action='store_true', help='Do not show progress bar')
    parser.add_argument('--delay_hours', '-dh', type=float, default=0, help='Delay to run')

    # Processing
    parser.add_argument('--debug', '-db', default=False, action='store_true', help='Debug')
    parser.add_argument('--resume', '-r', default=False, action='store_true', help='Resume from checkpoint')
    parser.add_argument('--evaluate', '-e', default=False, action='store_true', help='Evaluate')
    parser.add_argument('--extract', '-ex', default=False, action='store_true', help='Extract')
    parser.add_argument('--visualize', '-v', default=False, action='store_true', help='Visualization')
    parser.add_argument('--generate_data', '-gd', default=False, action='store_true', help='Generate skeleton data')

    # Visualization
    parser.add_argument('--visualization_class', '-vc', type=int, default=0, help='Class: 1 ~ 60, 0 means true class')
    parser.add_argument('--visualization_sample', '-vs', type=int, default=0, help='Sample: 0 ~ batch_size-1')
    parser.add_argument('--visualization_frames', '-vf', type=int, nargs='+', default=[], help='Frame: 0 ~ max_frame-1')

    # Dataloader
    parser.add_argument('--dataset', '-d', type=str, default='', help='Select dataset')
    parser.add_argument('--dataset_args', default=dict(), help='Args for creating dataset')

    # Model
    parser.add_argument('--model_type', '-mt', type=str, default='', help='Args for creating model')
    parser.add_argument('--model_args', default=dict(), help='Args for creating model')
    
    # Optimizer
    parser.add_argument('--optimizer', '-o', type=str, default='', help='Initial optimizer')
    parser.add_argument('--optimizer_args', default=dict(), help='Args for optimizer')

    # LR_Scheduler
    parser.add_argument('--lr_scheduler', '-ls', type=str, default='', help='Initial learning rate scheduler')
    parser.add_argument('--scheduler_args', default=dict(), help='Args for scheduler')

    # Runner & Debug & skeleton file maker
    parser.add_argument('--runner', '-run', default=False, action='store_true', help='Testing runner')
    parser.add_argument('--video', '-vp', type=str, default='', help='videos/url')
    parser.add_argument('--fps', type=int, default=15, help='frame extraction count per sec')
    parser.add_argument('--short-side', type=int, default=480, help='specify the short-side length of the image')
    parser.add_argument('--complexity', type=int, default=1, choices=range(0, 3), help='Complexity of the pose landmark model: 0, 1 or 2. Landmark accuracy as well as inference latency generally go up with the model complexity. Default to 1.')
    parser.add_argument('--visualize_skeleton', '-vsk', default=False, action='store_true', help='Make skeleton added videos')
    parser.add_argument('--label', type=int, default=1, choices=range(120, 123), help='data label. 1:painting, 2: interview, 3:pause, 4:ntu60')
    parser.add_argument('--generate_skeleton_file', '-gs', default=False, action='store_true', help='Make skeleton files from videos')
    parser.add_argument('--finetuning', '-ft',default=False, action='store_true', help='Fine tuning')
    
    # make onnx model 
    parser.add_argument('--convertonnx','-mo',default=False, action='store_true', help='convert pytorch model to onnx model')
    parser.add_argument('--onnx_fname','-ofname',type = str, default = '', help = 'Onnx file name')
    
    return parser


def update_parameters(parser, args):
    if os.path.exists('../configs/{}.yaml'.format(args.config)):
        with open('../configs/{}.yaml'.format(args.config), 'r') as f:
            try:
                yaml_arg = yaml.load(f, Loader=yaml.FullLoader)
            except:
                yaml_arg = yaml.load(f)
            default_arg = vars(args)
            for k in yaml_arg.keys():
                if k not in default_arg.keys():
                    raise ValueError('Do NOT exist this parameter {}'.format(k))
            parser.set_defaults(**yaml_arg)
    else:
        raise ValueError('Do NOT exist this file in \'configs\' folder: {}.yaml!'.format(args.config))
    return parser.parse_args()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()


# python onnx2tflite.py -mo -ofname data/out.onnx -c media20 -g 1