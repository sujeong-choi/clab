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
    parser = argparse.ArgumentParser(description='Method for converting onnx model to tflite model')
  
    # make onnx model 
    parser.add_argument('--convertonnx','-mo',default=False, action='store_true', help='convert pytorch model to onnx model')
    parser.add_argument('--onnx_fname','-ofname',type = str, default = '', help = 'Onnx file name')
    
    return parser

if __name__ == '__main__':
    main()


# python onnx2tflite.py -mo -ofname data/out.onnx