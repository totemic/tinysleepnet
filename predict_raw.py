import argparse
import glob
import importlib
import os
import numpy as np
import shutil
import sklearn.metrics as skmetrics
import tensorflow as tf
from scipy.signal import resample

from data import load_data, get_subject_files
from model import TinySleepNet
from minibatching import (iterate_minibatches,
                          iterate_batch_seq_minibatches,
                          iterate_batch_multiple_seq_minibatches)
from utils import (get_balance_class_oversample,
                   print_n_samples_each_class,
                   save_seq_ids,
                   load_seq_ids)
from logger import get_logger
from pathlib import Path
import re

def load_data(eeg_path):
    # = '/nfs/homes/prince/ml/sleep_staging/session_export/D03/A0004/1625897580.406/eeg_1625897580.406.npz'):
    zdata = np.load(eeg_path)
    eeg_ts = zdata['ts_array']
    eeg_data = zdata['eeg_array']
    eeg_val = zdata['valid_array']
    wlen_s = 30
    wlen = 256 * wlen_s
    #
    crop = -1 * (eeg_data.shape[0]%wlen)
    m_aug_data = eeg_data[:crop]
    re_aug_data = m_aug_data.reshape(int(m_aug_data.shape[0]/wlen),wlen,4)
    #
    xll=[]
    for kk in range(4): ## MUSE 4-electrodes
        ore_aug_data = re_aug_data[:,:,kk]
        ## downsample to 100hz
        resampled_aug = np.array([resample(x-np.mean(x), 3000) for x in ore_aug_data])
        xx = resampled_aug.reshape(resampled_aug.shape[0], resampled_aug.shape[1], 1, 1)
        xll.append(xx)
    #
    ocx = np.stack(xll)
    ocy = np.zeros(3000)
    return ocx, ocy


def predict(
    config_file,
    model_dir,
    output_dir,
    log_file,
    use_best=True,
):

    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

    # eeg_path = '/nfs/homes/prince/ml/sleep_staging/session_export/'
    eeg_path = './session_export/'

    fll = [str(xx) for xx in Path(eeg_path).rglob('eeg_*.npz')]
    # for fn in fll: print(fn)
    # print(len(fll))

    spec = importlib.util.spec_from_file_location("*", config_file)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    config = config.predict

    # Create output directory for the specified fold_idx
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    # Add dummy class weights
    config["class_weights"] = np.ones(config["n_classes"], dtype=np.float32)

    fold_idx = 0

    model = TinySleepNet(
        config=config,
        output_dir=os.path.join(model_dir, str(fold_idx)),
        use_rnn=True,
        testing=True,
        use_best=use_best,
    )

    for fn in fll:
        fno = re.sub('eeg_', 'ds_stages_',fn)
        print('inference>', fn)

        night_x, night_y = load_data(fn)

        oll = []
        for kk in range(4): ## MUSE 4-electrodes
            test_minibatch_fn = iterate_batch_multiple_seq_minibatches(
                [night_x[kk]], [night_y],
                batch_size=config["batch_size"],
                seq_length=config["seq_length"],
                shuffle_idx=None,
                augment_seq=False,
            )
            test_outs = model.evaluate(test_minibatch_fn)
            oll.append(test_outs["test/preds"])

        stages=np.stack(oll)

        print('stages>', fno, stages.shape)
        np.savez(fno, stages=stages)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--model_dir", type=str, default="./out_sleepedf/finetune")
    parser.add_argument("--output_dir", type=str, default="./output/predict")
    parser.add_argument("--log_file", type=str, default="./output/output.log")
    parser.add_argument("--use-best", dest="use_best", action="store_true")
    parser.add_argument("--no-use-best", dest="use_best", action="store_false")
    parser.set_defaults(use_best=False)
    args = parser.parse_args()

    predict(
        config_file=args.config_file,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        log_file=args.log_file,
        use_best=args.use_best,
    )
