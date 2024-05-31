import torch
import os
import argparse
from glob import glob
import soundfile as sf
from torchaudio.compliance.kaldi import mfcc
# from osdc.utils.oladd import overlap_add
from scipy.signal import get_window
import numpy as np
from osdc.features.ola_feats import compute_feats_windowed
import yaml
from train import OSDC_AMI

parser = argparse.ArgumentParser("Single-Channel inference, average logits")
parser.add_argument("--exp_dir", type=str, default="../exp/tcn")
parser.add_argument("--checkpoint_name", type=str, default="checkpoints/epoch=97-step=59388.ckpt")
parser.add_argument("--wav_dir", type=str, default="F:/data/DIHARD3/preprocess/eval")
parser.add_argument("--out_dir", type=str, default="../exp/tcn/preds/checkpoints/epoch=97-step=59388.ckpt")
parser.add_argument("--gpus", type=str, default="0")
parser.add_argument("--window_size", type=int, default=1600)
parser.add_argument("--lookahead", type=int, default=800)
parser.add_argument("--lookbehind", type=int, default=800)
parser.add_argument("--regex", type=str, default="")

def plain_single_file_predict(model, wav_dir, train_configs, out_dir, window_size=400, lookahead=200, lookbehind=200, regex=""):

    model = model.eval().cuda()
    wavs = glob(os.path.join(wav_dir, "**/*{}*.wav".format(regex)), recursive=True)

    assert len(wavs) > 0, "No file found"

    for wav in wavs:
        print("Processing File {}".format(wav))
        audio, _ = sf.read(wav)


        if train_configs["feats"]["type"] in ["mfcc_kaldi", "wavlm"]:
            feats_func = lambda x: mfcc(torch.from_numpy(x.astype("float32").reshape(1, -1)),
                                             **train_configs["mfcc_kaldi"]).transpose(0, 1)
        else:
            raise NotImplementedError

        tot_feats = compute_feats_windowed(feats_func, audio)
        tot_feats = tot_feats.detach().cpu().numpy()
        pred_func = lambda x : model(torch.from_numpy(x).unsqueeze(0).cuda()).detach().cpu().numpy()
        preds = overlap_add(tot_feats, pred_func, audio, window_size, window_size // 2, lookahead=lookahead, lookbehind=lookbehind)
        out_file = os.path.join(out_dir, wav.split("\\")[-1].split(".wav")[0] + ".logits")
        np.save(out_file, preds)

def overlap_add(tensor, function, audio, window_size=1600 * 6, stride=1600 * 3, win_type='hann', pad_left=True,
                n_classes=0, lookahead=1600*2, lookbehind=1600):
    # tensor assumed to be B, C , T

    assert len(tensor.shape) >= 2

    if window_size // stride != 2 or not pad_left:
        raise NotImplementedError

    orig_length = tensor.shape[-1] # original length
    if pad_left:
        pad_left = stride
    else:
        raise NotImplementedError

    pad_right = stride + lookahead # always pad the end
    n, r = divmod(orig_length + pad_right, stride)
    if r != 0:
        n = n+1
        pad_right += stride*n - (orig_length + pad_right)

    pad_dims = [(0,0) for x in range(len(tensor.shape[1:]))]
    npad = (*pad_dims, (pad_left, pad_right))
    tensor = np.pad(tensor, npad, mode="constant")
    window = get_window(win_type, window_size)
    # make window same dimension as tensor channels
    reshape_dims = [1]*len(tensor.shape[:-1])
    window = window.reshape((*reshape_dims, -1))

    if n_classes:
        b, ch, t = window
        window = window # TODO
        raise NotImplementedError

    # # first segment
    # temp = function(tensor[..., :window_size + lookahead])[..., :window_size] * window
    #
    # result = [] # will be discarded
    # buff = temp[..., stride:window_size]
    #
    # for i in range(1, n):
    #     temp = np.copy(tensor[..., i*stride - lookbehind: i*stride+ window_size + lookahead][..., lookbehind:window_size + lookbehind])
    #     temp = function(temp)
    #     temp *= window
    #     result.append(temp[..., :stride] + buff)
    #     buff = temp[..., stride:window_size]
    #
    # result = np.concatenate(result, -1)

    STEP_SIZE = 80000
    all_predictions = []
    audio_data = audio

    for i in range(0, len(audio_data), STEP_SIZE):
        block_size = min(STEP_SIZE, len(audio_data) - i)
        block_data = audio_data[i:i + block_size]
        if block_size < STEP_SIZE:
            block_data = np.pad(block_data, (0, STEP_SIZE - block_size), 'constant', constant_values=0)
        prediction = function(block_data)
        all_predictions.append(prediction)
    result = np.concatenate(all_predictions, axis=-1)

    return result[..., :orig_length]

if __name__ == "__main__":
    args = parser.parse_args()
    with open(os.path.join(args.exp_dir, "confs.yml"), "r") as f:
        confs = yaml.load(f, Loader=yaml.FullLoader)
    # test if compatible with lightning
    confs.update(args.__dict__)

    model = OSDC_AMI(confs)
    if confs["checkpoint_name"].startswith("avg"):
        state_dict = torch.load(os.path.join(confs["exp_dir"], confs["checkpoint_name"]),
                                map_location='cpu')

    else:

        state_dict = torch.load(os.path.join(confs["exp_dir"], confs["checkpoint_name"]),
                            map_location='cpu')["state_dict"]

    model.load_state_dict(state_dict)
    model = model.model
    os.makedirs(confs["out_dir"], exist_ok=True)
    plain_single_file_predict(model, confs["wav_dir"],
                              confs, confs["out_dir"], window_size=args.window_size,
                              lookahead=args.lookahead, lookbehind=args.lookbehind, regex=args.regex)
