import librosa
import argparse
import pandas as pd
import numpy as np
import glob
import torch
import torchaudio
import torchvision
from PIL import Image

# extract_processing
parser = argparse.ArgumentParser()
parser.add_argument("--sampling_rate", default=16000, type=int)

def extract_spectogram(name, values, clip, target):

    num_channels = 3
    window_sizes = [25, 50, 100]
    hop_sizes = [10, 25, 50]
    specs = []
    for i in range(num_channels):
        window_length = int(round(window_sizes[1] * args.sampling_rate / 1000))
        hop_length = int(round(hop_sizes[1] * args.sampling_rate / 1000))
        clip = torch.Tensor(clip)
        # 사람이 들을 수 있는 경계값으로 스케일 해주는 작업
        spec = torchaudio.transforms.MelSpectrogram(sample_rate=args.sampling_rate, n_fft=4410,
                                                   win_length=window_length, hop_length=hop_length, n_mels=128)(clip)
        eps = 1e-6
        spec = spec.numpy()
        spec = np.log(spec + eps)
        spec = np.asarray(torchvision.transforms.Resize((128, 250))(Image.fromarray(spec)))
        specs.append(spec)
    new_entry = {}
    audio_array = np.array(specs)
    np_name = './data/audios/save_np/{}.npy'.format(name)
    np.save(np_name, audio_array)
    new_entry["audio"] = clip.numpy()
    new_entry["values"] = np_name
    new_entry["target"] = target
    values.append(new_entry)
    return values

def extract_features():
    values = []
    paths = glob.glob("./data/audios/*")
    for path in paths:
        label = path.split('\\')[-1]
        if label == 'bellypain':
            target = 0
        elif label == 'burping':
            target = 1
        elif label == 'discomfort':
            target = 2
        elif label == 'hungry':
            target = 3
        elif label == 'tired':
            target = 4
        elif label == 'save_np':
            # 정답 정하지 않음
            continue
        else:
            print("not found label")
            exit()

        file_list = glob.glob(path + '/*.wav')
        for file in file_list:
            # librosa.load return value -1 ~ 1로 정규화 되서 값이 나옴
            clip, sr = librosa.load(file, sr=args.sampling_rate)
            extract_spectogram(file.split('\\')[-1], values, clip, target)

    import random

    random.shuffle(values)

    df = pd.DataFrame(values)

    df.to_csv("./data/files/total_audio_list.csv")

    train_df = df.iloc[:int(len(df) * 0.8)]
    train_df.to_csv("./data/files/train.csv")
    val_df = df.iloc[:int(len(df) * 0.8):]
    val_df.to_csv("./data/files/val.csv")

    print("end processing")

if __name__ == "__main__":
    args = parser.parse_args()
    extract_features()