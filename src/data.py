import librosa, torchaudio, torch 
import torch.nn as nn
import torch.nn.functional as F

def convert_to_contiguous(lst):
    return {value: index for index, value in enumerate(sorted(set(lst)))}
    
def sample_noise(example):
    if example["label"] == "_silence_":
        random_offset = randint(0, len(example["audio"]["array"]) - example["audio"]["sampling_rate"] - 1)
        example["audio"]["array"] = example["audio"]["array"][random_offset : random_offset + example["audio"]["sampling_rate"]]
    return example

def preprocess_audio_batch(batch):
    batch = [sample_noise(example) for example in batch]
    audio_arrays = [example["audio"]["array"] for example in batch]
    sampling_rates = [example["audio"]["sampling_rate"] for example in batch]
    labels = [example["label"] for example in batch]

    batch_mfccs = []
    for audio, sr in zip(audio_arrays, sampling_rates):
        waveform = torch.tensor(audio, dtype=torch.float32)

        target_length = sr
        if waveform.size(0) > target_length:
            waveform = waveform[:target_length]
        else:
            waveform = F.pad(waveform, (0, target_length - waveform.size(0)))
        waveform = waveform.unsqueeze(0)

        mfccs = torchaudio.transforms.MFCC(
            sample_rate=sr,
            n_mfcc=13,
            melkwargs={
                "n_fft": 1024,
                "hop_length": int(sr * 0.01), # 10ms shift
                "win_length": int(sr * 0.03), # 30ms window
                "n_mels": 23,
                "center": False
            }
        )(waveform)
        batch_mfccs.append(mfccs)

    batch_mfccs = torch.stack(batch_mfccs)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return batch_mfccs, labels