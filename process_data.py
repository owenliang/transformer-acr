'''
1、mp4->features
2、text->tokens
''' 

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
import torch 
import torchaudio
# download ffmpeg(https://ffmpeg.org/download.html), add it to PATH first
import ffmpy #  conda install ffmpy
import os

def load_metadata(filename):
    with open(filename, 'r') as fp:
        records=[]
        for line in fp:
            records.append(line.strip().split(' ')[0])
    return records

def load_record_txt(metaname):
    txt_filename=f'data/lrs2/{metaname}.txt'
    with open(txt_filename,'r') as fp:
        text=fp.readline().split(':')[1].strip()
    return text

def train_tokenizer(all_metas):
    tokenizer=Tokenizer(BPE())
    trainer=BpeTrainer(vocab_size=1000,special_tokens=['[UNK]','[PAD]','[BOS]','[EOS]'])
    def iter_all_txt():
        for metaname in all_metas:
            yield load_record_txt(metaname)
    tokenizer.train_from_iterator(iter_all_txt(),trainer=trainer,length=len(all_metas))
    return tokenizer

def load_sample(metaname):
    sample_file=f'dataset/{metaname}.pt'
    return torch.load(sample_file)

def load_tokenizer():
    return Tokenizer.from_file('tokenizer.json')

def process_data(all_metas,tokenizer):
    samples=0
    for metaname in all_metas:
        txt_filename=f'data/lrs2/{metaname}.txt'
        mp4_filename=f'data/lrs2/{metaname}.mp4'
        sample_file=f'dataset/{metaname}.pt'
        if os.path.exists(sample_file): 
            continue
        with open(txt_filename,'r') as fp:
            text=fp.readline().split(':')[1].strip()
            tokens=tokenizer.encode(f'[BOS]{text}[EOS]')
        wav_filename=mp4_filename.replace('.mp4','.wav')
        ff=ffmpy.FFmpeg(inputs={mp4_filename:None},outputs={wav_filename:None})
        ff.run()
        waveform,sample_rate=torchaudio.load(wav_filename,backend='soundfile')  # pip install PySoundFile on windows, pip install soundfile on linux
        audio_features=torchaudio.compliance.kaldi.fbank(waveform*32768,num_mel_bins=80) # waveform first reverted to int16(single channel)
        print(f'filename:{wav_filename} waveform:{waveform.shape} sample_rate:{sample_rate} audio_features:{audio_features.shape}')
        os.makedirs(os.path.dirname(sample_file),exist_ok=True)
        sample={'audio_features':audio_features,'sample_rate':sample_rate,'tokens':tokens}
        torch.save(sample,sample_file)
    return samples
    
if __name__=='__main__':
    train_metas=load_metadata('data/train.txt')
    val_metas=load_metadata('data/val.txt')
    test_metas=load_metadata('data/test.txt')
    all_metas=set(train_metas+val_metas+test_metas)
    print(f'train_metas:{len(train_metas)} val_metas:{len(val_metas)} test_metas:{len(test_metas)} all_metas:{len(all_metas)}')
    
    # train tokenizer
    if not os.path.exists('tokenizer.json'):
        tokenizer=train_tokenizer(all_metas)
        tokenizer.save('tokenizer.json',pretty=True)
    else:
        tokenizer=Tokenizer.from_file('tokenizer.json')
    print(f'tokenizer vocab_size:{tokenizer.get_vocab_size()} encode("hello world"): {tokenizer.encode("hello world")}')    
    
    # pre-process data
    samples=process_data(all_metas,tokenizer)
    print(f'new processed samples:{samples}')