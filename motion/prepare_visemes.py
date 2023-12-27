import os,sys
import torch
sys.path.append('./')
import utils
from text.symbols import symbols
from models import SynthesizerTrn, PosteriorEncoder, Generator
from mel_processing import spectrogram_torch, mel_spectrogram_torch, spec_to_mel_torch
import torchaudio
from visemes_tools import load_post_enc_dec_model, get_device



# 读取records目录下的  *.wav 音频文件和 *.npy 表情数据[n, 61]，相同的文件名为一组。 
# 前5组[file1.wav, file1.npy]生成训练数据列表 val_visemes.list
# 剩余的组生成测试数据列表 train_visemes.list
def gen_visemes_train_val_list(hps, input_dir='./records/', output_dir = './filelists/'):
    enc, dec = load_post_enc_dec_model(hps, device=get_device())
    print('enc, dec loaded')
    # read all files in input_dir
    files = os.listdir(input_dir)
    # filter wav files
    wav_files = filter(lambda x: x.endswith('.wav'), files)
    wav_files = sorted(wav_files)
    # overwrite the list file
    with open(output_dir + 'val_visemes.list', 'w') as f:
        f.write('')
    with open(output_dir + 'train_visemes.list', 'w') as f:
        f.write('')
    # iterate wav files
    for i, wav_file in enumerate(wav_files):
        # get the corresponding npy file and make sure it exists
        wav_file = input_dir + wav_file
        print('processing wav file: ', wav_file)
        npy_file = wav_file[:-4] + '.npy'
        if not os.path.exists(npy_file):
            print('npy file {} does not exist'.format(npy_file))
            continue
        audio_norm, sampling_rate = torchaudio.load(wav_file, frame_offset=0, num_frames=-1, normalize=True, channels_first=True)
        spec = spectrogram_torch(audio_norm, hps.data.filter_length,
            hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
            center=False)
        spec = spec.to(device=get_device())
        audio_norm = audio_norm.unsqueeze(0)
        x_lengths = torch.clamp_min(torch.sum(spec, [1, 2]), 1).long()
        z, m_q, logs_q, y_mask = enc(spec, x_lengths=x_lengths, g=None)
        print('get z of wav file: ', wav_file)
        z_file_path = wav_file[:-4] + '.z.npy'
        z = z.to(device='cpu')
        # save z
        torch.save(z, z_file_path)
        print('z saved to ', z_file_path)


        # generate the line for the list file
        line = z_file_path + '|' + npy_file + '\n'
        # write the line to the list file
        if i < 5:
            with open(output_dir + 'val_visemes.list', 'a') as f:
                f.write(line)
        else:
            with open(output_dir + 'train_visemes.list', 'a') as f:
                f.write(line)


if __name__ == '__main__':
    hps = utils.get_hparams_from_file('./configs/config.json')
    gen_visemes_train_val_list(hps)    