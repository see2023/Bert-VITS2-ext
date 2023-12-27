import os,sys
import torch
import numpy as np
sys.path.append('./')
import utils
from models import VisemesNet
from mel_processing import spectrogram_torch
import torchaudio
from config import config
from visemes_tools import load_post_enc_dec_model, get_device


#测试wav文件到visemes
if __name__ == '__main__':
    # 从入参获取wav文件
    if sys.argv.__len__() < 2:
        print('python wav_to_visemes.py wav_file')
        exit(1)
    wav_file = sys.argv[1]
    if not os.path.exists(wav_file):
        print('wav_file not exists')
        exit(1)
    # load hps
    hps = utils.get_hparams_from_file('./configs/config.json')
    device = get_device()
    # load enc, dec, v_model
    enc, dec = load_post_enc_dec_model(hps, device=device)
    print('net_g loaded')

    net_v = VisemesNet(hps.model.hidden_channels).to(device)
    _ = net_v.eval()
    _ = utils.load_checkpoint(config.webui_config.v_model, net_v, None, skip_optimizer=True)
    print("load v_model from", config.webui_config.v_model)

    # load wav file
    audio_norm, sampling_rate = torchaudio.load(wav_file, frame_offset=0, num_frames=-1, normalize=True, channels_first=True)
    # check sampling_rate == 44100
    if sampling_rate != 44100:
        print('sampling_rate error:', sampling_rate)
        print('ffmpeg -i input.wav -ar 44100 output.wav')
        exit(1)
    spec = spectrogram_torch(audio_norm, hps.data.filter_length,
        hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
        center=False)
    spec = spec.to(device=get_device())
    audio_norm = audio_norm.unsqueeze(0)
    x_lengths = torch.clamp_min(torch.sum(spec, [1, 2]), 1).long()

    # get z
    z, m_q, logs_q, y_mask = enc(spec, x_lengths=x_lengths, g=None)
    print('get z of wav file: ', wav_file)
    visemes_file_path = wav_file[:-4] + '.v.npy'

    # generate visemes
    visemes = net_v(z)
    visemes = visemes.squeeze(0)
    visemes = visemes.transpose(0, 1)
    visemes = visemes.data.cpu().float().numpy()
    print('visemes shape:', visemes.shape)

    # save visemes 
    np.save(visemes_file_path, visemes)
    print('visemes saved to ', visemes_file_path)
