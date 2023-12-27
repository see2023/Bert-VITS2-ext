
import os,sys
import torch
sys.path.append('./')
import utils
from text.symbols import symbols
from models import SynthesizerTrn, PosteriorEncoder, Generator
from mel_processing import spectrogram_torch, mel_spectrogram_torch, spec_to_mel_torch
import torchaudio

def get_device():
    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else (
            "mps"
            if sys.platform == "darwin" and torch.backends.mps.is_available()
            else "cpu"
        )
    )
    print("Using device: {}".format(device))
    return device

def load_post_enc_dec_model(hps, model_path = './OUTPUT_MODEL/models/G_3000.pth', device='cpu'):
    # load the model
    print('Loading model from {}'.format(model_path))
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).to(device)
    _ = net_g.eval()

    _ = utils.load_checkpoint(model_path, net_g, None, skip_optimizer=True)
    print('Model loaded')

    return net_g.get_post_enc_dec()

def test_wav_enc_dec(hps, input_file='test_in.wav', output_file='test_out.wav', enc = None, dec = None):
    if enc == None or dec == None:
        enc, dec = load_post_enc_dec_model(hps, device=get_device())
    audio_norm, sampling_rate = torchaudio.load(input_file, frame_offset=0, num_frames=-1, normalize=True, channels_first=True)
    # 短时傅里叶变换， 非 mel普
    spec = spectrogram_torch(audio_norm, hps.data.filter_length,
        hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
        center=False)
    spec = spec.to(device=get_device())
    audio_norm = audio_norm.unsqueeze(0)
    print('audio_norm.shape: ', audio_norm.shape, 'spec.shape', spec.shape,  'file: ', input_file)
    x_lengths = torch.clamp_min(torch.sum(spec, [1, 2]), 1).long()
    z, m_q, logs_q, y_mask = enc(spec, x_lengths=x_lengths, g=None)
    print('z.shape: ', z.shape)
    y = dec(z)
    print('y.shape: ', y.shape)
    y = y.squeeze(0).data.cpu()
    #save y to output_file
    torchaudio.save(output_file, y, sampling_rate)
    print('output_file: ', output_file, 'saved')


if __name__ == '__main__':
    hps = utils.get_hparams_from_file('./configs/config.json')
    # test_wav_enc_dec(hps)