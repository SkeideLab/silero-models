from turtle import width
import torch
import random
from glob import glob
from omegaconf import OmegaConf
from utils import (init_jit_model,
                   split_into_batches,
                   read_audio,
                   read_batch,
                   prepare_model_input)
import numpy as np
import ipywidgets as widgets
from scipy.io import wavfile
from IPython.display import Audio, display, clear_output
from torchaudio.functional import vad
import glob
import torchaudio
import sys
import os
import textwrap
from os.path import exists
os.system("pip install -q omegaconf torchaudio pydub")


print(os.getcwd())
# %%

os.system("module load ffmpeg")
# silero imports

device = torch.device('cpu')   # you can use any pytorch device
models = OmegaConf.load('models.yml')

# imports for uploading/recording
# %%

# wav to text method

# %%


def wav_to_text(episode, name):
    f = f'test/{episode}/{name}.wav'
    batch = read_batch([f])
    input = prepare_model_input(batch, device=device)
    output = model(input)
    return decoder(output[0].cpu())


# @markdown { run: "auto" }

language = "German"  # @param ["English", "German", "Spanish"]

print(language)
if language == 'German':
    model, decoder = init_jit_model(
        models.stt_models.de.latest.jit, device=device)
elif language == "Spanish":
    model, decoder = init_jit_model(
        models.stt_models.es.latest.jit, device=device)
else:
    model, decoder = init_jit_model(
        models.stt_models.en.latest.jit, device=device)

# @markdown { run: "auto" }

use_VAD = "Yes"  # @param ["Yes", "No"]

# @markdown Either record audio from microphone or upload audio from file (.mp3 or .wav) { run: "auto" }

# @param ["Record", "Upload (.mp3 or .wav)"]
record_or_upload = "Upload (.mp3 or .wav)"
record_seconds = 4  # @param {type:"number", min:1, max:10, step:1}
sample_rate = 16000


def _apply_vad(audio, boot_time=0, trigger_level=9, **kwargs):
    print('\nVAD applied\n')
    vad_kwargs = dict(locals().copy(), **kwargs)
    vad_kwargs['sample_rate'] = sample_rate
    del vad_kwargs['kwargs'], vad_kwargs['audio']
    audio = vad(torch.flip(audio, ([0])), **vad_kwargs)
    return vad(torch.flip(audio, ([0])), **vad_kwargs)


def _recognize(audio, episode):
    display(Audio(audio, rate=sample_rate, autoplay=True))
    #name = audio_file.split(".wav")[0]
    name = audio_file.split("/")[-1]
    name = name.split(".wav")[0]
    if use_VAD == "Yes":
        audio = _apply_vad(audio)
    wavfile.write(f'test/{episode}/{name}.wav', 16000,
                  (32767*audio).numpy().astype(np.int16))
    transcription = wav_to_text(episode, name)

    f = open(f'transcriptions/{episode}/{name}.txt', 'w')
    print('\n\nTRANSCRIPTION:\n', file=f)
    #print(transcription, file=f, width=50)
    wrapper = textwrap.TextWrapper(width=50)
    transcription = wrapper.wrap(text=transcription)
    print("\n".join(transcription), file=f)
    f.close()


def _upload_audio(b):
    # lclear_output()
    # audio = upload_audio()
    audio = read_audio()
    _recognize(audio)
    return audio


# %%
os.chdir('/cobra/u/skapoor/git/COSMIC/Audio_processing')
if not os.path.exists('transcriptions'):
    os.mkdir('transcriptions')
if not os.path.exists('test'):
    os.mkdir('test')
# %%
"""
os.system('module load ffmpeg')
for folder in glob.glob('/cobra/u/skapoor/COSMIC_data/*'):
    print(folder)
    for audio_file in glob.glob(f'{folder}/*.mp3'):
        wavname = audio_file.split(".mp3")[0]
        print(f'ffmpeg -i {audio_file} {wavname}.wav')
        os.system(f'ffmpeg -i {audio_file} {wavname}.wav')"""

# %%
for folder in glob.glob('/cobra/u/skapoor/COSMIC_data/*'):
    episode = folder.split('/')[-1]
    print(episode)
    if not os.path.exists(f'transcriptions/{episode}'):
        os.mkdir(f'transcriptions/{episode}')

    if not os.path.exists(f'test/{episode}'):
        os.mkdir(f'test/{episode}')
    for audio_file in glob.glob(f'{folder}/*.wav'):

        # audio = _upload_audio("")
        print(audio_file)
        audio = read_audio(audio_file)
        # print(file)
        #audio = _upload_audio(audio)
        # audio, sample_rate = librosa.load(file)
        _recognize(audio, episode)

# %%