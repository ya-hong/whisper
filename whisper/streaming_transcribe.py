import argparse
import os
import warnings
from typing import Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
import torch
import tqdm

from .audio import SAMPLE_RATE, N_FRAMES, HOP_LENGTH, pad_or_trim, log_mel_spectrogram, N_MELS
from .decoding import DecodingOptions, DecodingResult
from .tokenizer import LANGUAGES, TO_LANGUAGE_CODE, get_tokenizer
from .utils import exact_div, format_timestamp, make_safe, optional_int, optional_float, str2bool, get_writer

if TYPE_CHECKING:
    from .model import Whisper

import ffmpeg
import time
from io import BytesIO
import wave
    

def load_audio(file: bytes, sr: int = SAMPLE_RATE):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: bytes
        The bytes of audio file

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    inp = file
    file = 'pipe:'

    try:
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr
                    )
            .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True, input=inp)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def streaming_transcribe(
    model: "Whisper",
    audio_stream,
    *,
    verbose: Optional[bool] = None,
    compression_ratio_threshold: Optional[float] = 2.4,
    logprob_threshold: Optional[float] = -1.0,
    no_speech_threshold: Optional[float] = 0.6,
    condition_on_previous_text: bool = True,
    **decode_options,
):
    """
    Transcribe an audio file using Whisper

    Parameters
    ----------
    model: Whisper
        The Whisper model instance

    audio_stream: 
        use method read() to get last bytes of audio waveform,
        use attribute frame_count

    verbose: bool
        Whether to display the text being decoded to the console. If True, displays all the details,
        If False, displays minimal details. If None, does not display anything

    compression_ratio_threshold: float
        If the gzip compression ratio is above this value, treat as failed

    logprob_threshold: float
        If the average log probability over sampled tokens is below this value, treat as failed

    no_speech_threshold: float
        If the no_speech probability is higher than this value AND the average log probability
        over sampled tokens is below `logprob_threshold`, consider the segment as silent

    condition_on_previous_text: bool
        if True, the previous output of the model is provided as a prompt for the next window;
        disabling may make the text inconsistent across windows, but the model becomes less prone to
        getting stuck in a failure loop, such as repetition looping or timestamps going out of sync.

    decode_options: dict
        Keyword arguments to construct `DecodingOptions` instances

    Returns
    -------
    A dictionary containing the resulting text ("text") and segment-level details ("segments"), and
    the spoken language ("language"), which is detected when `decode_options["language"]` is None.
    """

    input_stride = exact_div(
        N_FRAMES, model.dims.n_audio_ctx
    )  # mel frames per output token: 2
    time_precision = (
        input_stride * HOP_LENGTH / SAMPLE_RATE
    )  # time per output token: 0.02 (seconds)

    dtype = torch.float16 if decode_options.get(
        "fp16", True) else torch.float32

    if model.device == torch.device("cpu"):
        if torch.cuda.is_available():
            warnings.warn("Performing inference on CPU when CUDA is available")
        if dtype == torch.float16:
            warnings.warn("FP16 is not supported on CPU; using FP32 instead")
            dtype = torch.float32

    if dtype == torch.float32:
        decode_options["fp16"] = False

    if decode_options.get("language", None) is None:
        decode_options["language"] = "en"

    task = decode_options.get("task", "transcribe")
    tokenizer = get_tokenizer(model.is_multilingual,
                              language=decode_options["language"], task=task)

    def gen_segment(
        *, start: float, end: float, text_tokens: torch.Tensor, result: DecodingResult, **extra_info
    ):
        # text = tokenizer.decode([token for token in text_tokens if token < tokenizer.eot])
        text = tokenizer.decode(text_tokens)

        if verbose:
            print(make_safe(f"[{format_timestamp(start)} --> {format_timestamp(end)}] {text}"))

        return {
            "id": 0,
            "seek": seek,
            "start": start,
            "end": end,
            "text": text,
            "tokens": text_tokens.tolist(),
            "temperature": result.temperature,
            "avg_logprob": result.avg_logprob,
            "compression_ratio": result.compression_ratio,
            "no_speech_prob": result.no_speech_prob,
            **extra_info
        }
        

    mel = torch.zeros((N_MELS, N_FRAMES)).to(model.device).to(dtype)
    offset = 0  # 不参与decode的部分
    seek = N_FRAMES # 已经输出的部分
    
    prompt_tokens_list = []
    prev_tokens_list = []
    last_prompt = None

    audio_break = False

    while True:
        if audio_stream.frame_count < HOP_LENGTH:
            if seek < offset + N_FRAMES * 0.7:
                # no buffer and don't need new buff
                last_mel = torch.zeros((N_MELS, 0))
            elif not audio_break:
                time.sleep(0.2)
                audio_break = True
                continue
            else:
                last_mel = torch.zeros((N_MELS, 20))
                audio_break = False
        else:
            last_bytes = audio_stream.read()
            audio_break = False
            last_audio = load_audio(last_bytes)
            last_mel = log_mel_spectrogram(last_audio)
        
        last_mel = last_mel.to(model.device).to(dtype)
        mel = torch.cat([mel, last_mel], dim=1)[:, -min(mel.shape[-1], 2*N_FRAMES):]
        offset = offset + last_mel.shape[-1]
        seek = max(seek, offset)
        segement = mel[:, -N_FRAMES:]
        """
        ...................|-------|-------|
                        offset    seek
                           <----N_FRAMES--->
                   <--------------mel------>
        """

        if condition_on_previous_text:
            last_tpos = 0 if last_prompt is None else last_prompt["tpos"]
            for item in prev_tokens_list:
                if last_tpos <= item["spos"] and item["spos"] < offset:
                    prompt_tokens_list.append(item["tokens"])
                    last_prompt = item
            prev_tokens_list = []
            prompt_tokens_list = prompt_tokens_list[-min(len(prompt_tokens_list), 10):]
            decode_options["prompt"] = [
                token 
                for tokens in prompt_tokens_list
                for token in tokens 
            ]

        options = DecodingOptions(**decode_options)
        result = model.decode(segement, options)

        # if logprob_threshold is not None and result.avg_logprob < logprob_threshold * 3 and :

        if logprob_threshold is not None and result.avg_logprob < logprob_threshold:
            if offset + N_FRAMES - seek > 1000:
                delta = last_mel.shape[-1]
                mel = mel[:, :-delta]
                seek += delta
                print("drop!")
            continue
        if compression_ratio_threshold is not None and result.compression_ratio > compression_ratio_threshold:
            continue
        
        tokens = torch.tensor(result.tokens)

        if no_speech_threshold is not None:
            if result.no_speech_prob > no_speech_threshold:
                seek = offset + N_FRAMES
                continue

        i = 0
        while i < len(tokens):
            stoken = tokens[i]
            assert(stoken >= tokenizer.timestamp_begin)

            for k, ttoken in enumerate(tokens[i+1:]):
                if ttoken >= tokenizer.timestamp_begin:
                    break
            j = i + 1 + k + 1 #[i, j)

            spos = (stoken.item() - tokenizer.timestamp_begin) * input_stride
            tpos = (ttoken.item() - tokenizer.timestamp_begin) * input_stride

            if i == len(tokens) - 1: # 最后一句没翻译完
                break
            if not i == 0 and (j >= len(tokens) - 1 and tpos >= N_FRAMES - (tpos - spos) * 0.3): 
                # 弃用了原本关于最后一句没翻译完的判断（最后一位是一个开始标记）
                # 那样识别出来的最后一句还是容易被改变
                break

            prev_tokens_list.append(dict(
                spos=offset + spos,
                tpos=offset + tpos,
                tokens=tokens[i:j],
                text=tokenizer.decode(tokens[i+1:j-1]),
            ))

            # print(spos+offset, tpos+offset, seek, j, len(tokens), tokenizer.decode(tokens[i:j]))

            if offset + tpos >= seek + (tpos - spos) * 0.4:
                assert(j - i >= 3)

                yield gen_segment(
                    start=  float((offset + spos) * HOP_LENGTH / SAMPLE_RATE),
                    end=    float((offset + tpos) * HOP_LENGTH / SAMPLE_RATE),

                    text_tokens=tokens[i+1:j-1],
                    result=result,
                    offset=offset,
                    spos=spos,
                    tpos=tpos,
                    delta_time = float((N_FRAMES - tpos) * HOP_LENGTH / SAMPLE_RATE),
                )
                seek = offset + tpos

            i = j
