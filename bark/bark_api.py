#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
time_start = time.perf_counter()
from scipy.io.wavfile import write as write_wav
import sys
import argparse
import logging
import os

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

# parse command line arguments
parser = argparse.ArgumentParser(description="Generate audio from text prompt")
parser.add_argument(
    "-t", 
    "--text", 
    type=str, 
    help="Text prompt to generate audio from", 
    default=""
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    help="Output file name",
    default="bark_generation.wav"
)
parser.add_argument(
    "-s",
    "--small",
    action="store_true",
    help="Use small models for faster generation",
    default=False
)

# parse
args = parser.parse_args()

if args.text == "":
    print("Input text prompt: ")

    # read stdin for text prompt
    text_prompt = sys.stdin.read()
else:
    text_prompt = args.text
    
# sanitize filename for output
output_filename = args.output
for char in ["\\", "/", ":", "*", "?", "\"", "<", ">", "|"]:
    output_filename.replace(char, "_")

if not output_filename.endswith(".wav"):
    output_filename += ".wav"

log.info(f"Text prompt: {text_prompt}")

sentences = []
if len(text_prompt) > 20:
    log.warning("Text prompt is longer than 20 characters, this may take a while...")
    import nltk
    nltk.download('punkt')  # Download the required punkt tokenizer if you haven't already

    from nltk.tokenize import sent_tokenize

    sentences = sent_tokenize(text_prompt)

    log.info(f"Split into {len(sentences)} sentences")
    log.debug(sentences)

if args.small:
    os.environ["SUNO_USE_SMALL_MODELS"] = "1"
    log.info("Using small models")

from bark import SAMPLE_RATE, generate_audio, preload_models

log.info("Loading models...")
# download and load all models
preload_models()

sentence_filenames = []
if len(sentences) > 0:
    for i, sentence in enumerate(sentences, start=1):
        log.info(f"Generating audio for sentence {i}...")
        # generate audio from text
        if i == 1:
            # generate first prompt with history
            full_gen, audio_array = generate_audio(
                sentence,
                output_full=True
            )
        else:
            # generate subsequent prompts with history
            full_gen, audio_array = generate_audio(
                sentence,
                history_prompt=full_gen,
                output_full=True
            )
    
        intermediate_filename = output_filename.replace(".wav", f"_{i}_{len(sentences)}.wav")
        sentence_filenames.append(intermediate_filename)

        log.info(f"Saving audio to {intermediate_filename}...")
        # save audio to disk
        write_wav(intermediate_filename, SAMPLE_RATE, audio_array)
        
else:
    log.info("Generating audio...")
    # generate audio from text
    audio_array = generate_audio(text_prompt)

    log.info(f"Saving audio to {output_filename}...")
    # save audio to disk
    write_wav(output_filename, SAMPLE_RATE, audio_array)

# finally merge all the audio files
if len(sentences) > 0:
    log.info("Merging audio files...")
    from pydub import AudioSegment

    combined = AudioSegment.empty()
    for filename in sentence_filenames:
        combined += AudioSegment.from_wav(filename)

    # save as mp3
    output_filename = output_filename.replace(".wav", ".mp3")
    
    combined.export(output_filename, format="mp3")
    log.info(f"Saved merged audio to {output_filename}")
    
    log.info("Deleting temporary files...")
    # delete temporary files
    for filename in sentence_filenames:
        log.debug(f"Deleting {filename}")
        os.remove(filename)             
             
log.info(f"Done! Took {time.perf_counter() - time_start:.3f} seconds")