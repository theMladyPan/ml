#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy.io.wavfile import write as write_wav
import sys
import argparse
import logging
import os

logging.basicConfig(level=logging.INFO)
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

if args.small:
    os.environ["SUNO_USE_SMALL_MODELS"] = "1"
    log.info("Using small models")

from bark import SAMPLE_RATE, generate_audio, preload_models

log.info("Loading models...")
# download and load all models
preload_models()

if args.text == "":
    print("Input text prompt: ")

    # read stdin for text prompt
    text_prompt = sys.stdin.read()
else:
    text_prompt = args.text
    
log.info(f"Text prompt: {text_prompt}")

# generate audio from text
audio_array = generate_audio(text_prompt)

if not args.output.endswith(".wav"):
    args.output += ".wav"
    
log.info(f"Saving audio to {args.output}...")
# save audio to disk
write_wav(args.output, SAMPLE_RATE, audio_array)
