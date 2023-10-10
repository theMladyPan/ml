from scipy.io.wavfile import write as write_wav
from bark import SAMPLE_RATE, generate_audio, preload_models
import scipy

# download and load all models
preload_models()

# generate audio from text
text_prompt = """
     Hello, my name is Suno. And, uh — and I like pizza. [laughs] 
     But I also have other interests such as playing tic tac toe.
"""

# read stdin for text prompt
import sys
text_prompt = sys.stdin.read()

audio_array = generate_audio(text_prompt)

# save audio to disk
write_wav("bark_generation.wav", SAMPLE_RATE, audio_array)


processor_small = AutoProcessor.from_pretrained("suno/bark-small")
model_small = AutoModel.from_pretrained("suno/bark-small")

processor = AutoProcessor.from_pretrained("suno/bark")
model = AutoModel.from_pretrained("suno/bark")


voice_preset = "v2/en_speaker_6"
voice_preset = "v2/de_speaker_1"
voice_preset = "./en_speaker_100"

speech = """Hello everyone!"""

inputs = processor_small(speech, voice_preset=voice_preset)

audio_array = model_small.generate(**inputs)
audio_array = audio_array.cpu().numpy().squeeze()

sample_rate = model_small.generation_config.sample_rate
scipy.io.wavfile.write("bark_out.wav", rate=sample_rate, data=audio_array)

exit(0)

# download and load all models
preload_models()

# generate audio from text
text_prompt = """
     Hello, my name is Suno. And, uh — and I like pizza. [laughs] 
     But I also have other interests such as playing tic tac toe.
"""
audio_array = generate_audio(text_prompt)

# save audio to disk
