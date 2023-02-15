import soundfile as sf
import librosa
import torch
import os
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from huggingsound import SpeechRecognitionModel
from transformers import WhisperProcessor, WhisperForConditionalGeneration, HubertForCTC
from config.definitions import ROOT_DIR


def load_audio(name_audio: str) -> torch.Tensor:
    # Get the path of the audio file to be transcribed
    file_path: str = os.path.join(ROOT_DIR, 'data', name_audio)

    input_audio, _ = librosa.load(file_path, sr=16000)
    return input_audio

# Transcription with facebook/wav2vec2-base-960h
def model1() -> str:
    tokenizer = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    input_values = tokenizer(input_audio, return_tensors="pt").input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)[0]

    return transcription

# Transcription with jonatasgrosman/wav2vec2-large-xlsr-53-english
def model2(name_audio: str) -> str:
    model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-english")
    audio_paths = [os.path.join(ROOT_DIR, 'data', name_audio)]
    transcriptions = model.transcribe(audio_paths)
    return transcriptions[0]["transcription"]

# Transcription with openai/whisper-tiny
def model3() -> str:
    # Load model and processor
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    model.config.forced_decoder_ids = None

    # Read audio files
    input_features = processor(input_audio, return_tensors="pt").input_features

    # Generate token ids
    predicted_ids = model.generate(input_features)
    # Decode token ids to text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
    return transcription

# Transcription with facebook/hubert-large-ls960-ft
def model4() -> str:
    # load model and processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
    model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")

    # tokenize
    input_values = processor(input_audio, return_tensors="pt").input_values
    
    # retrieve logits
    logits = model(input_values).logits
    
    # take argmax and decode
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]

    return transcription

# Transcription with  yongjian/wav2vec2-large-a
def model5() -> str:
    model = Wav2Vec2ForCTC.from_pretrained(r'yongjian/wav2vec2-large-a') # Note: PyTorch Model
    processor = Wav2Vec2Processor.from_pretrained(r'yongjian/wav2vec2-large-a')

    # Inference
    sample_rate = processor.feature_extractor.sampling_rate
    with torch.no_grad():
        model_inputs = processor(input_audio, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        logits = model(model_inputs.input_values, attention_mask=model_inputs.attention_mask).logits # use .cuda() for GPU acceleration
        pred_ids = torch.argmax(logits, dim=-1).cpu()
        pred_text = processor.batch_decode(pred_ids)

    return pred_text[0]

# Get results for the noise-free audio.wav
input_audio = load_audio('noise-free-audio.wav')

model1_transcription: str = model1()
model2_transcription: str  = model2('noise-free-audio.wav')
model3_transcription: str  = model3()
model4_transcription: str  = model4()
model5_transcription: str  = model5()

print(f"\nTranscription 1 with facebook/wav2vec2-base-960h: {model1_transcription}")
print(f"\nTranscription 1 with jonatasgrosman/wav2vec2-large-xlsr-53-english: {model2_transcription}")
print(f"\nTranscription 1 with openai/whisper-tiny: {model3_transcription}")
print(f"\nTranscription 1 with facebook/hubert-large-ls960-ft: {model4_transcription}")
print(f"\nTranscription 1 with yongjian/wav2vec2-large-a: {model5_transcription}\n")

# Get results for the audio-with-noise.wav
input_audio = load_audio('audio-with-noise.wav')

model1_transcription= model1()
model2_transcription= model2('audio-with-noise.wav')
model3_transcription= model3()
model4_transcription= model4()
model5_transcription= model5()

print(f"\nTranscription 2 with facebook/wav2vec2-base-960h: {model1_transcription}")
print(f"\nTranscription 2 with jonatasgrosman/wav2vec2-large-xlsr-53-english: {model2_transcription}")
print(f"\nTranscription 2 with openai/whisper-tiny: {model3_transcription}")
print(f"\nTranscription 2 with facebook/hubert-large-ls960-ft: {model4_transcription}")
print(f"\nTranscription 2 with yongjian/wav2vec2-large-a: {model5_transcription}\n")