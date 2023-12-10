from pydub import AudioSegment
from pydub.playback import play
from pathlib import Path
import wave
from scipy.io import wavfile
import noisereduce as nr


def sound_to_wav(sound, save_path=None):
    sound_wav = sound.export(save_path, format="wav")
    return sound_wav


def get_wav_audio(file_path):
    audio = AudioSegment.from_file(file_path)
    audio_wav_fnirs = io.BytesIO()
    audio.export(audio_wav_fnirs, format="wav")
    return audio_wav_fnirs


def reduce_noise(in_file, out_file, freq=16000, strength=0.80):
    rate, data = wavfile.read(in_file)
    reduced_noise = nr.reduce_noise(y=data, sr=freq, prop_decrease=strength)
    with wave.open(out_file, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(freq)
        wf.writeframes(b"".join(reduced_noise))


if __name__ == "__main__":
    import io

    music_path = Path("./assets/castle_in_the_sky.mp3")
    sound = AudioSegment.from_file(music_path)
    audio_wav = io.BytesIO()
    sound.export(audio_wav, format="wav")
    play(AudioSegment.from_wav(audio_wav))
