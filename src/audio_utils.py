from pydub import AudioSegment
from pydub.playback import play
from pathlib import Path

def sound_to_wav(sound,save_path=None):
    sound_wav=sound.export(save_path,format="wav")
    return sound_wav

if __name__ == "__main__":
    import io
    music_path=Path('./assets/castle_in_the_sky.mp3')
    sound = AudioSegment.from_file(music_path)
    audio_wav = io.BytesIO()
    sound.export(audio_wav,format="wav")
    play(AudioSegment.from_wav(audio_wav))