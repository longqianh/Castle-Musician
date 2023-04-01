from pydub import AudioSegment
from pydub.playback import play
from pathlib import Path

def sound_to_wav(sound,save_path=None):
    sound_wav=sound.export(save_path,format="wav")
    return sound_wav

if __name__ == "__main__":
    music_path=Path('./src/data/audio-to-piano/castle_in_the_sky.mp3')
    sound = AudioSegment.from_file(music_path)
    sound_wav=sound.export(format="wav")
    play(AudioSegment.from_wav(sound_wav))