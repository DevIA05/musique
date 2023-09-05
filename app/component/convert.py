from pydub import AudioSegment

def convert_mp3_to_wav(input_mp3_file, output_wav_file):
    try:
     
        audio = AudioSegment.from_mp3(input_mp3_file)

     
        audio.export(output_wav_file, format="wav")

        print(f"Converted {input_mp3_file} to {output_wav_file}")
    except Exception as e:
        print(f"Error converting {input_mp3_file} to WAV: {str(e)}")