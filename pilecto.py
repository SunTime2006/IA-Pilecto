from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git

%cd /content/Retrieval-based-Voice-Conversion-WebUI

import librosa
import soundfile as sf
import os

audio_dir = '/content/drive/MyDrive/dataset'

output_dir = '/content/processed_dataset'
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(audio_dir):
    if filename.endswith('.wav'):
        filepath = os.path.join(audio_dir, filename)
        
        y, sr = librosa.load(filepath, sr=44100)
        
        sf.write(os.path.join(output_dir, filename), y, sr)

print("Procesamiento de audio completado.")

!python train_model.py \
  --input_path "/content/processed_dataset" \
  --model_name "mi_modelo_voz" \
  --epochs 20 \
  --batch_size 4


!python infer.py \
  --input_audio "/content/audio_a_convertir.wav" \
  --model_path "/content/Retrieval-based-Voice-Conversion-WebUI/weights/mi_modelo_voz.pth" \
  --output_audio "/content/audio_convertido.wav"

print("Conversión de voz completada. El audio convertido está en /content/audio_convertido.wav")
