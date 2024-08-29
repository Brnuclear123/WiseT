from flask import Flask, request, redirect, render_template_string, send_file
import os
from pydub import AudioSegment
import whisper
import re
from collections import Counter
import csv
from datetime import datetime
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Palavras mágicas
MAGIC_WORDS = [
    "obrigado", "obrigada", "por favor", "desculpa", "desculpe", "por gentileza",
    "bom dia", "boa tarde", "boa noite", "agradeço", "agradece", "gratidão", 
    "sinto muito", "perdão", "me perdoe", "com licença"
]

def clean_transcription(text):
    text = re.sub(r'\s+', ' ', text)  # Remove espaços extras
    text = re.sub(r'\b(e|então)\b', r', \1', text, flags=re.IGNORECASE)  # Adiciona vírgula antes de 'e', 'então', etc.
    return text.strip()  # Remove espaços no início e fim

def calcular_estatisticas(texto, duracao_minutos):
    palavras = texto.lower().split()
    total_palavras = len(palavras)
    palavras_por_minuto = total_palavras / duracao_minutos
    contagem_palavras = Counter(palavras)
    palavras_ordenadas = contagem_palavras.most_common()
    total_magicas = sum(contagem_palavras[palavra] for palavra in MAGIC_WORDS)
    percentual_magicas = (total_magicas / total_palavras) * 100 if total_palavras > 0 else 0

    return {
        "total_palavras": total_palavras,
        "palavras_por_minuto": palavras_por_minuto,
        "palavras_ordenadas": palavras_ordenadas,
        "percentual_magicas": percentual_magicas
    }

def analisar_sentimento(transcribed_text):
    # Implementação simples baseada na presença de palavras-chave
    positive_words = ["bom", "ótimo", "excelente", "satisfeito", "maravilhoso"]
    negative_words = ["ruim", "péssimo", "horrível", "insatisfeito", "terrível"]

    positive_count = sum(1 for word in transcribed_text.lower().split() if word in positive_words)
    negative_count = sum(1 for word in transcribed_text.lower().split() if word in negative_words)

    if positive_count > negative_count:
        return "Satisfeito"
    elif negative_count > positive_count:
        return "Insatisfeito"
    else:
        return "Neutro"

def transcribe_audio_whisper(model, audio_path):
    result = model.transcribe(audio_path, word_timestamps=True)
    return result['segments']

def generate_csv(transcribed_segments, estatisticas, output_csv_path):
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['timestamp', 'transcription']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for segment in transcribed_segments:
            timestamp = format_timestamp(segment['start'] * 1000)
            transcription = segment['text'].strip()
            writer.writerow({'timestamp': timestamp, 'transcription': transcription})
        
        writer.writerow({})
        writer.writerow({'timestamp': 'Estatísticas', 'transcription': ''})
        writer.writerow({'timestamp': 'Total de palavras', 'transcription': estatisticas['total_palavras']})
        writer.writerow({'timestamp': 'Palavras por minuto', 'transcription': estatisticas['palavras_por_minuto']})
        writer.writerow({'timestamp': 'Percentual de palavras mágicas', 'transcription': estatisticas['percentual_magicas']})

        for palavra, quantidade in estatisticas['palavras_ordenadas']:
            writer.writerow({'timestamp': palavra, 'transcription': quantidade})

def format_timestamp(milliseconds):
    seconds = milliseconds // 1000
    minutes = seconds // 60
    seconds = seconds % 60
    return f"({minutes}:{seconds:02})"

def format_transcription(transcribed_segments):
    formatted_text = ""
    for segment in transcribed_segments:
        timestamp = format_timestamp(segment['start'] * 1000)
        transcription = segment['text'].strip()
        formatted_text += f"{timestamp} {transcription}\n\n"
    return formatted_text.strip()

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Carregar o modelo Whisper uma vez ao iniciar o servidor
whisper_model = whisper.load_model("base")

@app.route('/', methods=['GET', 'POST'])
def upload_audio():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename != '':
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Converte o arquivo para garantir compatibilidade
            audio = AudioSegment.from_file(filepath)
            converted_filepath = os.path.join(app.config['UPLOAD_FOLDER'], "converted.wav")
            audio.export(converted_filepath, format="wav")

            # Transcrição completa do áudio
            transcribed_segments = transcribe_audio_whisper(whisper_model, converted_filepath)
            formatted_transcription = format_transcription(transcribed_segments)
            duracao_minutos = len(audio) / (1000 * 60)

            # Cálculo das estatísticas
            estatisticas = calcular_estatisticas(formatted_transcription, duracao_minutos)

            # Análise de Sentimento
            sentimento = analisar_sentimento(formatted_transcription)

            # Geração do CSV com transcrição e estatísticas
            output_csv_path = os.path.join(app.config['UPLOAD_FOLDER'], f"transcription_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv")
            generate_csv(transcribed_segments, estatisticas, output_csv_path)

            # Exibir resultados na aba "Resultados"
            return render_template_string(render_html(), formatted_transcription=formatted_transcription, estatisticas=estatisticas, sentimento=sentimento, output_csv_path=output_csv_path)
        else:
            return redirect('/')
    
    return render_template_string(render_html(), formatted_transcription='', estatisticas={}, sentimento='N/A', output_csv_path='')

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return send_file(file_path, as_attachment=True)

def render_html():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Upload de Áudio</title>
        <style>
            body {
                font-family: 'Helvetica', Arial, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f4f4f4;
            }
            .container {
                width: 80%;
                margin: 50px auto;
                background-color: #fff;
                border-radius: 10px;
                box-shadow: 0 0 20px rgba(0,0,0,0.1);
                overflow: hidden;
            }
            .tabs {
                display: flex;
                justify-content: space-around;
                background-color: #007bff;
                color: white;
            }
            .tab {
                flex: 1;
                text-align: center;
                padding: 15px 0;
                cursor: pointer;
                font-size: 1.2em;
            }
            .tab:hover {
                background-color: #0056b3;
            }
            .tab.active {
                background-color: #0056b3;
                border-bottom: 3px solid #ffcc00;
            }
            .content {
                padding: 20px;
            }
            h1 {
                color: #333;
            }
            form {
                background: #f9f9f9;
                padding: 20px;
                border-radius: 5px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
                text-align: center;
            }
            input[type="file"] {
                margin-bottom: 20px;
                font-size: 1em;
            }
            input[type="submit"] {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 1em;
            }
            input[type="submit"]:hover {
                background-color: #0056b3;
            }
            .result {
                margin-top: 20px;
                padding: 20px;
                background-color: #e9ecef;
                border-radius: 5px;
            }
        </style>
        <script>
            function switchTab(tabIndex) {
                const tabs = document.querySelectorAll('.tab');
                const contents = document.querySelectorAll('.content');
                tabs.forEach((tab, index) => {
                    tab.classList.remove('active');
                    contents[index].style.display = 'none';
                });
                tabs[tabIndex].classList.add('active');
                contents[tabIndex].style.display = 'block';
            }
        </script>
    </head>
    <body onload="switchTab(0)">
        <div class="container">
            <div class="tabs">
                <div class="tab active" onclick="switchTab(0)">Upload de Áudio</div>
                <div class="tab" onclick="switchTab(1)">Resultados</div>
                <div class="tab" onclick="switchTab(2)">Estatísticas</div>
            </div>
            <div class="content">
                <h1>Faça o upload do arquivo de áudio aqui!</h1>
                <form method="POST" enctype="multipart/form-data">
                    <input type="file" name="file" accept="audio/*" required>
                    <input type="submit" value="Upload">
                </form>
            </div>
            <div class="content" style="display:none;">
                <h1>Resultados</h1>
                <div class="result">
                    {% if formatted_transcription %}
                        <pre>{{ formatted_transcription }}</pre>
                        <h3>Sentimento Geral: {{ sentimento }}</h3>
                        <p><a href="{{ url_for('download_file', filename=output_csv_path.split('/')[-1]) }}" download>Clique aqui para baixar a transcrição em CSV</a></p>
                    {% else %}
                        <p>Os resultados aparecerão aqui após o processamento do áudio.</p>
                    {% endif %}
                </div>
            </div>
            <div class="content" style="display:none;">
                <h1>Estatísticas</h1>
                <div class="result">
                    {% if estatisticas.total_palavras %}
                        <p>Total de palavras: {{ estatisticas.total_palavras }}</p>
                        <p>Palavras por minuto: {{ estatisticas.palavras_por_minuto }}</p>
                        <h3>Palavras mais faladas:</h3>
                        <ul>
                            {% for palavra, quantidade in estatisticas.palavras_ordenadas %}
                                <li>{{ palavra }}: {{ quantidade }}</li>
                            {% endfor %}
                        </ul>
                        <p>Percentual de palavras mágicas: {{ estatisticas.percentual_magicas }}%</p>
                    {% else %}
                        <p>As estatísticas aparecerão aqui após o processamento do áudio.</p>
                    {% endif %}
                </div>
            </div>
        </div>
    </body>
    </html>
    """

if __name__ == '__main__':
    app.run(debug=True)
