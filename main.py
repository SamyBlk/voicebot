from gevent import monkey
monkey.patch_all()
import base64
import json
import os
import re
import tempfile
import warnings
import requests
import time
import threading
from flask import Flask, request, Response
from flask_sock import Sock
from pydub import AudioSegment, effects
from dotenv import load_dotenv
from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions
from rag import query_rag
import sys
from flask_cors import CORS
#from privacy import ReversiblePIIAnonymizer
#from light_privacy import anonymize_text, deanonymize_text


warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="websockets")

load_dotenv()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
deepgram_client = DeepgramClient(api_key=DEEPGRAM_API_KEY)
DEEPGRAM_TTS_URL = 'https://api.deepgram.com/v1/speak?model=aura-helios-en'
DEEPGRAM_HEADERS = {
    "Authorization": f"Token {DEEPGRAM_API_KEY}",
    "Content-Type": "application/json"
}

mic_muted = threading.Event()
stt_start_time = None

app = Flask(__name__)
sock = Sock(app)
CORS(app)


INCOMING_CALL_ROUTE = '/'
WEBSOCKET_ROUTE = '/realtime'

def shutdown_server():
    print("[Shutdown] Server will shut down now.")
    os._exit(0)



class AudioTextProcessor:
    @staticmethod
    def generate_audio(text):
        payload = {"text": text}
        try:
            with requests.post(DEEPGRAM_TTS_URL, stream=True, headers=DEEPGRAM_HEADERS, json=payload) as r:
                r.raise_for_status()
                return r.content
        except Exception as e:
            print(f"[ERROR] TTS generation failed: {e}")
            return None

    @staticmethod
    def convert_mp3_to_mulaw(mp3_data):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as mp3_file:
            mp3_file.write(mp3_data)
            mp3_path = mp3_file.name

        audio = AudioSegment.from_file(mp3_path, format="mp3")
        audio = audio.set_frame_rate(8000).set_channels(1)
        audio = effects.normalize(audio)
        audio = audio - 2

        fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        audio.export(wav_path, format="wav", codec="pcm_mulaw")

        try:
            with open(wav_path, "rb") as f:
                mulaw_data = f.read()
        finally:
            os.remove(wav_path)

        return mulaw_data

audio_processor = AudioTextProcessor()

#privacy_layer = ReversiblePIIAnonymizer()

@sock.route(WEBSOCKET_ROUTE)
def transcription_websocket(ws):
    dg_connection = deepgram_client.listen.websocket.v("1")
    stream_sid = None
    is_finals = []

    def on_open(connection, event):
        print("[Deepgram] WebSocket connected")

    def on_message(connection, result):
        nonlocal stream_sid
        if result.channel.alternatives[0].transcript:
            sentence = result.channel.alternatives[0].transcript
            if result.is_final:
                is_finals.append(sentence)
                if result.speech_final:
                    utterance = " ".join(is_finals)
                    print(f"\n[User] {utterance}")
                    is_finals.clear()

                    # --- PRIVACY: Anonymize before LLM ---
                    #utterance_anon = privacy_layer.anonymize(utterance)
                    #utterance_anon = anonymize_text(utterance)
                    #print(f"[Anonymized] {utterance_anon}")


                    # Mots pour quitter
                    if any(cmd in utterance.lower() for cmd in ["exit", "quit", "goodbye", "bye"]):
                        print("[EXIT] Exit command detected. Sending goodbye and shutting down...")

                        # Générer et envoyer l'audio de sortie
                        farewell = "Thank you for calling. Goodbye!"
                        mp3 = audio_processor.generate_audio(farewell)
                        if mp3:
                            mulaw = audio_processor.convert_mp3_to_mulaw(mp3)
                            if stream_sid:
                                payload_b64 = base64.b64encode(mulaw).decode("utf-8")
                                ws.send(json.dumps({
                                    "event": "media",
                                    "streamSid": stream_sid,
                                    "media": {"payload": payload_b64}
                                }))
                                time.sleep(2)  # temps de lecture

                        try:
                            dg_connection.finish()
                            ws.close()
                        except:
                            pass

                        # Couper Flask depuis un thread
                        threading.Thread(target=shutdown_server).start()
                        return

                    # --- RAG ---
                    total_start = time.time()
                    stt_end = time.time()
                    try:
                        rag_start = time.time()
                        response = query_rag(utterance)
                        #response_anon = query_rag(utterance_anon)
                        #response = query_rag(utterance)
                        rag_end = time.time()
                        print(f"[RAG] {response}")
                    except Exception as e:
                        response = "I'm sorry, I couldn't understand."
                        print(f"[ERROR] RAG failed: {e}")
                        rag_end = time.time()

                    # --- PRIVACY: Deanonymize after LLM ---
                    #response = privacy_layer.deanonymize(response_anon)
                    #response = deanonymize_text(response_anon)
                    #privacy_layer.reverse_map.clear()

                    # --- TTS ---
                    tts_start = time.time()
                    mp3 = audio_processor.generate_audio(response)
                    if mp3:
                        mulaw = audio_processor.convert_mp3_to_mulaw(mp3)
                        if stream_sid:
                            payload_b64 = base64.b64encode(mulaw).decode("utf-8")
                            ws.send(json.dumps({
                                "event": "media",
                                "streamSid": stream_sid,
                                "media": {"payload": payload_b64}
                            }))
                    tts_end = time.time()
                    total_end = time.time()

                    print(f"\n--- Timing Metrics ---")
                    print(f"STT latency     : {(rag_start - stt_end):.2f}s")
                    print(f"RAG latency     : {(rag_end - rag_start):.2f}s")
                    print(f"TTS latency     : {(tts_end - tts_start):.2f}s")
                    print(f"Total response  : {(total_end - stt_end):.2f}s")
                    print("----------------------\n")

    def on_close(connection, event):
        print("[Deepgram] WebSocket closed")

    def on_error(connection, error):
        print(f"[Deepgram] Error: {error}")

    dg_connection.on(LiveTranscriptionEvents.Open, on_open)
    dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
    dg_connection.on(LiveTranscriptionEvents.Close, on_close)
    dg_connection.on(LiveTranscriptionEvents.Error, on_error)

    options = LiveOptions(
        model="nova-3",
        language="en-US",
        smart_format=True,
        encoding="mulaw",
        sample_rate=8000,
        channels=1,
        interim_results=True,
        utterance_end_ms=1000,
        vad_events=True
    )

    if not dg_connection.start(options):
        print("[ERROR] Could not connect to Deepgram")
        return

    try:
        while True:
            message = json.loads(ws.receive())
            event = message.get("event")

            if event == "start":
                stream_sid = message["start"]["streamSid"]
                print(f"[Twilio] Stream started: {stream_sid}")
            elif event == "media":
                audio = base64.b64decode(message["media"]["payload"])
                dg_connection.send(audio)
            elif event == "stop":
                print("[Twilio] Stream stopped")
                dg_connection.finish()
                ws.close()
                break

    except Exception as e:
        print(f"[ERROR] WebSocket exception: {e}")
        try:
            dg_connection.finish()
            ws.close()
        except:
            pass

# Routes HTTP Twilio
@app.route(INCOMING_CALL_ROUTE, methods=["GET", "POST"])
def receive_call():
    if request.method == 'POST':
        xml = f"""
<Response>
    <Say>Welcome to Apple Support Assistant. Please speak after the beep.</Say>
    <Connect>
        <Stream url="wss://{request.host}{WEBSOCKET_ROUTE}" />
    </Connect>
</Response>
""".strip()
        return Response(xml, mimetype='text/xml')
    return "Voicebot is ready."



if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 5000))
    print(f" Flask server running on http://0.0.0.0:{PORT}")
    #print("  Start ngrok and configure your Twilio webhook manually.")
    app.run(host="0.0.0.0", port=PORT)