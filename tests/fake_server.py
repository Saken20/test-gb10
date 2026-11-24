"""Stdlib fake of every stage service, so tests need no models or GPU. By hand: python tests/fake_server.py 8000"""
import base64
import io
import json
import re
import sys
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import numpy as np

TRANSCRIPT = "what is the weather like today"
ANSWER = "It is sunny and warm today. A light jacket is enough for the evening."
PIECES = re.findall(r"\S+\s*", ANSWER)
TTS_RATE = 24000
TTS_SEC_PER_CHAR = 0.04


def sine_pcm16(seconds, rate=TTS_RATE, amplitude=0.3):
    t = np.arange(int(seconds * rate)) / rate
    return (amplitude * np.sin(2 * np.pi * 220 * t) * 32767).astype("<i2").tobytes()


def to_wav(pcm, rate=TTS_RATE):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def has_speech(wav_bytes):
    with wave.open(io.BytesIO(wav_bytes)) as w:
        frames = w.readframes(w.getnframes())
    x = np.frombuffer(frames, "<i2")
    return x.size > 0 and int(np.abs(x).max()) > 300


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep pytest output clean
        pass

    def _body(self):
        return self.rfile.read(int(self.headers.get("Content-Length", 0)))

    def _json(self, obj, status=200):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _stream(self, content_type, chunks, delay=0.002):
        # HTTP/1.0 without Content-Length: the body ends when the connection closes
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.end_headers()
        for ch in chunks:
            self.wfile.write(ch)
            self.wfile.flush()
            time.sleep(delay)

    def do_POST(self):
        srv = self.server
        body = self._body()
        path = urlsplit(self.path).path
        srv.requests.append((path, dict(self.headers), body))
        if path == "/fail":
            return self._json({"error": "boom"}, 500)
        if path == "/tts_empty":
            return self._stream("audio/wav", [])
        if path == "/v1/chat/completions":
            req = json.loads(body)
            srv.llm_requests.append(req)
            if srv.llm_fail_after is not None and len(srv.llm_requests) > srv.llm_fail_after:
                return self._json({"error": "overloaded"}, 500)
            events = [{"choices": [{"delta": {"reasoning_content": "thinking"}}]}]
            events += [{"choices": [{"delta": {"content": p}}]} for p in PIECES]
            if srv.send_usage:
                events.append({"choices": [], "usage": {"completion_tokens": len(PIECES) + 1, "prompt_tokens": 7}})
            lines = [f"data: {json.dumps(e)}\n\n".encode() for e in events] + [b"data: [DONE]\n\n"]
            return self._stream("text/event-stream", lines)
        if path == "/v1/audio/transcriptions":
            wav = body[body.index(b"RIFF"):]
            wav = wav[:int.from_bytes(wav[4:8], "little") + 8]
            return self._json({"text": TRANSCRIPT if has_speech(wav) else ""})
        if path == "/asr":
            wav = base64.b64decode(json.loads(body)["audio_base64"])
            return self._json({"result": {"text": TRANSCRIPT if has_speech(wav) else ""}})
        if path == "/predict":
            text = json.loads(body)["text"].lower()
            return self._json({"intent": "weather" if "weather" in text else "other"})
        if path in ("/tts", "/tts_json", "/tts_pcm"):
            pcm = sine_pcm16(TTS_SEC_PER_CHAR * len(json.loads(body)["text"]))
            if path == "/tts_json":
                return self._json({"audio": base64.b64encode(to_wav(pcm)).decode()})
            data = pcm if path == "/tts_pcm" else to_wav(pcm)
            step = 4800
            return self._stream("application/octet-stream", [data[i:i + step] for i in range(0, len(data), step)])
        self._json({"error": "not found"}, 404)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128  # load tests open many connections at once


def start(port=0):
    server = Server(("127.0.0.1", port), Handler)
    server.requests, server.llm_requests = [], []
    server.send_usage, server.llm_fail_after = True, None
    server.url = f"http://127.0.0.1:{server.server_address[1]}"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


if __name__ == "__main__":
    srv = start(int(sys.argv[1]) if len(sys.argv) > 1 else 8000)
    print(f"Fake services on {srv.url}:", flush=True)
    for name, path in (("llm", "/v1/chat/completions"), ("asr (openai)", "/v1/audio/transcriptions"),
                       ("asr (http)", "/asr"), ("nlu", "/predict"), ("tts (wav)", "/tts"),
                       ("tts (json_base64)", "/tts_json"), ("tts (pcm16)", "/tts_pcm")):
        print(f"  {name:18s} {srv.url}{path}", flush=True)
    threading.Event().wait()
