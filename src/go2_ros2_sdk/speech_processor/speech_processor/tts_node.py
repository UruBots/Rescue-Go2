#!/usr/bin/env python3

# Copyright (c) 2024, RoboVerse community
# SPDX-License-Identifier: BSD-3-Clause

"""
TTS Node con Piper (offline, sin internet, sin API key)

Genera voz local usando el binario de Piper y la envia al altavoz del robot
via WebRTC. El topico de entrada es /tts (std_msgs/String).

Uso rapido (sin lanzar ROS2):
  echo "Hola perro" | ~/piper_voices/piper/piper \
    --model ~/piper_voices/es_ES-davefx-medium.onnx --output_raw \
    | aplay -r 22050 -f S16_LE -t raw -
"""

import base64
import io
import json
import os
import subprocess
import time
import threading
from typing import Optional, List

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from go2_interfaces.msg import WebRtcReq

# ---------------------------------------------------------------------------
# Configuracion por defecto
# ---------------------------------------------------------------------------
DEFAULT_PIPER_BIN  = os.path.expanduser("~/piper_voices/piper/piper")
DEFAULT_PIPER_MODEL = os.path.expanduser("~/piper_voices/es_ES-davefx-medium.onnx")
SAMPLE_RATE = 22050   # Hz que usa el modelo davefx
CHUNK_SIZE  = 16 * 1024  # bytes por fragmento al mandar al robot


# ---------------------------------------------------------------------------
# Generador de audio con Piper
# ---------------------------------------------------------------------------
class PiperTTS:
    """Genera audio RAW (PCM S16LE) usando el binario de Piper."""

    def __init__(self, piper_bin: str, model_path: str):
        self.piper_bin  = piper_bin
        self.model_path = model_path

    def synthesize_raw(self, text: str) -> Optional[bytes]:
        """Devuelve bytes PCM S16LE o None si falla."""
        try:
            result = subprocess.run(
                [self.piper_bin, "--model", self.model_path, "--output_raw"],
                input=text.encode("utf-8"),
                capture_output=True,
                timeout=15
            )
            if result.returncode != 0:
                return None
            return result.stdout
        except Exception:
            return None

    def synthesize_wav(self, text: str) -> Optional[bytes]:
        """Devuelve bytes WAV (con cabecera) o None si falla."""
        raw = self.synthesize_raw(text)
        if raw is None:
            return None
        return self._raw_to_wav(raw)

    @staticmethod
    def _raw_to_wav(raw: bytes, sample_rate: int = SAMPLE_RATE,
                    num_channels: int = 1, bits_per_sample: int = 16) -> bytes:
        """Envuelve PCM raw en cabecera WAV estándar."""
        data_size   = len(raw)
        byte_rate   = sample_rate * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8
        chunk_size  = 36 + data_size

        header = (
            b"RIFF"
            + chunk_size.to_bytes(4, "little")
            + b"WAVE"
            + b"fmt "
            + (16).to_bytes(4, "little")       # sub-chunk1 size
            + (1).to_bytes(2, "little")         # PCM format
            + num_channels.to_bytes(2, "little")
            + sample_rate.to_bytes(4, "little")
            + byte_rate.to_bytes(4, "little")
            + block_align.to_bytes(2, "little")
            + bits_per_sample.to_bytes(2, "little")
            + b"data"
            + data_size.to_bytes(4, "little")
        )
        return header + raw


# ---------------------------------------------------------------------------
# Nodo ROS2
# ---------------------------------------------------------------------------
class TTSNode(Node):
    """Nodo TTS offline usando Piper. Suscribe /tts y envia audio al robot."""

    # Topicos WebRTC del Go2 (deben ser strings exactos)
    RTC_TOPIC = {"AUDIO_HUB_REQ": "rt/api/audiohub/request"}

    def __init__(self):
        super().__init__("tts_node")

        # Parametros
        self.declare_parameter("piper_bin",   DEFAULT_PIPER_BIN)
        self.declare_parameter("piper_model", DEFAULT_PIPER_MODEL)
        self.declare_parameter("local_playback", False)
        self.declare_parameter("chunk_size", CHUNK_SIZE)

        piper_bin   = self.get_parameter("piper_bin").get_parameter_value().string_value
        piper_model = self.get_parameter("piper_model").get_parameter_value().string_value
        self.local_playback = self.get_parameter("local_playback").get_parameter_value().bool_value
        self.chunk_size     = self.get_parameter("chunk_size").get_parameter_value().integer_value

        # Validar que existan los archivos
        if not os.path.isfile(piper_bin):
            self.get_logger().error(f"❌ Binario de Piper no encontrado: {piper_bin}")
            self.get_logger().error("   Ejecuta el script de instalacion primero.")
            return
        if not os.path.isfile(piper_model):
            self.get_logger().error(f"❌ Modelo de voz no encontrado: {piper_model}")
            return

        self.piper = PiperTTS(piper_bin, piper_model)
        self._lock = threading.Lock()  # evita reproduccion simultanea

        # Comunicacion ROS2
        self.subscription = self.create_subscription(String, "/tts", self._tts_callback, 10)
        self.play_wav_sub = self.create_subscription(String, "/play_wav_on_robot", self._play_wav_callback, 10)
        self.audio_pub = self.create_publisher(WebRtcReq, "/webrtc_req", 10)

        self.get_logger().info("🎤 TTS Node (Piper offline) iniciado")
        self.get_logger().info(f"   Binario : {piper_bin}")
        self.get_logger().info(f"   Modelo  : {piper_model}")
        self.get_logger().info(f"   Playback: {'Local (PC)' if self.local_playback else 'Robot'}")

    # ------------------------------------------------------------------
    def _tts_callback(self, msg: String) -> None:
        text = msg.data.strip()
        if not text:
            return

        self.get_logger().info(f'🔊 TTS: "{text}"')

        # Sintetizar en hilo separado para no bloquear el spin
        threading.Thread(target=self._synthesize_and_play, args=(text,), daemon=True).start()

    def _play_wav_callback(self, msg: String) -> None:
        filepath = msg.data.strip()
        if not filepath:
            return

        self.get_logger().info(f'🔊 Recibido WAV para reproducir en robot: "{filepath}"')
        if not os.path.exists(filepath):
            self.get_logger().error(f"❌ Archivo WAV no encontrado: {filepath}")
            return

        # Leer archivo y reproducir en hilo separado
        try:
            with open(filepath, "rb") as f:
                wav_data = f.read()
            threading.Thread(target=self._play_on_robot_thread, args=(wav_data,), daemon=True).start()
        except Exception as e:
            self.get_logger().error(f"❌ Error leyendo WAV: {e}")

    def _play_on_robot_thread(self, wav_data: bytes) -> None:
        with self._lock:
            self._play_on_robot(wav_data)

    def _synthesize_and_play(self, text: str) -> None:
        with self._lock:
            wav_data = self.piper.synthesize_wav(text)
            if wav_data is None:
                self.get_logger().error("❌ Piper no pudo generar el audio")
                return

            if self.local_playback:
                self._play_locally(wav_data)
            else:
                self._play_on_robot(wav_data)

    # ------------------------------------------------------------------
    def _play_locally(self, wav_data: bytes) -> None:
        """Reproduce el audio en los altavoces del PC usando paplay (PipeWire/PulseAudio)."""
        try:
            # paplay funciona correctamente en Ubuntu con PipeWire/Wayland
            proc = subprocess.Popen(
                ["paplay", "--raw", f"--rate={SAMPLE_RATE}",
                 "--format=s16le", "--channels=1"],
                stdin=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )
            # Pasamos solo los bytes PCM sin cabecera WAV
            raw_pcm = wav_data[44:]  # saltar cabecera WAV de 44 bytes
            proc.communicate(input=raw_pcm, timeout=30)
            self.get_logger().info("✅ Reproduccion local completada")
        except FileNotFoundError:
            # paplay no disponible, intentar con aplay como fallback
            try:
                proc = subprocess.Popen(
                    ["aplay", "-f", "S16_LE", "-r", str(SAMPLE_RATE), "-t", "wav"],
                    stdin=subprocess.PIPE, stderr=subprocess.DEVNULL
                )
                proc.communicate(input=wav_data, timeout=30)
                self.get_logger().info("✅ Reproduccion local completada (aplay)")
            except Exception as e2:
                self.get_logger().error(f"❌ Error reproduccion local: {e2}")
        except Exception as e:
            self.get_logger().error(f"❌ Error reproduccion local: {e}")

    def _play_on_robot(self, wav_data: bytes) -> None:
        """Manda el WAV al altavoz del Go2 via WebRTC."""
        try:
            # Calcular duracion aproximada
            pcm_bytes = len(wav_data) - 44  # sin cabecera WAV
            duration  = pcm_bytes / (SAMPLE_RATE * 2)  # 2 bytes/sample mono

            # Codificar en base64 y partir en chunks
            b64 = base64.b64encode(wav_data).decode("utf-8")
            chunks: List[str] = [
                b64[i:i + self.chunk_size]
                for i in range(0, len(b64), self.chunk_size)
            ]
            total = len(chunks)

            self.get_logger().info(f"📤 Enviando al robot: {total} chunks, {duration:.1f}s")

            # Inicio de transmision
            self._send_audio_cmd(4001, "")
            time.sleep(0.1)

            # Enviar chunks
            for idx, chunk in enumerate(chunks, 1):
                block = {
                    "current_block_index": idx,
                    "total_block_number": total,
                    "block_content": chunk,
                }
                self._send_audio_cmd(4003, json.dumps(block))
                if idx % 10 == 0:
                    self.get_logger().info(f"   {idx}/{total} chunks")
                time.sleep(0.15)

            # Esperar reproduccion
            self.get_logger().info(f"⏳ Esperando {duration:.1f}s de reproduccion...")
            time.sleep(duration + 1.0)

            # Fin de transmision
            self._send_audio_cmd(4002, "")
            self.get_logger().info("✅ Reproduccion en robot completada")

        except Exception as e:
            self.get_logger().error(f"❌ Error reproduccion robot: {e}")

    def _send_audio_cmd(self, api_id: int, parameter: str) -> None:
        req = WebRtcReq()
        req.api_id   = api_id
        req.priority = 0
        req.parameter = parameter
        req.topic = self.RTC_TOPIC["AUDIO_HUB_REQ"]  # string: "rt/api/audiohub/request"
        self.audio_pub.publish(req)


# ---------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    try:
        node = TTSNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"❌ TTS Node error: {e}")
    finally:
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()