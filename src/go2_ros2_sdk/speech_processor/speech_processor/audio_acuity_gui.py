#!/usr/bin/env python3
# Copyright (c) 2026, RoboVerse community
# SPDX-License-Identifier: BSD-3-Clause

import os
import sys
import time
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# Colors and style configuration
BG_COLOR = "#0c0d14"          # Deep cosmic dark background
CARD_COLOR = "#141622"        # Slate card container
ACCENT_CYAN = "#00f2fe"       # Neon cyan highlight
ACCENT_BLUE = "#4facfe"       # Electric blue highlight
ACCENT_RED = "#ff3b30"        # Neon red recording indicator
TEXT_MAIN = "#ffffff"         # Muted white text
TEXT_MUTED = "#8a95a5"        # Muted gray text
FONT_FAMILY = "Helvetica"         # Modern typography (fallback to system default sans-serif)

class AudioAcuityNode(Node):
    """ROS2 Node for the Audio Acuity GUI"""
    def __init__(self):
        super().__init__("audio_acuity_gui_node")
        self.tts_pub = self.create_publisher(String, "/tts", 10)
        self.play_wav_pub = self.create_publisher(String, "/play_wav_on_robot", 10)
        self.get_logger().info("Audio Acuity GUI Node initialized")

    def send_tts(self, text: str):
        msg = String()
        msg.data = text
        self.tts_pub.publish(msg)
        self.get_logger().info(f"Published TTS message: '{text}'")

    def play_wav_on_robot(self, filepath: str):
        msg = String()
        msg.data = filepath
        self.play_wav_pub.publish(msg)
        self.get_logger().info(f"Published play WAV request: {filepath}")


class AudioAcuityApp:
    def __init__(self, root, node):
        self.root = root
        self.node = node
        self.record_proc = None
        self.recording = False
        self.wav_path = "/tmp/operator_voice.wav"

        # Window settings
        self.root.title("RoboCup - 2-Way Audio Acuity Control Panel")
        self.root.geometry("640x520")
        self.root.configure(bg=BG_COLOR)
        self.root.resizable(False, False)

        # Style customization
        self.style = ttk.Style()
        self.style.theme_use("clam")
        
        # Delay UI building slightly to let Tkinter / X11 fully initialize
        self.root.after(100, self._build_ui)
        
        # Start monitoring loop
        self.root.after(150, self._update_loop)

    def _build_ui(self):
        # Header banner
        header_frame = tk.Frame(self.root, bg=BG_COLOR, pady=20)
        header_frame.pack(fill="x", padx=20)
        
        title_label = tk.Label(
            header_frame, 
            text="2-WAY AUDIO ACUITY", 
            font=f"{FONT_FAMILY} 16 bold", 
            fg=ACCENT_CYAN, 
            bg=BG_COLOR
        )
        title_label.pack(anchor="w")
        
        desc_label = tk.Label(
            header_frame, 
            text="RoboCup Rescue - Operator Station Audio Controller", 
            font=f"{FONT_FAMILY} 10", 
            fg=TEXT_MUTED, 
            bg=BG_COLOR
        )
        desc_label.pack(anchor="w")

        # Main Card Frame
        main_card = tk.Frame(self.root, bg=CARD_COLOR, bd=1, relief="flat", highlightthickness=1, highlightbackground="#222638")
        main_card.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        # Status section
        status_frame = tk.Frame(main_card, bg=CARD_COLOR, pady=15, padx=20)
        status_frame.pack(fill="x")
        
        status_label = tk.Label(
            status_frame, 
            text="AMBIENT LISTENING STATUS:", 
            font=f"{FONT_FAMILY} 9 bold", 
            fg=TEXT_MUTED, 
            bg=CARD_COLOR
        )
        status_label.pack(side="left")

        self.status_val = tk.Label(
            status_frame, 
            text="🟢 ACTIVE (Robot Mic Stream)", 
            font=f"{FONT_FAMILY} 9 bold", 
            fg="#4cd964", 
            bg=CARD_COLOR
        )
        self.status_val.pack(side="left", padx=10)

        # Separator line
        sep = tk.Frame(main_card, height=1, bg="#222638")
        sep.pack(fill="x", padx=20)

        # Push to Talk Section
        ptt_frame = tk.Frame(main_card, bg=CARD_COLOR, pady=25)
        ptt_frame.pack(fill="x")

        self.ptt_btn = tk.Button(
            ptt_frame,
            text="🎤  PUSH TO TALK",
            font=f"{FONT_FAMILY} 14 bold",
            bg="#222638",
            fg=TEXT_MAIN,
            activebackground=ACCENT_CYAN,
            activeforeground="#000000",
            relief="flat",
            bd=0,
            padx=30,
            pady=15,
            cursor="hand2"
        )
        self.ptt_btn.pack()
        
        # Bind events for push-to-talk
        self.ptt_btn.bind("<ButtonPress-1>", self._start_recording)
        self.ptt_btn.bind("<ButtonRelease-1>", self._stop_recording)

        self.ptt_info = tk.Label(
            ptt_frame,
            text="Press and hold to record operator's voice. Release to play on robot speaker.",
            font=f"{FONT_FAMILY} 9 italic",
            fg=TEXT_MUTED,
            bg=CARD_COLOR,
            pady=10
        )
        self.ptt_info.pack()

        # Separator line
        sep2 = tk.Frame(main_card, height=1, bg="#222638")
        sep2.pack(fill="x", padx=20)

        # Text to Speech Section
        tts_frame = tk.Frame(main_card, bg=CARD_COLOR, pady=20, padx=20)
        tts_frame.pack(fill="both", expand=True)

        tts_label = tk.Label(
            tts_frame,
            text="TEXT-TO-SPEECH (OFFLINE PIPER)",
            font=f"{FONT_FAMILY} 9 bold",
            fg=ACCENT_BLUE,
            bg=CARD_COLOR
        )
        tts_label.pack(anchor="w", pady=(0, 5))

        # TTS Input box
        self.tts_entry = tk.Entry(
            tts_frame,
            font=f"{FONT_FAMILY} 11",
            bg=BG_COLOR,
            fg=TEXT_MAIN,
            insertbackground=TEXT_MAIN,
            bd=0,
            highlightthickness=1,
            highlightbackground="#222638",
            highlightcolor=ACCENT_BLUE
        )
        self.tts_entry.pack(fill="x", ipady=8, pady=(0, 10))
        self.tts_entry.insert(0, "Hola, soy el perro robot. Escucho correctamente.")

        # Send TTS button
        send_btn = tk.Button(
            tts_frame,
            text="⚡ SEND TTS",
            font=f"{FONT_FAMILY} 10 bold",
            bg="#222638",
            fg=ACCENT_BLUE,
            activebackground=ACCENT_BLUE,
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            pady=8,
            cursor="hand2",
            command=self._send_tts
        )
        send_btn.pack(fill="x")

        # Footer Status bar
        self.footer_status = tk.Label(
            self.root,
            text="System status: Connected to ROS2. Ready.",
            font=f"{FONT_FAMILY} 9",
            fg=TEXT_MUTED,
            bg=BG_COLOR,
            anchor="w",
            padx=20,
            pady=10
        )
        self.footer_status.pack(fill="x", side="bottom")

    def _start_recording(self, event=None):
        if self.recording:
            return
        self.recording = True
        
        # Change UI style to recording state
        self.ptt_btn.configure(bg=ACCENT_RED, fg="#ffffff", text="🔴 RECORDING...")
        self.status_val.configure(text="⏸️ TALKING (PTT active)", fg=ACCENT_RED)
        self.footer_status.configure(text="Recording audio from local microphone...")

        # Delete previous recording file if exists
        if os.path.exists(self.wav_path):
            try:
                os.remove(self.wav_path)
            except Exception:
                pass

        # Spawn arecord process in background
        # 16kHz, mono, WAV format
        try:
            self.record_proc = subprocess.Popen(
                ["arecord", "-f", "S16_LE", "-r", "16000", "-c", "1", "-t", "wav", self.wav_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except FileNotFoundError:
            self.recording = False
            self._reset_ptt_button()
            messagebox.showerror("Error", "arecord is not installed on this system. Cannot record voice.")

    def _stop_recording(self, event=None):
        if not self.recording:
            return
        self.recording = False

        self.ptt_btn.configure(bg="#222638", fg=TEXT_MAIN, text="⌛ PROCESSING...")
        self.root.update()

        # Stop subprocess
        if self.record_proc:
            try:
                self.record_proc.terminate()
                self.record_proc.wait(timeout=1.0)
            except Exception:
                pass
            self.record_proc = None

        # Verify recording exists and has content
        if os.path.exists(self.wav_path) and os.path.getsize(self.wav_path) > 44:
            self.node.play_wav_on_robot(self.wav_path)
            self.footer_status.configure(text=f"Sent voice audio to robot speaker (Path: {self.wav_path})")
            
            # Show a brief "Voice Sent" feedback
            self.ptt_btn.configure(bg="#4cd964", fg="#ffffff", text="✅ VOICE TRANSMITTED")
            self.root.after(1500, self._reset_ptt_button)
        else:
            self.footer_status.configure(text="Recording canceled (no audio captured).")
            self._reset_ptt_button()

    def _reset_ptt_button(self):
        self.ptt_btn.configure(bg="#222638", fg=TEXT_MAIN, text="🎤  PUSH TO TALK")
        self.status_val.configure(text="🟢 ACTIVE (Robot Mic Stream)", fg="#4cd964")

    def _send_tts(self):
        text = self.tts_entry.get().strip()
        if not text:
            return
        self.node.send_tts(text)
        self.footer_status.configure(text=f"TTS request sent: '{text}'")

    def _update_loop(self):
        # Process ROS2 callbacks in the main thread to prevent segfaults
        try:
            rclpy.spin_once(self.node, timeout_sec=0.0)
        except Exception:
            pass
        self.root.after(10, self._update_loop)


def main(args=None):
    # Start Tkinter application FIRST
    root = tk.Tk()

    # Initialize ROS2 SECOND
    rclpy.init(args=args)
    node = AudioAcuityNode()

    app = AudioAcuityApp(root, node)

    try:
        root.mainloop()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
