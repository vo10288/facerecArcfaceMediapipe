#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Face Compare 1:1 - Forensic Edition
Linux + macOS + Windows

Pipeline:
- OpenCV: caricamento immagini, crop, overlay, salvataggi
- MediaPipe Face Mesh: keypoint/landmark del volto
- InsightFace / ArcFace: embedding e confronto biometrico 1:1
- Tkinter: GUI cross-platform

Funzioni:
- Carica due immagini
- Mostra per ciascuna immagine:
  1) immagine originale
  2) immagine con keypoint gialli
  3) crop del volto principale
- Calcola similarita biometrica con ArcFace (cosine similarity)
- Soglia di compatibilita configurabile da GUI
- Esporta HTML
- Esporta CSV
- Esporta pacchetto forense con timestamp
- Calcola hash MD5 e SHA256 delle immagini sorgenti
- Salva copie originali, annotate e crop
- Aggiorna log storico CSV
"""

from __future__ import annotations

import base64
import csv
import hashlib
import os
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from shutil import copy2
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox, ttk

try:
    from insightface.app import FaceAnalysis
except Exception as exc:
    FaceAnalysis = None
    INSIGHTFACE_ERROR = exc
else:
    INSIGHTFACE_ERROR = None

APP_TITLE = "Face Compare 1:1 - MediaPipe + ArcFace Forensic"
DEFAULT_THRESHOLD = 0.55
EXPORT_DIRNAME = "face_compare_exports"
HISTORY_CSV = "face_compare_history.csv"
WINDOW_BG = "#111111"
PANEL_BG = "#1b1b1b"
TEXT_FG = "#f3f3f3"
MUTED = "#bbbbbb"
ACCENT = "#ffd400"
PREVIEW_SIZE = (320, 240)
CROP_SIZE = (220, 220)
YELLOW_BGR = (0, 255, 255)


@dataclass
class FaceResult:
    image_path: str
    original_bgr: np.ndarray
    annotated_bgr: np.ndarray
    crop_bgr: Optional[np.ndarray]
    embedding: Optional[np.ndarray]
    similarity_ready: bool
    face_count_arcface: int
    mp_landmark_count: int
    bbox: Optional[Tuple[int, int, int, int]]
    warning: str


class FaceCompareApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1680x980")
        self.root.minsize(1280, 800)
        self.root.configure(bg=WINDOW_BG)

        self.path_a: Optional[str] = None
        self.path_b: Optional[str] = None
        self.result_a: Optional[FaceResult] = None
        self.result_b: Optional[FaceResult] = None
        self.tk_refs = []
        self.threshold_var = tk.StringVar(value=f"{DEFAULT_THRESHOLD:.2f}")

        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.arcface = self._init_arcface()
        self._build_ui()

    def _init_arcface(self):
        if FaceAnalysis is None:
            raise RuntimeError(f"InsightFace non disponibile: {INSIGHTFACE_ERROR}")
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
        return app

    def _build_ui(self) -> None:
        top = tk.Frame(self.root, bg=WINDOW_BG)
        top.pack(fill="x", padx=12, pady=12)

        tk.Label(top, text=APP_TITLE, bg=WINDOW_BG, fg=TEXT_FG, font=("Helvetica", 18, "bold")).pack(anchor="w")
        tk.Label(
            top,
            text="Originale + keypoint gialli + crop volto | MediaPipe per landmark | ArcFace per confronto | export forense",
            bg=WINDOW_BG,
            fg=MUTED,
            font=("Helvetica", 10),
        ).pack(anchor="w", pady=(4, 0))

        controls = tk.Frame(self.root, bg=WINDOW_BG)
        controls.pack(fill="x", padx=12, pady=(0, 10))

        ttk.Button(controls, text="Apri Immagine A", command=self.load_a).pack(side="left", padx=4)
        ttk.Button(controls, text="Apri Immagine B", command=self.load_b).pack(side="left", padx=4)
        ttk.Button(controls, text="Confronta", command=self.compare).pack(side="left", padx=4)
        ttk.Button(controls, text="Esporta HTML", command=self.export_html).pack(side="left", padx=4)
        ttk.Button(controls, text="Esporta CSV", command=self.export_csv).pack(side="left", padx=4)
        ttk.Button(controls, text="Esporta Pacchetto", command=self.export_package).pack(side="left", padx=4)
        tk.Label(controls, text="Soglia:", bg=WINDOW_BG, fg=TEXT_FG).pack(side="left", padx=(18, 4))
        tk.Entry(controls, textvariable=self.threshold_var, width=8, justify="center").pack(side="left", padx=4)
        ttk.Button(controls, text="Esci", command=self.root.destroy).pack(side="right", padx=4)

        self.status_var = tk.StringVar(value="Carica due immagini e premi Confronta.")
        tk.Label(
            self.root,
            textvariable=self.status_var,
            bg=WINDOW_BG,
            fg=ACCENT,
            anchor="w",
            justify="left",
            font=("Helvetica", 11, "bold"),
        ).pack(fill="x", padx=14, pady=(0, 8))

        main = tk.Frame(self.root, bg=WINDOW_BG)
        main.pack(fill="both", expand=True, padx=10, pady=8)

        self.panel_a = self._make_side(main, "Immagine A")
        self.panel_a["frame"].pack(side="left", fill="both", expand=True, padx=6)

        self.panel_b = self._make_side(main, "Immagine B")
        self.panel_b["frame"].pack(side="left", fill="both", expand=True, padx=6)

        bottom = tk.Frame(self.root, bg=WINDOW_BG)
        bottom.pack(fill="x", padx=12, pady=(0, 12))

        self.metrics = tk.Text(
            bottom,
            height=14,
            bg="#191919",
            fg=TEXT_FG,
            insertbackground=TEXT_FG,
            relief="flat",
            wrap="word",
            font=("Courier", 11),
        )
        self.metrics.pack(fill="both", expand=True)
        self._set_metrics("Pronto.\n")

    def _make_side(self, parent: tk.Widget, title: str):
        frame = tk.LabelFrame(parent, text=title, bg=WINDOW_BG, fg=TEXT_FG, font=("Helvetica", 12, "bold"), padx=8, pady=8, bd=2)
        grid = tk.Frame(frame, bg=WINDOW_BG)
        grid.pack(fill="both", expand=True)

        original = self._make_image_box(grid, "Originale")
        original["frame"].grid(row=0, column=0, padx=6, pady=6, sticky="nsew")
        annotated = self._make_image_box(grid, "Keypoint gialli")
        annotated["frame"].grid(row=0, column=1, padx=6, pady=6, sticky="nsew")
        crop = self._make_image_box(grid, "Crop volto")
        crop["frame"].grid(row=1, column=0, padx=6, pady=6, sticky="nsew")

        info = tk.Text(grid, height=10, width=42, bg=PANEL_BG, fg=TEXT_FG, relief="flat", wrap="word", font=("Courier", 10))
        info.grid(row=1, column=1, padx=6, pady=6, sticky="nsew")

        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)
        grid.rowconfigure(0, weight=1)
        grid.rowconfigure(1, weight=1)

        return {"frame": frame, "original": original["label"], "annotated": annotated["label"], "crop": crop["label"], "info": info}

    def _make_image_box(self, parent: tk.Widget, title: str):
        frame = tk.LabelFrame(parent, text=title, bg=WINDOW_BG, fg=MUTED, padx=6, pady=6)
        label = tk.Label(frame, text="Nessuna immagine", bg=PANEL_BG, fg=MUTED, width=42, height=16)
        label.pack(fill="both", expand=True)
        return {"frame": frame, "label": label}

    def _get_threshold(self) -> float:
        try:
            value = float(self.threshold_var.get().strip().replace(",", "."))
        except Exception:
            value = DEFAULT_THRESHOLD
        return max(0.0, min(1.0, value))

    def load_a(self) -> None:
        self.path_a = self._pick_image()
        if self.path_a:
            self.status_var.set(f"Immagine A caricata: {os.path.basename(self.path_a)}")
            self.result_a = None

    def load_b(self) -> None:
        self.path_b = self._pick_image()
        if self.path_b:
            self.status_var.set(f"Immagine B caricata: {os.path.basename(self.path_b)}")
            self.result_b = None

    def _pick_image(self) -> Optional[str]:
        return filedialog.askopenfilename(
            title="Seleziona immagine",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff"), ("All files", "*.*")],
        )

    def compare(self) -> None:
        try:
            if not self.path_a or not self.path_b:
                messagebox.showwarning("Attenzione", "Seleziona entrambe le immagini.")
                return

            self.result_a = self._process_image(self.path_a)
            self.result_b = self._process_image(self.path_b)

            self._render_result(self.panel_a, self.result_a)
            self._render_result(self.panel_b, self.result_b)

            similarity_text = "N/D"
            verdict = "Impossibile confrontare"
            similarity = None
            if self.result_a.embedding is not None and self.result_b.embedding is not None:
                similarity = cosine_similarity(self.result_a.embedding, self.result_b.embedding)
                similarity_text = f"{similarity:.4f}"
                verdict = self._interpret_similarity(similarity)

            lines = []
            lines.append("=== RISULTATO COMPARAZIONE 1:1 ===")
            lines.append(f"File A: {self.result_a.image_path}")
            lines.append(f"File B: {self.result_b.image_path}")
            lines.append("")
            lines.append(f"Somiglianza biometrica (ArcFace): {similarity_text}")
            lines.append(f"Soglia impostata: {self._get_threshold():.2f}")
            lines.append(f"Valutazione: {verdict}")
            lines.append(f"MD5 A: {file_md5(self.result_a.image_path)}")
            lines.append(f"SHA256 A: {file_sha256(self.result_a.image_path)}")
            lines.append(f"MD5 B: {file_md5(self.result_b.image_path)}")
            lines.append(f"SHA256 B: {file_sha256(self.result_b.image_path)}")
            lines.append("")
            lines.append("--- Dettagli immagine A ---")
            lines.extend(self._summary_lines(self.result_a))
            lines.append("")
            lines.append("--- Dettagli immagine B ---")
            lines.extend(self._summary_lines(self.result_b))
            self._set_metrics("\n".join(lines))

            if similarity is None:
                self.status_var.set("Confronto terminato con limitazioni: embedding non disponibile su una o entrambe le immagini.")
            else:
                self.status_var.set(f"Confronto terminato. Similarita ArcFace: {similarity:.4f} | soglia {self._get_threshold():.2f} | {verdict}")
        except Exception as exc:
            traceback.print_exc()
            messagebox.showerror("Errore", f"Errore durante il confronto:\n{exc}")

    def _interpret_similarity(self, score: float) -> str:
        threshold = self._get_threshold()
        medium = max(0.0, threshold - 0.20)
        if score >= threshold:
            return "Compatibilita alta"
        if score >= medium:
            return "Compatibilita media - valutare con cautela"
        return "Compatibilita bassa"

    def _summary_lines(self, result: FaceResult):
        bbox_text = str(result.bbox) if result.bbox else "Nessuna"
        return [
            f"Volti rilevati da ArcFace: {result.face_count_arcface}",
            f"Landmark MediaPipe: {result.mp_landmark_count}",
            f"Bounding box: {bbox_text}",
            f"Embedding pronto: {'Si' if result.similarity_ready else 'No'}",
            f"Note: {result.warning or 'Nessun avviso'}",
        ]

    def _process_image(self, path: str) -> FaceResult:
        bgr = cv2.imread(path)
        if bgr is None:
            raise RuntimeError(f"Impossibile leggere il file: {path}")

        annotated = bgr.copy()
        crop = None
        bbox = None
        warning = ""
        mp_landmark_count = 0
        embedding = None
        similarity_ready = False

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_result = self.mp_face_mesh.process(rgb)

        if mp_result.multi_face_landmarks:
            h, w = bgr.shape[:2]
            face_landmarks = mp_result.multi_face_landmarks[0]
            mp_landmark_count = len(face_landmarks.landmark)
            xs = []
            ys = []
            for lm in face_landmarks.landmark:
                x = int(lm.x * w)
                y = int(lm.y * h)
                xs.append(x)
                ys.append(y)
                cv2.circle(annotated, (x, y), 1, YELLOW_BGR, -1, lineType=cv2.LINE_AA)
            if xs and ys:
                x1 = max(min(xs) - 20, 0)
                y1 = max(min(ys) - 20, 0)
                x2 = min(max(xs) + 20, w)
                y2 = min(max(ys) + 20, h)
                bbox = (x1, y1, x2, y2)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), YELLOW_BGR, 2)
                crop = bgr[y1:y2, x1:x2].copy() if y2 > y1 and x2 > x1 else None
        else:
            warning = "MediaPipe non ha rilevato landmark facciali."

        arc_faces = self.arcface.get(bgr)
        face_count_arcface = len(arc_faces)
        if face_count_arcface > 0:
            best = max(arc_faces, key=lambda f: area_from_bbox(f.bbox))
            emb = getattr(best, "embedding", None)
            if emb is not None:
                embedding = np.asarray(emb, dtype=np.float32)
                similarity_ready = True
            if bbox is None:
                abox = best.bbox.astype(int)
                x1, y1, x2, y2 = map(int, abox)
                x1 = max(x1, 0)
                y1 = max(y1, 0)
                x2 = min(x2, bgr.shape[1])
                y2 = min(y2, bgr.shape[0])
                bbox = (x1, y1, x2, y2)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), YELLOW_BGR, 2)
                if y2 > y1 and x2 > x1:
                    crop = bgr[y1:y2, x1:x2].copy()
        else:
            warning = join_notes(warning, "ArcFace non ha rilevato volti validi.")

        if face_count_arcface > 1:
            warning = join_notes(warning, "Sono stati rilevati piu volti: usato il volto principale.")
        if crop is None:
            warning = join_notes(warning, "Crop volto non disponibile.")

        return FaceResult(
            image_path=path,
            original_bgr=bgr,
            annotated_bgr=annotated,
            crop_bgr=crop,
            embedding=embedding,
            similarity_ready=similarity_ready,
            face_count_arcface=face_count_arcface,
            mp_landmark_count=mp_landmark_count,
            bbox=bbox,
            warning=warning,
        )

    def _render_result(self, panel, result: FaceResult) -> None:
        self._set_label_image(panel["original"], result.original_bgr, PREVIEW_SIZE)
        self._set_label_image(panel["annotated"], result.annotated_bgr, PREVIEW_SIZE)
        self._set_label_image(panel["crop"], result.crop_bgr, CROP_SIZE)
        info_lines = [
            f"File: {os.path.basename(result.image_path)}",
            f"ArcFace volti: {result.face_count_arcface}",
            f"MediaPipe landmark: {result.mp_landmark_count}",
            f"BBox: {result.bbox if result.bbox else 'Nessuna'}",
            f"Embedding ArcFace: {'Si' if result.similarity_ready else 'No'}",
            f"MD5: {file_md5(result.image_path)}",
            f"SHA256: {file_sha256(result.image_path)}",
            f"Avvisi: {result.warning or 'Nessuno'}",
        ]
        panel["info"].delete("1.0", tk.END)
        panel["info"].insert("1.0", "\n".join(info_lines))

    def _set_label_image(self, label: tk.Label, bgr: Optional[np.ndarray], size: Tuple[int, int]) -> None:
        if bgr is None:
            label.configure(image="", text="Non disponibile")
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil.thumbnail(size, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(pil)
        self.tk_refs.append(photo)
        label.configure(image=photo, text="")
        label.image = photo

    def _set_metrics(self, text: str) -> None:
        self.metrics.delete("1.0", tk.END)
        self.metrics.insert("1.0", text)

    def export_csv(self) -> None:
        try:
            if not self.result_a or not self.result_b:
                messagebox.showwarning("Attenzione", "Esegui prima il confronto.")
                return
            save_path = filedialog.asksaveasfilename(
                title="Salva report CSV",
                defaultextension=".csv",
                initialfile=f"face_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                filetypes=[("CSV", "*.csv")],
            )
            if not save_path:
                return

            similarity = ""
            verdict = ""
            if self.result_a.embedding is not None and self.result_b.embedding is not None:
                score = cosine_similarity(self.result_a.embedding, self.result_b.embedding)
                similarity = f"{score:.6f}"
                verdict = self._interpret_similarity(score)

            rows = [
                "timestamp;image_a;image_b;similarity_arcface;threshold;verdict;faces_a;faces_b;landmarks_a;landmarks_b;warning_a;warning_b;md5_a;sha256_a;md5_b;sha256_b",
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')};{self.result_a.image_path};{self.result_b.image_path};{similarity};{self._get_threshold():.2f};{verdict};{self.result_a.face_count_arcface};{self.result_b.face_count_arcface};{self.result_a.mp_landmark_count};{self.result_b.mp_landmark_count};{self.result_a.warning};{self.result_b.warning};{file_md5(self.result_a.image_path)};{file_sha256(self.result_a.image_path)};{file_md5(self.result_b.image_path)};{file_sha256(self.result_b.image_path)}",
            ]
            Path(save_path).write_text("\n".join(rows), encoding="utf-8")
            messagebox.showinfo("Completato", f"CSV salvato in:\n{save_path}")
        except Exception as exc:
            traceback.print_exc()
            messagebox.showerror("Errore", f"Impossibile esportare CSV:\n{exc}")

    def export_package(self) -> None:
        try:
            if not self.result_a or not self.result_b:
                messagebox.showwarning("Attenzione", "Esegui prima il confronto.")
                return
            base_dir = filedialog.askdirectory(title="Seleziona directory di destinazione")
            if not base_dir:
                return

            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            out_dir = Path(base_dir) / EXPORT_DIRNAME / f"compare_{ts}"
            out_dir.mkdir(parents=True, exist_ok=True)
            originals_dir = out_dir / "originals"
            annotated_dir = out_dir / "annotated"
            crops_dir = out_dir / "crops"
            originals_dir.mkdir(exist_ok=True)
            annotated_dir.mkdir(exist_ok=True)
            crops_dir.mkdir(exist_ok=True)

            a_name = Path(self.result_a.image_path).name
            b_name = Path(self.result_b.image_path).name
            copy2(self.result_a.image_path, originals_dir / f"A_{a_name}")
            copy2(self.result_b.image_path, originals_dir / f"B_{b_name}")
            cv2.imwrite(str(annotated_dir / "A_annotated.png"), self.result_a.annotated_bgr)
            cv2.imwrite(str(annotated_dir / "B_annotated.png"), self.result_b.annotated_bgr)
            if self.result_a.crop_bgr is not None:
                cv2.imwrite(str(crops_dir / "A_crop.png"), self.result_a.crop_bgr)
            if self.result_b.crop_bgr is not None:
                cv2.imwrite(str(crops_dir / "B_crop.png"), self.result_b.crop_bgr)

            similarity = None
            verdict = ""
            if self.result_a.embedding is not None and self.result_b.embedding is not None:
                similarity = cosine_similarity(self.result_a.embedding, self.result_b.embedding)
                verdict = self._interpret_similarity(similarity)

            (out_dir / "report.html").write_text(self._build_html(similarity), encoding="utf-8")
            csv_content = [
                "timestamp;image_a;image_b;similarity_arcface;threshold;verdict;faces_a;faces_b;landmarks_a;landmarks_b;warning_a;warning_b;md5_a;sha256_a;md5_b;sha256_b",
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')};{self.result_a.image_path};{self.result_b.image_path};{'' if similarity is None else f'{similarity:.6f}'};{self._get_threshold():.2f};{verdict};{self.result_a.face_count_arcface};{self.result_b.face_count_arcface};{self.result_a.mp_landmark_count};{self.result_b.mp_landmark_count};{self.result_a.warning};{self.result_b.warning};{file_md5(self.result_a.image_path)};{file_sha256(self.result_a.image_path)};{file_md5(self.result_b.image_path)};{file_sha256(self.result_b.image_path)}",
            ]
            (out_dir / "report.csv").write_text("\n".join(csv_content), encoding="utf-8")
            self._append_history_csv(out_dir, similarity, verdict)
            messagebox.showinfo("Completato", f"Pacchetto esportato in:\n{out_dir}")
        except Exception as exc:
            traceback.print_exc()
            messagebox.showerror("Errore", f"Impossibile esportare pacchetto:\n{exc}")

    def _append_history_csv(self, out_dir: Path, similarity: Optional[float], verdict: str) -> None:
        history_path = out_dir.parent / HISTORY_CSV
        header = [
            "timestamp", "image_a", "image_b", "similarity_arcface", "threshold", "verdict",
            "md5_a", "sha256_a", "md5_b", "sha256_b", "export_dir"
        ]
        row = [
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            self.result_a.image_path,
            self.result_b.image_path,
            "" if similarity is None else f"{similarity:.6f}",
            f"{self._get_threshold():.2f}",
            verdict,
            file_md5(self.result_a.image_path),
            file_sha256(self.result_a.image_path),
            file_md5(self.result_b.image_path),
            file_sha256(self.result_b.image_path),
            str(out_dir),
        ]
        append_csv_row(history_path, header, row)

    def export_html(self) -> None:
        try:
            if not self.result_a or not self.result_b:
                messagebox.showwarning("Attenzione", "Esegui prima il confronto.")
                return
            save_path = filedialog.asksaveasfilename(
                title="Salva report HTML",
                defaultextension=".html",
                initialfile=f"face_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                filetypes=[("HTML", "*.html")],
            )
            if not save_path:
                return
            similarity = None
            if self.result_a.embedding is not None and self.result_b.embedding is not None:
                similarity = cosine_similarity(self.result_a.embedding, self.result_b.embedding)
            Path(save_path).write_text(self._build_html(similarity), encoding="utf-8")
            messagebox.showinfo("Completato", f"Report salvato in:\n{save_path}")
        except Exception as exc:
            traceback.print_exc()
            messagebox.showerror("Errore", f"Impossibile esportare HTML:\n{exc}")

    def _build_html(self, similarity: Optional[float]) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        verdict = self._interpret_similarity(similarity) if similarity is not None else "N/D"
        sim_text = f"{similarity:.4f}" if similarity is not None else "N/D"
        a_orig = img_to_b64(self.result_a.original_bgr)
        a_ann = img_to_b64(self.result_a.annotated_bgr)
        a_crop = img_to_b64(self.result_a.crop_bgr)
        b_orig = img_to_b64(self.result_b.original_bgr)
        b_ann = img_to_b64(self.result_b.annotated_bgr)
        b_crop = img_to_b64(self.result_b.crop_bgr)

        return f"""<!DOCTYPE html>
<html lang='it'>
<head>
<meta charset='utf-8'>
<title>Face Compare Report</title>
<style>
body {{ background:#111; color:#f0f0f0; font-family:Arial,Helvetica,sans-serif; margin:20px; }}
.card {{ background:#1b1b1b; border:1px solid #333; border-radius:12px; padding:16px; margin-bottom:20px; }}
.grid {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px; }}
img {{ max-width:100%; border-radius:8px; border:1px solid #444; }}
.h {{ color:#ffd400; }}
.small {{ color:#bbb; font-size:13px; }}
pre {{ white-space:pre-wrap; background:#161616; padding:12px; border-radius:8px; }}
</style>
</head>
<body>
<h1 class='h'>Face Compare 1:1 - MediaPipe + ArcFace</h1>
<p class='small'>Generato il {now}</p>
<div class='card'>
<h2>Esito confronto</h2>
<p><b>Somiglianza biometrica (ArcFace):</b> {sim_text}</p>
<p><b>Soglia impostata:</b> {self._get_threshold():.2f}</p>
<p><b>Valutazione:</b> {verdict}</p>
</div>
<div class='card'>
<h2>Immagine A</h2>
<p class='small'>{self.result_a.image_path}</p>
<div class='grid'>
<div><h3>Originale</h3><img src='data:image/png;base64,{a_orig}'></div>
<div><h3>Keypoint gialli</h3><img src='data:image/png;base64,{a_ann}'></div>
<div><h3>Crop volto</h3><img src='data:image/png;base64,{a_crop}'></div>
</div>
<pre>{chr(10).join(self._summary_lines(self.result_a))}
MD5: {file_md5(self.result_a.image_path)}
SHA256: {file_sha256(self.result_a.image_path)}</pre>
</div>
<div class='card'>
<h2>Immagine B</h2>
<p class='small'>{self.result_b.image_path}</p>
<div class='grid'>
<div><h3>Originale</h3><img src='data:image/png;base64,{b_orig}'></div>
<div><h3>Keypoint gialli</h3><img src='data:image/png;base64,{b_ann}'></div>
<div><h3>Crop volto</h3><img src='data:image/png;base64,{b_crop}'></div>
</div>
<pre>{chr(10).join(self._summary_lines(self.result_b))}
MD5: {file_md5(self.result_b.image_path)}
SHA256: {file_sha256(self.result_b.image_path)}</pre>
</div>
</body>
</html>
"""


def file_md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def area_from_bbox(bbox) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


def join_notes(a: str, b: str) -> str:
    if a and b:
        return a + " | " + b
    return a or b


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    v1 = v1.astype(np.float32)
    v2 = v2.astype(np.float32)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def append_csv_row(csv_path: Path, header: list[str], row: list[str]) -> None:
    exists = csv_path.exists()
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')
        if not exists:
            writer.writerow(header)
        writer.writerow(row)


def img_to_b64(bgr: Optional[np.ndarray]) -> str:
    if bgr is None:
        blank = np.zeros((220, 220, 3), dtype=np.uint8)
        cv2.putText(blank, "N/D", (70, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        bgr = blank
    ok, buf = cv2.imencode('.png', bgr)
    if not ok:
        raise RuntimeError('Impossibile convertire immagine in PNG')
    return base64.b64encode(buf.tobytes()).decode('ascii')


def main() -> None:
    root = tk.Tk()
    style = ttk.Style()
    try:
        style.theme_use('clam')
    except Exception:
        pass
    FaceCompareApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
