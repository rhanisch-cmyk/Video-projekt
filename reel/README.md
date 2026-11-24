# Leadership-Vortrag „Change“ – Reel

**`Leadership-Change-Reel.mp4`**: 1080×1920 (9:16), 66 s, H.264/AAC, −14 LUFS, burned-in German captions.

Source: `Leadership-Vortrag - Change - Hanisch, der Projekt-Doctor.mp4` (Dropbox, 2:17, 1920×1080).

## Story (source timecodes)
| Reel | Source | Content |
|---|---|---|
| 0:00 | 0:49.6–0:52.1 | Hook: „…den Auftrag bekomme, einen Standort zu schließen.“ |
| 0:02 | 1:19.6–1:26.0 | „Einen Tag bevor das offiziell verkündet wird … nee, an unserem Standort kann das nicht passieren.“ |
| 0:09 | 0:32.2–0:43.2 | „Fakt: wenn wir nicht permanent daran arbeiten, uns zu verändern …“ |
| 0:20 | 1:26.6–2:05.0 | Leitsatz „Nichts bleibt bestehen, außer eine Überprüfung rechtfertigt den derzeitigen Zustand.“ + explanation |
| 0:58 | 2:05.0–2:12.6 | End card: Vortrag & Beratung / Leadership-Training / Ronald Hanisch – Projekt-Doctor |

Speaker shots use a face-tracked 9:16 crop; text slides and end cards are shown full width over a blurred fill.
Opening hook text: „Wer nicht transformiert, muss restrukturieren.“

## Rebuild
```
pip install faster-whisper "opencv-python-headless<5"
python3 build_reel.py SOURCE.mp4 transcript.json OUT_DIR
```
Edit `PIECES` in `build_reel.py` to change cuts/framing.
