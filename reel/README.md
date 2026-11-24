# Leadership-Vortrag Reels

## „Change“

**`Leadership-Change-Reel.mp4`**: 1080×1920 (9:16), 66 s, H.264/AAC, −14 LUFS, burned-in German captions.

Source: `Leadership-Vortrag - Change - Hanisch, der Projekt-Doctor.mp4` (Dropbox, 2:17, 1920×1080).

### Story (source timecodes)
| Reel | Source | Content |
|---|---|---|
| 0:00 | 0:49.6–0:52.1 | Hook: „…den Auftrag bekomme, einen Standort zu schließen.“ |
| 0:02 | 1:19.6–1:26.0 | „Einen Tag bevor das offiziell verkündet wird … nee, an unserem Standort kann das nicht passieren.“ |
| 0:09 | 0:32.2–0:43.2 | „Fakt: wenn wir nicht permanent daran arbeiten, uns zu verändern …“ |
| 0:20 | 1:26.6–2:05.0 | Leitsatz „Nichts bleibt bestehen, außer eine Überprüfung rechtfertigt den derzeitigen Zustand.“ + explanation |
| 0:58 | 2:05.0–2:12.6 | End card: Vortrag & Beratung / Leadership-Training / Ronald Hanisch – Projekt-Doctor |

Speaker shots use a face-tracked 9:16 crop; text slides and end cards are shown full width over a blurred fill.
Opening hook text: „Wer nicht transformiert, muss restrukturieren.“

## „Durchhalten“

**`Leadership-Durchhalten-Reel.mp4`**: 1080×1920 (9:16), 56 s, H.264/AAC, −14.5 LUFS, burned-in German captions.

Source: `Leadership-Vortrag - Durchhalten - Projekt-Doctor.mp4` (Dropbox, 1:32, 1920×1080).
Opening hook text: „Wer lang genug dabei bleibt, der schafft das.“

| Reel | Source | Content |
|---|---|---|
| 0:00 | 0:20.7–0:25.2 | „Sobald wir etwas starten, dann ist es etwas holprig.“ |
| 0:04 | 0:25.0–0:48.3 | Google Docs vs. Microsoft Office story (pauses + formatting aside removed) |
| 0:23 | 0:49.4–0:57.8 | „Irgendwann mal kommt der Punkt, da ist es viel, viel besser …“ |
| 0:31 | 0:58.4–1:14.4 | The old way still works better than the new one, and we have to push through |
| 0:44 | 1:15.2–1:21.4 | „Der Durchbruch, der kommt definitiv. Wer lang genug dabei bleibt, der schafft das.“ |
| 0:50 | 1:21.5–1:29.8 | End cards (name card uses the intro music, because "Das war's für heute" is spoken under it in the source) |

ASR corrections in captions: „rauskam“, „zum Vergessen“.

## Rebuild
```
pip install faster-whisper "opencv-python-headless<5"
python3 build_reel.py durchhalten.json SOURCE.mp4 transcript-durchhalten.json OUT_DIR
```
Each `*.json` config lists the cut pieces (source in/out, `crop`/`fit`, framing options), hook text and caption fixes.
`change.json` holds the Change reel's cuts; the speaker tracker was improved afterwards, so a rebuild may frame slightly differently from the delivered file.
