# Local Study setup notes

## What the Study workspace processes

- Lecture audio files are transcribed on-device with Faster-Whisper. The transcript retains timestamps so the agent can cite lecture time ranges.
- PowerPoint `.pptx` files are extracted slide by slide. Native text, tables, chart titles/data, and speaker notes are included. Text inside embedded slide images is read with local Tesseract OCR.
- PDFs are extracted page by page. Image-only/scanned pages are rendered locally and passed to Tesseract OCR when the page has no meaningful text layer.
- Word `.docx`, plain-text, and Markdown sources are extracted locally.
- Extracted passages are chunked, embedded with the configured local FastEmbed model, and stored in the user's local Study library.

No course file, transcript, or extracted text is sent to a cloud transcription or OCR service by this implementation.

## Windows OCR prerequisite

The Python dependency `pytesseract` is only a bridge to the OCR engine. It does not include the Tesseract executable.

1. Install the Windows build of Tesseract OCR on the machine running Odysseus.
2. Ensure the Tesseract executable is on the Windows `PATH`, or set the environment variable `ODYSSEUS_TESSERACT_CMD` to the full path of `tesseract.exe`.
3. Restart Odysseus so the process receives the updated environment.
4. If a different OCR language is required, install that Tesseract language data and set `ODYSSEUS_STUDY_OCR_LANG` (default: `eng`). PDF rendering defaults to 220 DPI; set `ODYSSEUS_STUDY_OCR_DPI` between 150 and 400 to trade speed for recognition quality.

If Tesseract is unavailable, text-native documents can still be imported. Scanned PDF pages without a text layer cannot be imported until OCR is available; PowerPoint source metadata reports when embedded-image OCR was skipped.

## Local model downloads

Faster-Whisper and FastEmbed may download their model files the first time they are used. These downloads are model assets only; course documents and transcripts are not uploaded. After download, inference and indexing run locally. For a fully offline first run, pre-populate the relevant local model caches before disconnecting from the network.

## Notes PDF formatting

The PDF renderer selects Arial on Windows or DejaVu Sans on Linux when available, and recognises the semantic HTML marker `<span class="supplementary">…</span>` and renders that passage in red. Use it for clearly labelled external research or model-supplied context, with a real source hyperlink/citation in the same passage. Ordinary course-source notes remain in the normal text colour.

## Current scope

Supported audio formats: WAV, MP3, M4A, MP4, WebM, OGG, FLAC and AAC.

Supported course documents: PDF, PPTX, DOCX, TXT and Markdown.

Legacy binary PowerPoint/Word formats (`.ppt`, `.doc`) and standalone image uploads are not supported. Browser-based live system-audio capture is available through the screen-sharing picker, but it is browser-dependent and should be treated as experimental; importing a saved lecture recording is the reliable path.
