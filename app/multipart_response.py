# SPDX-FileCopyrightText: 2026 Buse Nur Sabah
# SPDX-License-Identifier: GPL-3.0-only



from fastapi.responses import StreamingResponse
import json
import os


def multipart_response(json_data, pdf_path):
    boundary = "BPA_BOUNDARY"

    def generate():
        # ============
        # JSON Part
        # ============
        yield f"--{boundary}\r\n".encode()
        yield b"Content-Type: application/json\r\n\r\n"
        yield json.dumps(json_data).encode()
        yield b"\r\n"

        # ===========
        # PDF Part
        # ===========
        yield f"--{boundary}\r\n".encode()
        yield b"Content-Type: application/pdf\r\n"
        yield b"Content-Disposition: attachment; filename=report.pdf\r\n\r\n"

        with open(pdf_path, "rb") as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                yield chunk

        yield b"\r\n"
        yield f"--{boundary}--\r\n".encode()

    return StreamingResponse(
        generate(),
        media_type=f"multipart/mixed; boundary={boundary}"
    )
