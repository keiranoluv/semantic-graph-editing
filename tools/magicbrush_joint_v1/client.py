"""Local HTTP transport only; never loads weights or contacts an OpenAI service."""

import base64
import json
import mimetypes

import requests

from tools.magicbrush_joint_v1.prepare import PACKAGE
from tools.magicbrush_joint_v1.schema import SCHEMA
from tools.prepare_magicbrush import ROOT


def payload(cfg, row):
    content = [{"type": "text", "text": "SOURCE image (first):"}]
    for role in ("source_image", "target_image"):
        if role == "target_image":
            content.append({"type": "text", "text": "EDITED TARGET image (second):"})
        path = ROOT / row[role]["path"]
        mime = mimetypes.guess_type(path.name)[0]
        if mime not in ("image/png", "image/jpeg", "image/webp"):
            raise ValueError("unsupported_image_mime")
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}})
    metadata = {k: row[k] for k in ("caption", "metadata") if k in row}
    content.append(
        {"type": "text", "text": json.dumps({"instruction": row["instruction"], "optional_metadata": metadata})}
    )
    return {
        "model": cfg["server"]["served_model_name"],
        "messages": [
            {"role": "system", "content": (PACKAGE / "prompt_v1.txt").read_text()},
            {"role": "user", "content": content},
        ],
        "seed": cfg["seed"],
        **cfg["generation"],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "joint_annotation", "strict": True, "schema": SCHEMA},
        },
    }


class LocalClient:
    def __init__(self, cfg):
        self.cfg = cfg
        self.session = requests.Session()
        self.session.trust_env = False  # Never route paired images through an ambient proxy.

    def check_model(self):
        response = self.session.get(self.cfg["server"]["base_url"] + "/models", timeout=(10, 30), allow_redirects=False)
        if response.status_code != 200 or self.cfg["server"]["served_model_name"] not in {
            m["id"] for m in response.json()["data"]
        }:
            raise ValueError("local_server_model_identity_mismatch")

    def annotate(self, row):
        body = bytearray()
        status = None
        try:
            with self.session.post(
                self.cfg["server"]["base_url"] + "/chat/completions",
                json=payload(self.cfg, row),
                timeout=(10, self.cfg["http"]["timeout_seconds"]),
                allow_redirects=False,
                stream=True,
            ) as response:
                status = response.status_code
                for block in response.iter_content(16384):
                    body.extend(block)
                    if len(body) > self.cfg["http"]["max_response_bytes"]:
                        return {
                            "http_status": status,
                            "body": body.decode("utf-8", errors="replace"),
                            "error": "response_size_limit",
                        }
            return {
                "http_status": status,
                "body": body.decode("utf-8"),
                "error": None if status == 200 else f"http_{status}",
            }
        except (requests.RequestException, UnicodeError) as error:
            return {
                "http_status": status,
                "body": body.decode("utf-8", errors="replace"),
                "error": type(error).__name__ + ":" + str(error),
            }
