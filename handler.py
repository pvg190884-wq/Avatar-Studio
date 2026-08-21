import runpod
import base64
import subprocess
import os
import uuid
import shutil
import traceback

def handler(event):
    input_data = event.get("input", {})
    job_id = str(uuid.uuid4())
    work_dir = f"/tmp/{job_id}"
    os.makedirs(work_dir, exist_ok=True)

    try:
        if "image_base64" not in input_data or "audio_base64" not in input_data:
            return {"error": "нужны image_base64 и audio_base64"}

        image_path = f"{work_dir}/source.jpg"
        with open(image_path, "wb") as f:
            f.write(base64.b64decode(input_data["image_base64"]))

        audio_path = f"{work_dir}/driven.wav"
        with open(audio_path, "wb") as f:
            f.write(base64.b64decode(input_data["audio_base64"]))

        expression_scale = str(input_data.get("expression_scale", 0.7))
        pose_style = str(input_data.get("pose_style", 0))
        size = str(input_data.get("size", 512))
        still = input_data.get("still", True)
        enhancer = input_data.get("enhancer", "gfpgan")

        result_dir = f"{work_dir}/results"

        cmd = [
            "python", "/app/SadTalker/inference.py",
            "--driven_audio", audio_path,
            "--source_image", image_path,
            "--result_dir", result_dir,
            "--size", size,
            "--expression_scale", expression_scale,
            "--pose_style", pose_style,
        ]
        if still:
            cmd.append("--still")
        if enhancer:
            cmd += ["--enhancer", enhancer]

        proc = subprocess.run(
            cmd, cwd="/app/SadTalker",
            capture_output=True, text=True, timeout=1800
        )

        if proc.returncode != 0:
            return {"error": "sadtalker failed", "stderr": proc.stderr[-3000:]}

        video_path = None
        for root, _, files in os.walk(result_dir):
            for f in files:
                if f.endswith(".mp4"):
                    video_path = os.path.join(root, f)

        if not video_path:
            return {"error": "видео не сгенерировано", "stderr": proc.stderr[-3000:]}

        with open(video_path, "rb") as vf:
            video_base64 = base64.b64encode(vf.read()).decode("utf-8")

        return {"video_base64": video_base64}

    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()[-2000:]}

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

runpod.serverless.start({"handler": handler})
