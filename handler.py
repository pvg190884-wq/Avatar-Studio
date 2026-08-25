import runpod
import base64
import subprocess
import os
import re
import uuid
import shutil

# Ленивая загрузка XTTS-v2: модель НЕ грузится при старте контейнера/импорте
# модуля (иначе холодный старт воркера может не успеть пройти проверку
# готовности со стороны RunPod до того, как модель встанет на GPU).
# Модель загружается один раз, при первом реальном запросе, требующем TTS.
tts_model = None


def get_tts_model():
    global tts_model
    if tts_model is None:
        print("Загружаю модель XTTS-v2...")
        from TTS.api import TTS
        tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
        print("XTTS-v2 готова.")
    return tts_model


# Параметры генерации XTTS-v2, подобранные под более живую, менее
# "роботизированную" интонацию, чем дефолтные значения библиотеки.
# temperature выше дефолтного (0.65) даёт больше естественной вариативности
# высоты тона и темпа между фразами, не срываясь в нестабильность.
# repetition_penalty повышен, чтобы избежать характерного "проглатывания"
# слов на длинных фразах.
TTS_GENERATION_PARAMS = {
    "temperature": 0.75,
    "repetition_penalty": 5.0,
    "top_k": 50,
    "top_p": 0.85,
}

# Длительность пауз между предложениями (мс), в зависимости от знака
# препинания, которым предложение заканчивается — имитирует то, как
# человек делает более долгую паузу после точки/восклицания, чем после
# запятой внутри фразы.
PAUSE_MS_BY_PUNCTUATION = {
    ".": 450,
    "!": 450,
    "?": 500,
    "...": 600,
    ",": 200,
}


def split_into_sentences(text):
    """Разбивает текст на предложения, сохраняя знак препинания в конце
    каждого куска. Нужно для того, чтобы синтезировать TTS по фразам и
    вставлять между ними естественные паузы, а не одним ровным потоком."""
    # Разбиваем по границе предложения: точка/!/?/многоточие, за которыми
    # следует пробел и заглавная буква (или конец строки).
    parts = re.split(r'(?<=[.!?…])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def pause_after(sentence):
    """Определяет длительность паузы после предложения по завершающему
    знаку препинания."""
    stripped = sentence.rstrip()
    if stripped.endswith("..."):
        return PAUSE_MS_BY_PUNCTUATION["..."]
    if stripped.endswith("…"):
        return PAUSE_MS_BY_PUNCTUATION["..."]
    if stripped and stripped[-1] in PAUSE_MS_BY_PUNCTUATION:
        return PAUSE_MS_BY_PUNCTUATION[stripped[-1]]
    return PAUSE_MS_BY_PUNCTUATION["."]


def synthesize_speech(text, speaker_wav_path, language, work_dir, speed=1.0):
    """Синтезирует речь по предложениям с естественными паузами между
    ними вместо одного монотонного потока. Возвращает путь к итоговому
    .wav файлу."""
    from pydub import AudioSegment

    model = get_tts_model()
    sentences = split_into_sentences(text)
    if not sentences:
        sentences = [text]

    segments = []
    for i, sentence in enumerate(sentences):
        chunk_path = f"{work_dir}/chunk_{i}.wav"
        model.tts_to_file(
            text=sentence,
            speaker_wav=speaker_wav_path,
            language=language,
            file_path=chunk_path,
            speed=speed,
            **TTS_GENERATION_PARAMS,
        )
        segments.append(AudioSegment.from_wav(chunk_path))
        if i < len(sentences) - 1:
            segments.append(AudioSegment.silent(duration=pause_after(sentence)))

    combined = segments[0]
    for seg in segments[1:]:
        combined += seg

    final_path = f"{work_dir}/synthesized.wav"
    combined.export(final_path, format="wav")
    return final_path


# Пресеты обработки голоса — понижение/повышение тембра (через сдвиг
# высоты тона) плюс лёгкая EQ-подкраска (бас/треble), по той же схеме,
# что EMOTION_PRESETS ниже. Позже это станет выпадающим списком на сайте.
#   pitch_semitones — сдвиг высоты тона в полутонах (± ~1 октава = ±12)
#   bass_gain / treble_gain — усиление/ослабление низких/высоких частот в дБ
VOICE_STYLE_PRESETS = {
    "neutral": {"pitch_semitones": 0,  "bass_gain": 0,  "treble_gain": 0},
    "lower":   {"pitch_semitones": -2, "bass_gain": 3,  "treble_gain": -1},  # ниже и теплее
    "higher":  {"pitch_semitones": 2,  "bass_gain": -1, "treble_gain": 2},   # выше и чётче
    "warm":    {"pitch_semitones": -1, "bass_gain": 4,  "treble_gain": -2},  # тёплый, мягкий тембр
    "bright":  {"pitch_semitones": 1,  "bass_gain": -2, "treble_gain": 4},   # яркий, чёткий тембр
}

# Пресеты скорости речи — именованные варианты для выпадающего списка,
# как у emotion/voice_style. XTTS-v2 поддерживает continuous speed
# нативно (не через ffmpeg), поэтому применяется прямо при синтезе, а не
# постобработкой. Значение "custom" позволяет задать точное число через
# отдельный параметр speech_speed_value, если понадобится тонкая настройка
# сверх трёх стандартных вариантов.
SPEECH_TEMPO_PRESETS = {
    "slow":   0.85,
    "normal": 1.0,
    "fast":   1.15,
}


def apply_voice_style(audio_path, work_dir, style="neutral"):
    """Применяет сдвиг высоты тона и лёгкую EQ-обработку через ffmpeg.
    Сдвиг тона делается классическим трюком asetrate+atempo: меняем
    частоту дискретизации (что меняет и высоту, и скорость), а потом
    компенсируем скорость обратно через atempo — в итоге длительность
    аудио не меняется, меняется только высота тона."""
    preset = VOICE_STYLE_PRESETS.get(style, VOICE_STYLE_PRESETS["neutral"])
    pitch_semitones = preset["pitch_semitones"]
    bass_gain = preset["bass_gain"]
    treble_gain = preset["treble_gain"]

    if pitch_semitones == 0 and bass_gain == 0 and treble_gain == 0:
        return audio_path  # neutral — ничего не меняем, экономим шаг

    filters = []
    if pitch_semitones != 0:
        factor = 2 ** (pitch_semitones / 12)
        filters.append(f"asetrate=44100*{factor},aresample=44100,atempo={1/factor}")
    if bass_gain != 0:
        filters.append(f"bass=g={bass_gain}")
    if treble_gain != 0:
        filters.append(f"treble=g={treble_gain}")

    out_path = f"{work_dir}/voice_styled.wav"
    cmd = [
        "ffmpeg", "-y", "-i", audio_path,
        "-af", ",".join(filters),
        "-ar", "44100",
        out_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg (voice_style) упал: {proc.stderr[-2000:]}")
    return out_path


EMOTION_PRESETS = {
    "neutral": {"expression_scale": 0.7, "pose_style": 0, "still": True},
    "happy":   {"expression_scale": 1.1, "pose_style": 0, "still": False},
    "sad":     {"expression_scale": 0.6, "pose_style": 0, "still": True},
    "angry":   {"expression_scale": 1.0, "pose_style": 0, "still": False},
    "love":    {"expression_scale": 0.9, "pose_style": 0, "still": False},
}

# Коэффициент приглушения expression_scale, применяется только когда
# preprocess=full/extfull И still=False (то есть голове разрешено
# двигаться). Подобран как стартовая точка — при необходимости можно
# потюнить по факту тестов: меньше значение = меньше наклон/движение
# головы, но и менее выразительная эмоция.
FULL_PREPROCESS_DAMPING = 0.65


def run_sadtalker(image_path, audio_path, result_dir, expression_scale,
                   pose_style, size, still, enhancer, preprocess):
    cmd = [
        "python", "/app/SadTalker/inference.py",
        "--driven_audio", audio_path,
        "--source_image", image_path,
        "--result_dir", result_dir,
        "--size", str(size),
        "--expression_scale", str(expression_scale),
        "--pose_style", str(pose_style),
        "--preprocess", preprocess,
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
        raise RuntimeError(f"SadTalker упал: {proc.stderr[-3000:]}")

    for root, _, files in os.walk(result_dir):
        for f in files:
            if f.endswith(".mp4"):
                return os.path.join(root, f)
    raise RuntimeError("SadTalker не создал видео")


def handler(event):
    input_data = event.get("input", {}) or {}

    # Мягкий healthcheck: RunPod может дёргать handler пустым/тестовым
    # input на этапе проверки готовности воркера (Testing). Раньше такой
    # вызов падал с "нужен image_base64", что могло восприниматься как
    # сбой воркера. Теперь на явный healthcheck и на полностью пустой
    # input отвечаем "ok", не трогая GPU/модели.
    if input_data.get("healthcheck") is True or not input_data:
        return {"ok": True}

    job_id = str(uuid.uuid4())
    work_dir = f"/tmp/{job_id}"
    os.makedirs(work_dir, exist_ok=True)

    try:
        if "image_base64" not in input_data:
            return {"error": "нужен image_base64"}

        image_path = f"{work_dir}/source.jpg"
        with open(image_path, "wb") as f:
            f.write(base64.b64decode(input_data["image_base64"]))

        if "audio_base64" in input_data:
            audio_path = f"{work_dir}/driven.wav"
            with open(audio_path, "wb") as f:
                f.write(base64.b64decode(input_data["audio_base64"]))
            expression_scale = input_data.get("expression_scale", 0.7)
            pose_style = input_data.get("pose_style", 0)
            still = input_data.get("still", True)
            preprocess = input_data.get("preprocess", "full")
            # При preprocess=full/extfull сохраняем всё тело в кадре, но
            # зона обратной вклейки (paste-back) плохо переносит резкий
            # наклон/поворот головы — на границе шеи/плеч появляется шов.
            # Вместо того чтобы полностью гасить движение головы (still=
            # True), приглушаем его амплитуду через expression_scale —
            # так голова остаётся "живой", просто менее размашисто.
            if preprocess in ("full", "extfull") and not still:
                expression_scale *= FULL_PREPROCESS_DAMPING

        elif "text" in input_data and "voice_sample_base64" in input_data:
            voice_sample_path = f"{work_dir}/voice_sample.wav"
            with open(voice_sample_path, "wb") as f:
                f.write(base64.b64decode(input_data["voice_sample_base64"]))

            text = input_data["text"]
            language = input_data.get("language", "ru")
            emotion = input_data.get("emotion", "neutral")

            # Явная проверка вместо тихого фолбэка на neutral —
            # если пришло неизвестное имя эмоции, лучше сразу вернуть
            # ошибку, чем незаметно сгенерировать не то, что просили.
            if emotion not in EMOTION_PRESETS:
                return {
                    "error": (
                        f"неизвестная эмоция '{emotion}', "
                        f"доступны: {list(EMOTION_PRESETS.keys())}"
                    )
                }
            preset = EMOTION_PRESETS[emotion]

            voice_style = input_data.get("voice_style", "neutral")
            if voice_style not in VOICE_STYLE_PRESETS:
                return {
                    "error": (
                        f"неизвестный voice_style '{voice_style}', "
                        f"доступны: {list(VOICE_STYLE_PRESETS.keys())}"
                    )
                }

            speech_tempo = input_data.get("speech_tempo", "normal")
            if speech_tempo == "custom":
                speech_speed = input_data.get("speech_speed_value")
                if speech_speed is None:
                    return {
                        "error": (
                            "speech_tempo='custom' требует числового "
                            "speech_speed_value (например 1.05)"
                        )
                    }
            elif speech_tempo in SPEECH_TEMPO_PRESETS:
                speech_speed = SPEECH_TEMPO_PRESETS[speech_tempo]
            else:
                return {
                    "error": (
                        f"неизвестный speech_tempo '{speech_tempo}', "
                        f"доступны: {list(SPEECH_TEMPO_PRESETS.keys()) + ['custom']}"
                    )
                }

            audio_path = synthesize_speech(
                text=text,
                speaker_wav_path=voice_sample_path,
                language=language,
                work_dir=work_dir,
                speed=speech_speed,
            )
            audio_path = apply_voice_style(audio_path, work_dir, voice_style)

            expression_scale = preset["expression_scale"]
            pose_style = preset["pose_style"]
            still = preset["still"]
            preprocess = input_data.get("preprocess", "full")
            # См. комментарий в ветке audio_base64 выше — та же причина
            # и то же решение: приглушаем амплитуду, а не гасим движение.
            if preprocess in ("full", "extfull") and not still:
                expression_scale *= FULL_PREPROCESS_DAMPING

        else:
            return {"error": "нужны либо audio_base64, либо text + voice_sample_base64"}

        size = input_data.get("size", 512)
        enhancer = input_data.get("enhancer", "gfpgan")

        result_dir = f"{work_dir}/results"
        video_path = run_sadtalker(
            image_path, audio_path, result_dir,
            expression_scale, pose_style, size, still, enhancer, preprocess
        )

        with open(video_path, "rb") as vf:
            video_base64 = base64.b64encode(vf.read()).decode("utf-8")

        return {"video_base64": video_base64}

    except Exception as e:
        import traceback
        return {"error": str(e), "trace": traceback.format_exc()[-2000:]}

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


runpod.serverless.start({"handler": handler})
 
 
