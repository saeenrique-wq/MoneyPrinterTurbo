import json
import os.path
import re
from timeit import default_timer as timer

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None
from loguru import logger

from app.config import config
from app.utils import utils

model_size = config.whisper.get("model_size", "large-v3")
device = config.whisper.get("device", "cpu")
compute_type = config.whisper.get("compute_type", "int8")
model = None


def create(audio_file, subtitle_file: str = ""):
    global model
    if WhisperModel is None:
        logger.warning("faster_whisper not available, skipping whisper subtitle generation")
        return ""
    if not model:
        model_path = f"{utils.root_dir()}/models/whisper-{model_size}"
        model_bin_file = f"{model_path}/model.bin"
        if not os.path.isdir(model_path) or not os.path.isfile(model_bin_file):
            model_path = model_size

        logger.info(
            f"loading model: {model_path}, device: {device}, compute_type: {compute_type}"
        )
        try:
            model = WhisperModel(
                model_size_or_path=model_path, device=device, compute_type=compute_type
            )
        except Exception as e:
            logger.error(
                f"failed to load model: {e} \n\n"
                f"********************************************\n"
                f"this may be caused by network issue. \n"
                f"please download the model manually and put it in the 'models' folder. \n"
                f"see [README.md FAQ](https://github.com/harry0703/MoneyPrinterTurbo) for more details.\n"
                f"********************************************\n\n"
            )
            return None

    logger.info(f"start, output file: {subtitle_file}")
    if not subtitle_file:
        subtitle_file = f"{audio_file}.srt"

    segments, info = model.transcribe(
        audio_file,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    logger.info(
        f"detected language: '{info.language}', probability: {info.language_probability:.2f}"
    )

    start = timer()
    subtitles = []
    # Per-line list of {text, start, end} word dicts, aligned by index with
    # `subtitles`. Used to render word-by-word animated captions; kept
    # separate from the .srt file since SRT has no per-word timing concept.
    line_words: list[list[dict]] = []
    pending_words: list[dict] = []

    def recognized(seg_text, seg_start, seg_end):
        seg_text = seg_text.strip()
        if not seg_text:
            pending_words.clear()
            return

        msg = "[%.2fs -> %.2fs] %s" % (seg_start, seg_end, seg_text)
        logger.debug(msg)

        subtitles.append(
            {"msg": seg_text, "start_time": seg_start, "end_time": seg_end}
        )
        line_words.append(pending_words.copy())
        pending_words.clear()

    for segment in segments:
        words_idx = 0
        words_len = len(segment.words)

        seg_start = 0
        seg_end = 0
        seg_text = ""

        if segment.words:
            is_segmented = False
            for word in segment.words:
                if not is_segmented:
                    seg_start = word.start
                    is_segmented = True

                seg_end = word.end
                # If it contains punctuation, then break the sentence.
                seg_text += word.word
                clean_word = word.word.strip()
                if clean_word:
                    pending_words.append(
                        {"text": clean_word, "start": word.start, "end": word.end}
                    )

                if utils.str_contains_punctuation(word.word):
                    # remove last char
                    seg_text = seg_text[:-1]
                    if not seg_text:
                        continue

                    recognized(seg_text, seg_start, seg_end)

                    is_segmented = False
                    seg_text = ""

                if words_idx == 0 and segment.start < word.start:
                    seg_start = word.start
                if words_idx == (words_len - 1) and segment.end > word.end:
                    seg_end = word.end
                words_idx += 1

        if not seg_text:
            continue

        recognized(seg_text, seg_start, seg_end)

    end = timer()

    diff = end - start
    logger.info(f"complete, elapsed: {diff:.2f} s")

    idx = 1
    lines = []
    for subtitle in subtitles:
        text = subtitle.get("msg")
        if text:
            lines.append(
                utils.text_to_srt(
                    idx, text, subtitle.get("start_time"), subtitle.get("end_time")
                )
            )
            idx += 1

    sub = "\n".join(lines) + "\n"
    with open(subtitle_file, "w", encoding="utf-8") as f:
        f.write(sub)
    logger.info(f"subtitle file created: {subtitle_file}")

    write_word_timing_sidecar(subtitle_file, subtitles, line_words)


def word_timing_sidecar_path(subtitle_file: str) -> str:
    return f"{subtitle_file}.words.json"


def write_word_timing_sidecar(
    subtitle_file: str, subtitles: list[dict], line_words: list[list[dict]]
) -> None:
    """Persist per-word timestamps next to the .srt file.

    SRT has no concept of per-word timing, so word-by-word animated
    captions need this sidecar. Only whisper (subtitle_provider="whisper")
    produces word-level timestamps; other providers never call this.
    """
    try:
        payload = []
        for subtitle, words in zip(subtitles, line_words):
            payload.append(
                {
                    "start": subtitle.get("start_time"),
                    "end": subtitle.get("end_time"),
                    "text": subtitle.get("msg"),
                    "words": words,
                }
            )
        sidecar_path = word_timing_sidecar_path(subtitle_file)
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        logger.info(f"word timing sidecar created: {sidecar_path}")
    except Exception as exc:
        logger.warning(f"failed to write word timing sidecar: {exc}")


def file_to_subtitles(filename):
    if not filename or not os.path.isfile(filename):
        return []

    times_texts = []
    current_times = None
    current_text = ""
    index = 0
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            times = re.findall("([0-9]*:[0-9]*:[0-9]*,[0-9]*)", line)
            if times:
                current_times = line
            elif line.strip() == "" and current_times:
                index += 1
                times_texts.append((index, current_times.strip(), current_text.strip()))
                current_times, current_text = None, ""
            elif current_times:
                current_text += line

    # Flush the final block. SRT files whose last subtitle is not followed by a
    # trailing blank line never hit the blank-line branch above, so without this
    # the last subtitle would be silently dropped.
    if current_times:
        index += 1
        times_texts.append((index, current_times.strip(), current_text.strip()))
    return times_texts


def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def similarity(a, b):
    distance = levenshtein_distance(a.lower(), b.lower())
    max_length = max(len(a), len(b))
    return 1 - (distance / max_length)


def _srt_timestamp_to_seconds(timestamp: str) -> float:
    """Parse an SRT timestamp ("HH:MM:SS,mmm") into seconds."""
    hms, _, millis = timestamp.strip().partition(",")
    hours, minutes, seconds = (int(part) for part in hms.split(":"))
    return hours * 3600 + minutes * 60 + seconds + int(millis or 0) / 1000


def _load_word_sidecar(subtitle_file: str) -> list:
    sidecar_path = word_timing_sidecar_path(subtitle_file)
    if not os.path.isfile(sidecar_path):
        return []
    try:
        with open(sidecar_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning(f"failed to load word timing sidecar for correction: {exc}")
        return []


def correct(subtitle_file, video_script):
    subtitle_items = file_to_subtitles(subtitle_file)
    normalized_script = utils.normalize_script_for_subtitle_matching(video_script)
    script_lines = utils.split_string_by_punctuations(normalized_script)

    # Word-level timing (whisper only) is index-aligned with `subtitle_items`
    # at this point. As lines below get merged/split to match the script,
    # `source_ranges` records which original indices fed each output line so
    # the sidecar can be rebuilt instead of thrown away — throwing it away on
    # every trivial correction (a re-cased word, a dropped accent) would
    # disable animated captions on almost every real generation, since
    # whisper transcripts essentially never match the script byte-for-byte.
    original_words = [item.get("words", []) for item in _load_word_sidecar(subtitle_file)]

    corrected = False
    new_subtitle_items = []
    source_ranges: list[tuple[int, int] | None] = []
    script_index = 0
    subtitle_index = 0

    while script_index < len(script_lines) and subtitle_index < len(subtitle_items):
        script_line = script_lines[script_index].strip()
        subtitle_line = subtitle_items[subtitle_index][2].strip()

        if script_line == subtitle_line:
            new_subtitle_items.append(subtitle_items[subtitle_index])
            source_ranges.append((subtitle_index, subtitle_index + 1))
            script_index += 1
            subtitle_index += 1
        else:
            combined_subtitle = subtitle_line
            start_time = subtitle_items[subtitle_index][1].split(" --> ")[0]
            end_time = subtitle_items[subtitle_index][1].split(" --> ")[1]
            next_subtitle_index = subtitle_index + 1

            while next_subtitle_index < len(subtitle_items):
                next_subtitle = subtitle_items[next_subtitle_index][2].strip()
                if similarity(
                    script_line, combined_subtitle + " " + next_subtitle
                ) > similarity(script_line, combined_subtitle):
                    combined_subtitle += " " + next_subtitle
                    end_time = subtitle_items[next_subtitle_index][1].split(" --> ")[1]
                    next_subtitle_index += 1
                else:
                    break

            if similarity(script_line, combined_subtitle) > 0.8:
                logger.warning(
                    f"Merged/Corrected - Script: {script_line}, Subtitle: {combined_subtitle}"
                )
                new_subtitle_items.append(
                    (
                        len(new_subtitle_items) + 1,
                        f"{start_time} --> {end_time}",
                        script_line,
                    )
                )
                source_ranges.append((subtitle_index, next_subtitle_index))
                corrected = True
            else:
                logger.warning(
                    f"Mismatch - Script: {script_line}, Subtitle: {combined_subtitle}"
                )
                new_subtitle_items.append(
                    (
                        len(new_subtitle_items) + 1,
                        f"{start_time} --> {end_time}",
                        script_line,
                    )
                )
                source_ranges.append((subtitle_index, next_subtitle_index))
                corrected = True

            script_index += 1
            subtitle_index = next_subtitle_index

    # Process the remaining lines of the script.
    while script_index < len(script_lines):
        logger.warning(f"Extra script line: {script_lines[script_index]}")
        if subtitle_index < len(subtitle_items):
            new_subtitle_items.append(
                (
                    len(new_subtitle_items) + 1,
                    subtitle_items[subtitle_index][1],
                    script_lines[script_index],
                )
            )
            source_ranges.append((subtitle_index, subtitle_index + 1))
            subtitle_index += 1
        else:
            new_subtitle_items.append(
                (
                    len(new_subtitle_items) + 1,
                    "00:00:00,000 --> 00:00:00,000",
                    script_lines[script_index],
                )
            )
            # No original line backs this entry (script ran longer than the
            # transcript), so there's no word timing to carry forward.
            source_ranges.append(None)
        script_index += 1
        corrected = True

    if corrected:
        with open(subtitle_file, "w", encoding="utf-8") as fd:
            for i, item in enumerate(new_subtitle_items):
                fd.write(f"{i + 1}\n{item[1]}\n{item[2]}\n\n")
        logger.info("Subtitle corrected")

        sidecar_path = word_timing_sidecar_path(subtitle_file)
        if original_words:
            merged_payload = []
            for item, source_range in zip(new_subtitle_items, source_ranges):
                words = []
                if source_range:
                    start, end = source_range
                    for idx in range(start, min(end, len(original_words))):
                        words.extend(original_words[idx])
                times = item[1].split(" --> ")
                merged_payload.append(
                    {
                        "start": _srt_timestamp_to_seconds(times[0]),
                        "end": _srt_timestamp_to_seconds(times[1]),
                        "text": item[2],
                        "words": words,
                    }
                )
            try:
                with open(sidecar_path, "w", encoding="utf-8") as f:
                    json.dump(merged_payload, f, ensure_ascii=False)
                logger.info(f"word timing sidecar re-aligned after correction: {sidecar_path}")
            except Exception as exc:
                logger.warning(f"failed to rewrite word timing sidecar: {exc}")
    else:
        logger.success("Subtitle is correct")


if __name__ == "__main__":
    task_id = "c12fd1e6-4b0a-4d65-a075-c87abe35a072"
    task_dir = utils.task_dir(task_id)
    subtitle_file = f"{task_dir}/subtitle.srt"
    audio_file = f"{task_dir}/audio.mp3"

    subtitles = file_to_subtitles(subtitle_file)
    print(subtitles)

    script_file = f"{task_dir}/script.json"
    with open(script_file, "r") as f:
        script_content = f.read()
    s = json.loads(script_content)
    script = s.get("script")

    correct(subtitle_file, script)

    subtitle_file = f"{task_dir}/subtitle-test.srt"
    create(audio_file, subtitle_file)
