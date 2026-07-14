from dataclasses import dataclass
import re
from difflib import SequenceMatcher

@dataclass
class Match:
    start: int
    end: int
    score: float


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # remove trailing whitespace
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)

    return text


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def token_similarity(a: str, b: str):
    WORD = re.compile(r"\w+")
    sa = set(WORD.findall(a))
    sb = set(WORD.findall(b))

    if not sa and not sb:
        return 1.0

    return len(sa & sb) / len(sa | sb)


def score(a: str, b: str):
    return (
        similarity(a, b) * 0.7 +
        token_similarity(a, b) * 0.3
    )


def line_offsets(text: str):
    offsets = [0]
    pos = 0

    for line in text.splitlines(True):
        pos += len(line)
        offsets.append(pos)

    return offsets


def candidate_windows(text: str, target_lines: int):

    lines = text.splitlines(True)
    offsets = line_offsets(text)

    windows = []

    for size in range(
        max(1, target_lines - 3),
        target_lines + 4,
    ):

        for start in range(len(lines) - size + 1):
            end = start + size
            chunk = "".join(lines[start:end])
            windows.append(
                (
                    offsets[start],
                    offsets[end],
                    chunk,
                )
            )

    return windows


def find_best_match(file_text: str, target: str):

    file_text = normalize(file_text)
    target = normalize(target)

    idx = file_text.find(target)
    if idx != -1:
        return Match(idx, idx + len(target), 1.0)

    target_lines = len(target.splitlines())

    best = None

    for start, end, chunk in candidate_windows(
        file_text,
        target_lines,
    ):

        s = score(chunk, target)

        if best is None or s > best.score:
            best = Match(start, end, s)

    return best


class ReplaceError(Exception):
    pass


def replace(file_text: str, old: str, new: str):

    match = find_best_match(file_text, old)

    if match is None:
        raise ReplaceError("No candidate")

    if match.score < 0.92:
        raise ReplaceError(
            f"Low confidence ({match.score:.2f})"
        )

    return (
        file_text[:match.start]
        + new
        + file_text[match.end:]
    )


def choose_match(matches):

    matches.sort(
        key=lambda m: m.score,
        reverse=True,
    )

    best = matches[0]

    if len(matches) == 1:
        return best

    second = matches[1]

    if best.score - second.score < 0.03:
        raise ReplaceError(
            "Ambiguous match"
        )

    return best



def insert_before(file_text: str, anchor: str, new: str):
    match = find_best_match(file_text, anchor)

    if match is None:
        raise ReplaceError("No candidate")

    if match.score < 0.92:
        raise ReplaceError(
            f"Low confidence ({match.score:.2f})"
        )

    return (
        file_text[:match.start]
        + new
        + file_text[match.start:]
    )


def insert_after(file_text: str, anchor: str, new: str):
    match = find_best_match(file_text, anchor)

    if match is None:
        raise ReplaceError("No candidate")

    if match.score < 0.92:
        raise ReplaceError(
            f"Low confidence ({match.score:.2f})"
        )

    return (
        file_text[:match.end]
        + new
        + file_text[match.end:]
    )


def delete(file_text: str, old: str):
    return replace(file_text, old, "")


def apply_mutation(file_text: str, mutation):
    if mutation.type == "replace":
        if mutation.old is None:
            raise ValueError("replace requires old")
        return replace(
            file_text,
            mutation.old,
            mutation.new,
        )

    if mutation.type == "insert_before":
        if mutation.anchor is None:
            raise ValueError("insert_before requires anchor")
        return insert_before(
            file_text,
            mutation.anchor,
            mutation.new,
        )

    if mutation.type == "insert_after":
        if mutation.anchor is None:
            raise ValueError("insert_after requires anchor")
        return insert_after(
            file_text,
            mutation.anchor,
            mutation.new,
        )

    if mutation.type == "delete":
        if mutation.old is None:
            raise ValueError("delete requires old")
        return delete(
            file_text,
            mutation.old,
        )

    raise ValueError(f"Unknown mutation type: {mutation.type}")


def apply_file_edit(file_text: str, file_edit):
    for op in file_edit.operations:
        file_text = apply_mutation(
            file_text,
            op,
        )
    return file_text