"""Past Exams — uploaded exam papers used as style exemplars for Exam Prep.

Storage is deliberately OUTSIDE the course KB. Indexing exams like lecture
files would let questions on topics the course files don't cover (the
k-d tree question on an exam whose slides only teach heapify) leak into
chat retrieval, notes, the knowledge graph and Exam Prep topic planning.

  artifacts/courses/<id>/past_exams/
    index.json                {"version", "exams": [{exam_id, filename, status, stage, ...}]}
    <exam_id>/<filename>      the uploaded original
    <exam_id>/questions.json  segmented questions + LLM topic tags
    overrides.json            the student's tag corrections, keyed by question id;
                              re-applied on every read so a re-tag never clobbers them

Pipeline per exam (background task): extract text (ingest.extractors) →
segment (LLM: one entry per gradable question) → tag (LLM: which Exam Prep
topic each question primarily tests, or off_topic). Topic ids are the
Exam Prep bank's ids; if the bank has no topics yet they are planned first.

Generation hands the on-topic questions for a topic to
`ExamPrepSkill._generate_questions(exemplars=...)`, which writes new
questions into the shared exam bank with origin=past_exam_style, so the
existing quiz / grading / mastery / variant loop applies unchanged.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import uuid
from pathlib import Path
from typing import Any

from nano_notebooklm import config
from nano_notebooklm.ai import prompt_templates as prompts
from nano_notebooklm.skills import exam_prep as ep
from nano_notebooklm.skills.base import Skill
from nano_notebooklm.types import SkillResult

logger = logging.getLogger(__name__)

PAST_EXAMS_VERSION = 1
SEGMENT_WINDOW_CHARS = 12000
TAG_BATCH_SIZE = 15
MAX_QUESTIONS_PER_EXAM = 200
MAX_EXEMPLARS_PER_TOPIC = 5
# LLM tags below this confidence are not used as exemplars (a student
# override always counts) and are highlighted in the UI for review.
EXEMPLAR_MIN_CONFIDENCE = 0.5
GENERATE_KINDS = ("short_answer", "calculation", "multiple_choice")

MATCHES = ("primary", "partial", "off_topic", "untagged")
QUESTION_KINDS = ("multiple_choice", "short_answer", "calculation", "trace", "proof", "code", "other")

# (course_id, exam_id) pairs with a live processing task in THIS process.
# An index entry stuck at status=processing without one was interrupted by
# a server restart and is reported as failed.
_ACTIVE_JOBS: set[tuple[str, str]] = set()
_BACKGROUND_TASKS: set[asyncio.Task] = set()


def _index_lock(course_id: str) -> asyncio.Lock:
    return ep._lock_for(f"past_exams::{course_id}")


class LLMStepFailed(Exception):
    """Every LLM call of a step failed, so an empty result says nothing
    about the exam. Carries an error token for the UI."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


llm_error_code = ep.llm_error_code


def _worst(codes: list[str]) -> str:
    for code in ("llm_rate_limited", "llm_failed", "llm_bad_output"):
        if code in codes:
            return code
    return "llm_failed"


# ── persistence ───────────────────────────────────────────────────────


def _root(course_id: str) -> Path:
    return config.ARTIFACTS_DIR / "courses" / course_id / "past_exams"


def exam_dir(course_id: str, exam_id: str) -> Path:
    return _root(course_id) / exam_id


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("past_exams: unreadable %s: %s", path.name, type(e).__name__)
        return default


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def new_exam_id() -> str:
    return uuid.uuid4().hex[:12]


def _new_question_id() -> str:
    return f"pq_{uuid.uuid4().hex[:12]}"


def load_index(course_id: str) -> dict:
    data = _read_json(_root(course_id) / "index.json", None)
    if not isinstance(data, dict) or not isinstance(data.get("exams"), list):
        data = {"version": PAST_EXAMS_VERSION, "exams": []}
    for e in data["exams"]:
        if e.get("status") == "processing" and (course_id, e.get("exam_id")) not in _ACTIVE_JOBS:
            e["status"] = "failed"
            e["error"] = "interrupted"
    return data


def _save_index(course_id: str, index: dict) -> None:
    _write_json(_root(course_id) / "index.json", index)


def load_questions(course_id: str, exam_id: str) -> list[dict]:
    data = _read_json(exam_dir(course_id, exam_id) / "questions.json", {})
    qs = data.get("questions") if isinstance(data, dict) else None
    return qs if isinstance(qs, list) else []


def _save_questions(course_id: str, exam_id: str, questions: list[dict]) -> None:
    _write_json(
        exam_dir(course_id, exam_id) / "questions.json",
        {"exam_id": exam_id, "questions": questions},
    )


def load_overrides(course_id: str) -> dict:
    data = _read_json(_root(course_id) / "overrides.json", {})
    return data if isinstance(data, dict) else {}


def _bank_topics(course_id: str) -> list[dict]:
    return [t for t in ep.load_bank(course_id).get("topics", []) if not t.get("archived_topic")]


def _effective(q: dict, overrides: dict, valid_topic_ids: set[str]) -> dict:
    """Question with the student's override applied and topic ids that no
    longer exist in the bank (after a re-extract) dropped."""
    out = dict(q)
    out["overridden"] = False
    ov = overrides.get(q.get("id"))
    if isinstance(ov, dict):
        ov_ids = [t for t in ov.get("topic_ids") or [] if t in valid_topic_ids]
        if ov.get("match") == "off_topic" or ov_ids:
            out["topic_ids"] = ov_ids
            out["match"] = ov.get("match")
            out["confidence"] = 1.0
            out["overridden"] = True
            return out
    ids = [t for t in q.get("topic_ids") or [] if t in valid_topic_ids]
    out["stale_tags"] = len(ids) < len(q.get("topic_ids") or [])
    out["topic_ids"] = ids
    if out.get("match") in ("primary", "partial") and not ids:
        out["match"] = "untagged"
    return out


def _ready_exams(course_id: str) -> list[dict]:
    return [e for e in load_index(course_id)["exams"] if e.get("status") == "ready"]


def _exemplar(q: dict, exam: dict) -> dict:
    return {
        "id": q.get("id"),
        "ref": f"{exam.get('label') or exam.get('filename')} · Q{q.get('number') or '?'}",
        "text": q.get("text") or "",
        "solution": q.get("solution"),
        "focus": q.get("focus") if q.get("match") == "partial" else None,
    }


def collect_exemplars(course_id: str, topic_id: str, limit: int = MAX_EXEMPLARS_PER_TOPIC) -> list[dict]:
    """Past-exam questions tagged with `topic_id`, best first: primary over
    partial, questions with an official solution first, then confidence."""
    valid = {t["id"] for t in _bank_topics(course_id)}
    overrides = load_overrides(course_id)
    scored: list[tuple[tuple, dict]] = []
    for exam in _ready_exams(course_id):
        for raw in load_questions(course_id, exam["exam_id"]):
            q = _effective(raw, overrides, valid)
            if topic_id not in q["topic_ids"] or q.get("match") not in ("primary", "partial"):
                continue
            conf = float(q.get("confidence") or 0.0)
            if conf < EXEMPLAR_MIN_CONFIDENCE:
                continue
            key = (q["match"] == "primary", bool(q.get("solution")), conf)
            scored.append((key, _exemplar(q, exam)))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in scored[:limit]]


def load_exemplars_by_ids(course_id: str, ids: list[str]) -> list[dict]:
    wanted = set(ids)
    if not wanted:
        return []
    valid = {t["id"] for t in _bank_topics(course_id)}
    overrides = load_overrides(course_id)
    out: list[dict] = []
    for exam in _ready_exams(course_id):
        for raw in load_questions(course_id, exam["exam_id"]):
            if raw.get("id") in wanted:
                out.append(_exemplar(_effective(raw, overrides, valid), exam))
    return out


def _coerce_int(v: Any) -> int | None:
    try:
        return int(v) if v is not None and str(v).strip() != "" else None
    except (TypeError, ValueError):
        return None


def _coerce_points(v: Any) -> float | None:
    try:
        return float(v) if v is not None and str(v).strip() != "" else None
    except (TypeError, ValueError):
        return None


def _windows(pages: list[tuple[int, str]]) -> list[str]:
    """Group page texts into prompt-sized windows on page boundaries."""
    windows: list[str] = []
    buf: list[str] = []
    size = 0
    for page_no, text in pages:
        block = f"[page {page_no}]\n{text[:SEGMENT_WINDOW_CHARS]}"
        if buf and size + len(block) > SEGMENT_WINDOW_CHARS:
            windows.append("\n\n".join(buf))
            buf, size = [], 0
        buf.append(block)
        size += len(block)
    if buf:
        windows.append("\n\n".join(buf))
    return windows


# ── Skill ─────────────────────────────────────────────────────────────


class PastExamsSkill(Skill):
    """Actions: register | process | view | set_tags | retag | delete | generate."""

    name = "past_exams"
    description = "Ingest past exams, tag their questions by course topic, and generate questions in their style"

    async def execute(self, params: dict) -> SkillResult:
        action = params.get("action", "")
        course_id = params.get("course_id", "")
        if not course_id:
            return SkillResult(success=False, error="course_id required")
        user_lang = params.get("user_lang")
        try:
            if action == "register":
                return await self.register(course_id, params["exam_id"], params["filename"], params.get("engine", "pymupdf"))
            if action == "process":
                return await self.process(course_id, params["exam_id"], params.get("engine", "pymupdf"), params.get("lang", "en"), user_lang)
            if action == "view":
                return SkillResult(success=True, data=self.view(course_id))
            if action == "set_tags":
                return await self.set_tags(course_id, params["question_id"], params.get("topic_ids") or [], params.get("match", "primary"))
            if action == "retag":
                return await self.retag(course_id, user_lang)
            if action == "delete":
                return await self.delete(course_id, params["exam_id"])
            if action == "generate":
                return await self.generate(course_id, params.get("topic_ids") or [], int(params.get("count", 3)), user_lang)
            return SkillResult(success=False, error=f"unknown action: {action}")
        except ep.BankVersionTooNewError:
            return SkillResult(success=False, error="bank_version_too_new")
        except Exception as exc:
            logger.warning("past_exams action %s failed: %s", action, type(exc).__name__)
            return SkillResult(success=False, error="past_exams_failed")

    # ── upload lifecycle ───────────────────────────────────────────

    async def register(self, course_id: str, exam_id: str, filename: str, engine: str) -> SkillResult:
        async with _index_lock(course_id):
            index = load_index(course_id)
            index["exams"].append({
                "exam_id": exam_id,
                "filename": filename,
                "label": Path(filename).stem[:80],
                "uploaded_at": ep._now_iso(),
                "engine": engine,
                "status": "processing",
                "stage": "queued",
                "error": None,
                "question_count": 0,
            })
            _save_index(course_id, index)
        _ACTIVE_JOBS.add((course_id, exam_id))
        return SkillResult(success=True, data={"exam_id": exam_id})

    async def _update_entry(self, course_id: str, exam_id: str, **fields: Any) -> None:
        async with _index_lock(course_id):
            index = load_index(course_id)
            for e in index["exams"]:
                if e.get("exam_id") == exam_id:
                    e.update(fields)
                    break
            else:
                return
            _save_index(course_id, index)

    async def process(self, course_id: str, exam_id: str, engine: str, lang: str, user_lang: str | None) -> SkillResult:
        key = (course_id, exam_id)
        _ACTIVE_JOBS.add(key)
        try:
            entry = next((e for e in load_index(course_id)["exams"] if e.get("exam_id") == exam_id), None)
            if entry is None:
                return SkillResult(success=False, error="exam_not_found")
            path = exam_dir(course_id, exam_id) / entry["filename"]

            await self._update_entry(course_id, exam_id, stage="extracting")
            from nano_notebooklm.ingest.extractors import extract_file
            pages, _ = await asyncio.to_thread(extract_file, path, engine, lang)
            page_texts = [
                (p.page or p.slide or i + 1, p.text.strip())
                for i, p in enumerate(pages) if (p.text or "").strip()
            ]
            if not page_texts:
                await self._update_entry(course_id, exam_id, status="failed", error="no_text")
                return SkillResult(success=False, error="no_text")

            await self._update_entry(course_id, exam_id, stage="segmenting")
            questions = await self._segment(entry["filename"], page_texts)
            if not questions:
                await self._update_entry(course_id, exam_id, status="failed", error="no_questions_found")
                return SkillResult(success=False, error="no_questions_found")

            await self._update_entry(course_id, exam_id, stage="tagging")
            topics, topics_error = await self._ensure_topics(course_id, user_lang)
            tag_errors = await self._tag(questions, topics) if topics else []

            async with _index_lock(course_id):
                if not exam_dir(course_id, exam_id).exists():
                    return SkillResult(success=False, error="exam_not_found")
                _save_questions(course_id, exam_id, questions)
            # Questions are kept even when tagging failed: the UI offers a
            # re-tag, which costs far fewer calls than re-uploading.
            if tag_errors:
                warning = "tagging_" + _worst(tag_errors)
            else:
                warning = topics_error
            await self._update_entry(
                course_id, exam_id, status="ready", stage="done",
                question_count=len(questions), error=warning,
            )
            return SkillResult(success=True, data={"exam_id": exam_id, "question_count": len(questions)})
        except LLMStepFailed as exc:
            await self._update_entry(course_id, exam_id, status="failed", error=exc.code)
            return SkillResult(success=False, error=exc.code)
        except Exception as exc:
            logger.warning("past_exams process failed for %s/%s: %s", course_id, exam_id, type(exc).__name__)
            await self._update_entry(course_id, exam_id, status="failed", error="processing_failed")
            return SkillResult(success=False, error="processing_failed")
        finally:
            _ACTIVE_JOBS.discard(key)

    def start_processing(self, course_id: str, exam_id: str, engine: str, lang: str, user_lang: str | None) -> None:
        task = asyncio.create_task(self.process(course_id, exam_id, engine, lang, user_lang))
        _BACKGROUND_TASKS.add(task)
        task.add_done_callback(_BACKGROUND_TASKS.discard)

    # ── LLM steps ──────────────────────────────────────────────────

    async def _llm(self, prompt: str, task_type: str, temperature: float, max_tokens: int = 4096) -> Any:
        async with ep._get_variant_semaphore():
            return await asyncio.wait_for(
                self.router.complete_structured(
                    prompt, task_type=task_type,
                    system=prompts.PAST_EXAM_SYSTEM, temperature=temperature,
                    max_tokens=max_tokens,
                ),
                timeout=ep.EXAM_PREP_LLM_TIMEOUT_S,
            )

    async def _segment(self, filename: str, page_texts: list[tuple[int, str]]) -> list[dict]:
        """Raises LLMStepFailed when every window's call failed."""
        questions: list[dict] = []
        seen: set[str] = set()
        errors: list[str] = []
        windows = _windows(page_texts)
        for window in windows:
            prompt = prompts.PAST_EXAM_SEGMENT_PROMPT.format(filename=filename, exam_text=window)
            try:
                # The reply restates every question in full, and reasoning
                # models spend part of the budget on hidden thinking tokens,
                # so the 4096 default risks truncated (unparseable) JSON.
                data = await self._llm(prompt, "past_exam_segment", 0.0, max_tokens=16384)
            except Exception as exc:
                logger.warning("past_exams segment call failed: %s", type(exc).__name__)
                errors.append(llm_error_code(exc))
                continue
            raw = data.get("questions") if isinstance(data, dict) else data
            if not isinstance(raw, list):
                continue
            for rq in raw:
                if not isinstance(rq, dict):
                    continue
                text = str(rq.get("text") or "").strip()
                sig = ep._norm_signature(text)
                if not text or sig in seen:
                    continue
                seen.add(sig)
                kind = str(rq.get("kind") or "other")
                solution = rq.get("solution")
                questions.append({
                    "id": _new_question_id(),
                    "number": str(rq.get("number") or len(questions) + 1)[:16],
                    "text": text[:4000],
                    "points": _coerce_points(rq.get("points")),
                    "kind": kind if kind in QUESTION_KINDS else "other",
                    "solution": str(solution).strip()[:4000] if solution else None,
                    "page": _coerce_int(rq.get("page")),
                    "topic_ids": [],
                    "match": "untagged",
                    "confidence": 0.0,
                    "focus": None,
                })
                if len(questions) >= MAX_QUESTIONS_PER_EXAM:
                    return questions
        if windows and len(errors) == len(windows):
            raise LLMStepFailed(_worst(errors))
        return questions

    async def _ensure_topics(self, course_id: str, user_lang: str | None) -> tuple[list[dict], str | None]:
        """Exam Prep topics to tag against, planned from course material if
        the bank has none yet. Returns (topics, error_token_if_empty)."""
        topics = _bank_topics(course_id)
        if not topics:
            res = await ep.ExamPrepSkill(self.kb, self.router).plan_topics(course_id, {}, user_lang)
            if not res.success:
                logger.info("past_exams: topic planning unavailable for %s: %s", course_id, res.error)
                if res.error == "no_course_content":
                    return [], "no_course_topics"
                return [], "topic_planning_failed"
            topics = _bank_topics(course_id)
            if not topics:
                return [], "no_course_topics"
        out = []
        for t in topics:
            hint = ""
            for sc in t.get("source_chunks") or []:
                chunk = self.kb.find_chunk(sc.get("chunk_id", ""))
                if chunk is not None:
                    hint = " ".join(chunk.text.split())[:240]
                    break
            out.append({"id": t["id"], "name": t["name"], "hint": hint})
        return out, None

    async def _tag(self, questions: list[dict], topics: list[dict]) -> list[str]:
        """Set topic_ids / match / confidence / focus on each question in
        place. Questions the LLM skips or a failed batch stay `untagged`.
        Returns an error token per failed batch."""
        valid = {t["id"] for t in topics}
        topics_block = "\n".join(
            f"- {t['id']}: {t['name']}" + (f" (course excerpt: {t['hint']})" if t["hint"] else "")
            for t in topics
        )

        async def _batch(batch: list[dict]) -> str | None:
            questions_block = "\n\n".join(f"[{q['id']}] {q['text'][:700]}" for q in batch)
            prompt = prompts.PAST_EXAM_TAG_PROMPT.format(
                topics_block=topics_block, questions_block=questions_block,
            )
            try:
                data = await self._llm(prompt, "past_exam_tag", 0.0, max_tokens=8192)
            except Exception as exc:
                logger.warning("past_exams tag call failed: %s", type(exc).__name__)
                return llm_error_code(exc)
            tags = data.get("tags") if isinstance(data, dict) else data
            if not isinstance(tags, list):
                return "llm_failed"
            by_id = {q["id"]: q for q in batch}
            for tag in tags:
                if not isinstance(tag, dict):
                    continue
                q = by_id.get(str(tag.get("question_id") or ""))
                if q is None:
                    continue
                ids = [str(t) for t in tag.get("topic_ids") or [] if str(t) in valid][:2]
                match = str(tag.get("match") or "")
                if match not in ("primary", "partial") or not ids:
                    match, ids = "off_topic", []
                try:
                    conf = max(0.0, min(1.0, float(tag.get("confidence", 0.5))))
                except (TypeError, ValueError):
                    conf = 0.5
                focus = tag.get("focus") if match == "partial" else None
                q.update({
                    "topic_ids": ids,
                    "match": match,
                    "confidence": conf,
                    "focus": str(focus)[:600] if focus else None,
                })
            return None

        batches = [questions[i:i + TAG_BATCH_SIZE] for i in range(0, len(questions), TAG_BATCH_SIZE)]
        results = await asyncio.gather(*(_batch(b) for b in batches))
        return [r for r in results if r]

    # ── read / edit ────────────────────────────────────────────────

    def view(self, course_id: str) -> dict:
        bank = ep.load_bank(course_id)
        topics = [t for t in bank.get("topics", []) if not t.get("archived_topic")]
        valid = {t["id"] for t in topics}
        overrides = load_overrides(course_id)
        index = load_index(course_id)
        exemplar_counts: dict[str, int] = {t["id"]: 0 for t in topics}
        needs_retag = False
        exams_out = []
        for exam in index["exams"]:
            qs_out = []
            if exam.get("status") == "ready" and str(exam.get("error") or "").startswith(("tagging_", "topic_planning_")):
                needs_retag = True
            if exam.get("status") == "ready":
                for raw in load_questions(course_id, exam["exam_id"]):
                    q = _effective(raw, overrides, valid)
                    if q.get("stale_tags") or (q.get("match") == "untagged" and valid):
                        needs_retag = True
                    if q.get("match") in ("primary", "partial") and float(q.get("confidence") or 0) >= EXEMPLAR_MIN_CONFIDENCE:
                        for tid in q["topic_ids"]:
                            exemplar_counts[tid] = exemplar_counts.get(tid, 0) + 1
                    qs_out.append(q)
            exams_out.append({**exam, "questions": qs_out})
        topics_out = []
        for t in topics:
            topics_out.append({
                "id": t["id"],
                "name": t["name"],
                "exemplar_count": exemplar_counts.get(t["id"], 0),
                "past_exam_style_count": sum(
                    1 for q in t.get("questions", [])
                    if not q.get("archived") and q.get("origin") == ep.ORIGIN_PAST_EXAM_STYLE
                ),
            })
        return {
            "exams": exams_out,
            "topics": topics_out,
            "needs_retag": needs_retag,
            "processing": any(e.get("status") == "processing" for e in exams_out),
        }

    async def set_tags(self, course_id: str, question_id: str, topic_ids: list[str], match: str) -> SkillResult:
        if match not in ("primary", "partial", "off_topic"):
            return SkillResult(success=False, error="invalid_match")
        valid = {t["id"] for t in _bank_topics(course_id)}
        topic_ids = list(dict.fromkeys(topic_ids))
        if match == "off_topic":
            topic_ids = []
        elif not topic_ids or any(t not in valid for t in topic_ids):
            return SkillResult(success=False, error="invalid_topic")
        async with _index_lock(course_id):
            known = any(
                q.get("id") == question_id
                for e in load_index(course_id)["exams"]
                for q in load_questions(course_id, e["exam_id"])
            )
            if not known:
                return SkillResult(success=False, error="question_not_found")
            overrides = load_overrides(course_id)
            overrides[question_id] = {"topic_ids": topic_ids, "match": match, "updated_at": ep._now_iso()}
            _write_json(_root(course_id) / "overrides.json", overrides)
        return SkillResult(success=True, data=self.view(course_id))

    async def retag(self, course_id: str, user_lang: str | None) -> SkillResult:
        """Re-run tagging for every ready exam against the current topics.
        Student overrides live in overrides.json and keep winning."""
        topics, topics_error = await self._ensure_topics(course_id, user_lang)
        if not topics:
            return SkillResult(success=False, error=topics_error or "no_course_topics")
        errors: list[str] = []
        async with _index_lock(course_id):
            index = load_index(course_id)
            for exam in index["exams"]:
                if exam.get("status") != "ready":
                    continue
                qs = load_questions(course_id, exam["exam_id"])
                if not qs:
                    continue
                # Re-tag into a copy so a failed batch can't wipe tags that
                # were already good.
                fresh = [{**q, "topic_ids": [], "match": "untagged", "confidence": 0.0, "focus": None} for q in qs]
                exam_errors = await self._tag(fresh, topics)
                if exam_errors:
                    errors.extend(exam_errors)
                    fresh = [f if f["match"] != "untagged" else q for f, q in zip(fresh, qs)]
                _save_questions(course_id, exam["exam_id"], fresh)
                exam["error"] = ("tagging_" + _worst(exam_errors)) if exam_errors else None
            _save_index(course_id, index)
        if errors:
            return SkillResult(success=False, error=_worst(errors))
        return SkillResult(success=True, data=self.view(course_id))

    async def delete(self, course_id: str, exam_id: str) -> SkillResult:
        async with _index_lock(course_id):
            index = load_index(course_id)
            entry = next((e for e in index["exams"] if e.get("exam_id") == exam_id), None)
            if entry is None:
                return SkillResult(success=False, error="exam_not_found")
            if (course_id, exam_id) in _ACTIVE_JOBS:
                return SkillResult(success=False, error="exam_processing")
            qids = {q.get("id") for q in load_questions(course_id, exam_id)}
            shutil.rmtree(exam_dir(course_id, exam_id), ignore_errors=True)
            index["exams"] = [e for e in index["exams"] if e.get("exam_id") != exam_id]
            _save_index(course_id, index)
            overrides = load_overrides(course_id)
            if any(q in overrides for q in qids):
                _write_json(
                    _root(course_id) / "overrides.json",
                    {k: v for k, v in overrides.items() if k not in qids},
                )
        return SkillResult(success=True, data=self.view(course_id))

    # ── generation ─────────────────────────────────────────────────

    async def generate(self, course_id: str, topic_ids: list[str], count: int, user_lang: str | None) -> SkillResult:
        count = max(1, min(10, count))
        exemplars_by_topic = {tid: collect_exemplars(course_id, tid) for tid in dict.fromkeys(topic_ids)}
        skipped = {tid: "no_exemplars" for tid, ex in exemplars_by_topic.items() if not ex}
        targets = {tid: ex for tid, ex in exemplars_by_topic.items() if ex}
        if not targets:
            return SkillResult(success=False, error="no_exemplars")

        prep = ep.ExamPrepSkill(self.kb, self.router)
        async with ep._lock_for(course_id):
            bank = ep.load_bank(course_id)
            topic_by_id = {t["id"]: t for t in bank.get("topics", []) if not t.get("archived_topic")}
            jobs, job_ids, job_errors = [], [], []
            for tid, ex in targets.items():
                topic = topic_by_id.get(tid)
                if topic is None:
                    skipped[tid] = "unknown_topic"
                    continue
                errs: list[str] = []
                jobs.append(prep._generate_questions(
                    course_id, topic, count=count, kinds=GENERATE_KINDS,
                    variant_of=None, user_lang=user_lang, exemplars=ex, errors=errs,
                ))
                job_ids.append(tid)
                job_errors.append(errs)
            results = await asyncio.gather(*jobs, return_exceptions=True)
            added = {tid: (r if isinstance(r, int) else 0) for tid, r in zip(job_ids, results)}
            failed = {}
            for tid, r, errs in zip(job_ids, results, job_errors):
                if isinstance(r, BaseException):
                    errs.append(llm_error_code(r))
                if errs and not added[tid]:
                    failed[tid] = _worst(errs)
            ep.save_bank(course_id, bank)
        if failed and not any(added.values()):
            return SkillResult(success=False, error="generate_" + _worst(list(failed.values())))
        return SkillResult(success=True, data={
            "added": added,
            "skipped": skipped,
            "failed": failed,
            "view": prep._compute_view(bank),
        })
