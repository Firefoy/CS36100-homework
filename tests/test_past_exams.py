"""Past Exams: past-exam ingest, topic tagging, exemplar-driven generation.

The core scenario: a course whose files only teach heapify, plus an exam
with one heapify question and one k-d tree question. The k-d tree question
must be tagged off_topic and never reach generation, and nothing from the
exam may enter the course KB.

LLM calls are faked per task_type; the tag responder reads the question ids
out of the prompt because they're minted at segmentation time.
"""

from __future__ import annotations

import json
import re
import time

import pytest

from nano_notebooklm.skills import exam_prep as ep
from nano_notebooklm.skills import past_exams_skill as pes
from nano_notebooklm.skills.past_exams_skill import PastExamsSkill
from nano_notebooklm.types import Chunk, FileType, SearchResult

HEAPIFY_Q = "Given the array [3, 9, 2, 1, 4, 5], run heapify to build a max-heap and show the array after each sift-down."
KDTREE_Q = "Insert the points (2,3), (5,4), (9,6) into an empty 2-d k-d tree and draw the resulting tree."
HEAP_TOPIC = ep._stable_topic_id("Heapify")


class _FakeRouter:
    """complete_structured fake keyed by task_type. A queued response may be
    a dict, or a callable(prompt) -> dict."""

    def __init__(self, responses: dict[str, list]):
        self.responses = {k: list(v) for k, v in responses.items()}
        self.calls: list[dict] = []

    async def complete_structured(self, prompt, *, system="", task_type="", **kwargs):
        self.calls.append({"prompt": prompt, "system": system, "task_type": task_type})
        queue = self.responses.get(task_type)
        if not queue:
            raise RuntimeError(f"no canned response for {task_type}")
        r = queue.pop(0) if len(queue) > 1 else queue[0]
        return r(prompt) if callable(r) else r

    def prompts_for(self, task_type):
        return [c["prompt"] for c in self.calls if c["task_type"] == task_type]


class _FakeKB:
    def __init__(self, chunks):
        self._chunks = list(chunks)
        self._by_id = {c.chunk_id: c for c in chunks}

    def search(self, query, top_k=10, course_id=None):
        return [
            SearchResult(chunk_id=c.chunk_id, text=c.text, source_file=c.source_file,
                         location=c.location, score=0.5, course_id=c.course_id)
            for c in self._chunks
        ][:top_k]

    def get_chunks(self, course_id=None):
        return list(self._chunks)

    def find_chunk(self, chunk_id):
        return self._by_id.get(chunk_id)


def _heap_chunk() -> Chunk:
    return Chunk(
        chunk_id="c_heap", doc_id="d1", course_id="ds", file_type=FileType.PDF,
        text="Heapify turns an array into a heap by calling sift-down on every internal node, bottom-up, in O(n).",
        source_file="heaps.pdf", location="PDF p.3", page=3,
    )


def _seed_bank(course_id: str = "ds", questions: list[dict] | None = None) -> None:
    bank = ep._empty_bank(course_id)
    bank["topics"] = [{
        "id": HEAP_TOPIC, "name": "Heapify", "weight": 0.8,
        "source_chunks": [{"chunk_id": "c_heap", "source_file": "heaps.pdf", "location": "PDF p.3"}],
        "questions": questions or [], "created_at": ep._now_iso(),
    }]
    ep.save_bank(course_id, bank)


def _segment_response(include_solution: bool = True) -> dict:
    return {"questions": [
        {"number": "1", "text": HEAPIFY_Q, "points": 10, "kind": "trace",
         "solution": "[9, 4, 5, 1, 3, 2]" if include_solution else None, "page": 1},
        {"number": "2", "text": KDTREE_Q, "points": 10, "kind": "trace", "solution": None, "page": 2},
    ]}


def _tag_by_keyword(prompt: str) -> dict:
    """Tag heap questions to the heapify topic, everything else off_topic."""
    tags = []
    for qid, text in re.findall(r"^\[(pq_[a-f0-9]{12})\] (.*)$", prompt, flags=re.M):
        if "heap" in text.lower():
            tags.append({"question_id": qid, "topic_ids": [HEAP_TOPIC], "match": "primary", "confidence": 0.9})
        else:
            tags.append({"question_id": qid, "topic_ids": [], "match": "off_topic", "confidence": 0.95})
    return {"tags": tags}


async def _upload(skill: PastExamsSkill, course_id: str = "ds", body: str | None = None, name: str = "midterm_2024.md") -> str:
    exam_id = pes.new_exam_id()
    d = pes.exam_dir(course_id, exam_id)
    d.mkdir(parents=True)
    (d / name).write_text(body if body is not None else f"# Midterm\n\n1. {HEAPIFY_Q}\n\n2. {KDTREE_Q}\n", encoding="utf-8")
    await skill.register(course_id, exam_id, name, "pymupdf")
    res = await skill.process(course_id, exam_id, "pymupdf", "en", None)
    assert res.success or res.error, res
    return exam_id


def _questions_by_text(view: dict) -> dict[str, dict]:
    return {q["text"]: q for e in view["exams"] for q in e["questions"]}


# ── ingest + tagging ──────────────────────────────────────────────────


async def test_off_topic_exam_questions_are_tagged_and_kb_untouched(isolated_artifacts):
    _seed_bank()
    kb = _FakeKB([_heap_chunk()])
    router = _FakeRouter({"past_exam_segment": [_segment_response()], "past_exam_tag": [_tag_by_keyword]})
    skill = PastExamsSkill(kb, router)

    await _upload(skill)

    view = skill.view("ds")
    assert view["exams"][0]["status"] == "ready"
    by_text = _questions_by_text(view)
    assert by_text[HEAPIFY_Q]["match"] == "primary"
    assert by_text[HEAPIFY_Q]["topic_ids"] == [HEAP_TOPIC]
    assert by_text[KDTREE_Q]["match"] == "off_topic"
    assert by_text[KDTREE_Q]["topic_ids"] == []

    exemplars = pes.collect_exemplars("ds", HEAP_TOPIC)
    assert [e["text"] for e in exemplars] == [HEAPIFY_Q]
    assert view["topics"][0]["exemplar_count"] == 1

    # Exams never enter the course KB / uploads dir.
    assert [c.chunk_id for c in kb.get_chunks("ds")] == ["c_heap"]
    assert not (isolated_artifacts / "uploads").exists()


async def test_tag_prompt_lists_course_topics_with_excerpt(isolated_artifacts):
    _seed_bank()
    router = _FakeRouter({"past_exam_segment": [_segment_response()], "past_exam_tag": [_tag_by_keyword]})
    await _upload(PastExamsSkill(_FakeKB([_heap_chunk()]), router))
    tag_prompt = router.prompts_for("past_exam_tag")[0]
    assert f"- {HEAP_TOPIC}: Heapify" in tag_prompt
    assert "sift-down on every internal node" in tag_prompt


async def test_hallucinated_topic_id_becomes_off_topic(isolated_artifacts):
    _seed_bank()

    def _bad_tags(prompt):
        ids = re.findall(r"^\[(pq_[a-f0-9]{12})\]", prompt, flags=re.M)
        return {"tags": [{"question_id": i, "topic_ids": ["topic_kdtree"], "match": "primary", "confidence": 0.9} for i in ids]}

    router = _FakeRouter({"past_exam_segment": [_segment_response()], "past_exam_tag": [_bad_tags]})
    skill = PastExamsSkill(_FakeKB([_heap_chunk()]), router)
    await _upload(skill)
    assert all(q["match"] == "off_topic" for q in _questions_by_text(skill.view("ds")).values())


async def test_topics_are_planned_when_bank_is_empty(isolated_artifacts):
    router = _FakeRouter({
        "exam_prep_plan": [{"topics": [{"name": "Heapify", "weight": 0.8, "source_chunks": ["c_heap"]}]}],
        "past_exam_segment": [_segment_response()],
        "past_exam_tag": [_tag_by_keyword],
    })
    skill = PastExamsSkill(_FakeKB([_heap_chunk()]), router)
    await _upload(skill)
    assert [t["id"] for t in ep.load_bank("ds")["topics"]] == [HEAP_TOPIC]
    assert _questions_by_text(skill.view("ds"))[HEAPIFY_Q]["match"] == "primary"


async def test_no_course_content_leaves_questions_untagged(isolated_artifacts):
    router = _FakeRouter({"past_exam_segment": [_segment_response()]})
    skill = PastExamsSkill(_FakeKB([]), router)
    await _upload(skill)
    exam = skill.view("ds")["exams"][0]
    assert exam["status"] == "ready"
    assert exam["error"] == "no_course_topics"
    assert {q["match"] for q in exam["questions"]} == {"untagged"}


async def test_empty_file_fails_with_no_text(isolated_artifacts):
    _seed_bank()
    skill = PastExamsSkill(_FakeKB([_heap_chunk()]), _FakeRouter({}))
    await _upload(skill, body="   \n")
    exam = skill.view("ds")["exams"][0]
    assert exam["status"] == "failed"
    assert exam["error"] == "no_text"


async def test_processing_entry_without_live_job_reports_interrupted(isolated_artifacts):
    pes._write_json(pes._root("ds") / "index.json", {"version": 1, "exams": [
        {"exam_id": "abcdefabcdef", "filename": "x.md", "status": "processing", "stage": "tagging"},
    ]})
    exam = pes.load_index("ds")["exams"][0]
    assert exam["status"] == "failed"
    assert exam["error"] == "interrupted"


def test_windows_split_on_page_boundaries():
    pages = [(1, "a" * 7000), (2, "b" * 7000), (3, "c" * 100)]
    windows = pes._windows(pages)
    assert len(windows) == 2
    assert windows[0].startswith("[page 1]")
    assert "[page 2]" in windows[1] and "[page 3]" in windows[1]


# ── overrides + retag ─────────────────────────────────────────────────


async def test_override_survives_retag_and_excludes_exemplar(isolated_artifacts):
    _seed_bank()
    router = _FakeRouter({"past_exam_segment": [_segment_response()], "past_exam_tag": [_tag_by_keyword]})
    skill = PastExamsSkill(_FakeKB([_heap_chunk()]), router)
    await _upload(skill)
    heap_q = _questions_by_text(skill.view("ds"))[HEAPIFY_Q]

    res = await skill.set_tags("ds", heap_q["id"], [], "off_topic")
    assert res.success
    assert pes.collect_exemplars("ds", HEAP_TOPIC) == []

    res = await skill.retag("ds", None)
    assert res.success
    q = _questions_by_text(res.data)[HEAPIFY_Q]
    assert q["match"] == "off_topic" and q["overridden"] is True
    # The underlying LLM tag was refreshed, the override still wins.
    raw = pes.load_questions("ds", res.data["exams"][0]["exam_id"])
    assert next(r for r in raw if r["text"] == HEAPIFY_Q)["match"] == "primary"


async def test_set_tags_rejects_unknown_topic_and_question(isolated_artifacts):
    _seed_bank()
    router = _FakeRouter({"past_exam_segment": [_segment_response()], "past_exam_tag": [_tag_by_keyword]})
    skill = PastExamsSkill(_FakeKB([_heap_chunk()]), router)
    await _upload(skill)
    kd_q = _questions_by_text(skill.view("ds"))[KDTREE_Q]
    assert (await skill.set_tags("ds", kd_q["id"], ["topic_nope"], "primary")).error == "invalid_topic"
    assert (await skill.set_tags("ds", "pq_000000000000", [HEAP_TOPIC], "primary")).error == "question_not_found"


async def test_stale_topic_ids_flag_needs_retag(isolated_artifacts):
    _seed_bank()
    router = _FakeRouter({"past_exam_segment": [_segment_response()], "past_exam_tag": [_tag_by_keyword]})
    skill = PastExamsSkill(_FakeKB([_heap_chunk()]), router)
    await _upload(skill)
    assert skill.view("ds")["needs_retag"] is False

    bank = ep.load_bank("ds")
    bank["topics"][0]["id"] = "topic_renamed"
    ep.save_bank("ds", bank)
    view = skill.view("ds")
    assert view["needs_retag"] is True
    assert _questions_by_text(view)[HEAPIFY_Q]["match"] == "untagged"


async def test_delete_removes_exam_and_its_overrides(isolated_artifacts):
    _seed_bank()
    router = _FakeRouter({"past_exam_segment": [_segment_response()], "past_exam_tag": [_tag_by_keyword]})
    skill = PastExamsSkill(_FakeKB([_heap_chunk()]), router)
    exam_id = await _upload(skill)
    heap_q = _questions_by_text(skill.view("ds"))[HEAPIFY_Q]
    await skill.set_tags("ds", heap_q["id"], [], "off_topic")

    res = await skill.delete("ds", exam_id)
    assert res.success and res.data["exams"] == []
    assert not pes.exam_dir("ds", exam_id).exists()
    assert pes.load_overrides("ds") == {}
    assert (await skill.delete("ds", exam_id)).error == "exam_not_found"


# ── generation ────────────────────────────────────────────────────────


def _mimic_response(prompts_seen=None):
    def _r(prompt):
        return {"questions": [
            {"prompt": "Run heapify on [4, 10, 3, 5, 1] to build a max-heap. What is the final array?",
             "type": "short_answer", "answer": "[10, 5, 3, 4, 1]",
             "explanation": "Sift-down from index 1 then index 0.", "difficulty": "medium"},
        ]}
    return _r


async def test_generate_uses_only_on_topic_exemplars(isolated_artifacts):
    _seed_bank()
    router = _FakeRouter({
        "past_exam_segment": [_segment_response()],
        "past_exam_tag": [_tag_by_keyword],
        "exam_prep_questions": [_mimic_response()],
    })
    skill = PastExamsSkill(_FakeKB([_heap_chunk()]), router)
    await _upload(skill)
    heap_q = _questions_by_text(skill.view("ds"))[HEAPIFY_Q]

    res = await skill.generate("ds", [HEAP_TOPIC], 2, None)
    assert res.success, res.error
    assert res.data["added"] == {HEAP_TOPIC: 1}

    prompt = router.prompts_for("exam_prep_questions")[0]
    assert HEAPIFY_Q in prompt
    assert "k-d tree" not in prompt and "(2,3)" not in prompt
    assert "Official answer: [9, 4, 5, 1, 3, 2]" in prompt
    assert "sift-down on every internal node" in prompt  # grounded in course material

    q = ep.load_bank("ds")["topics"][0]["questions"][0]
    assert q["origin"] == ep.ORIGIN_PAST_EXAM_STYLE
    assert q["exemplar_ids"] == [heap_q["id"]]
    assert q["exemplar_refs"] == ["midterm_2024 · Q1"]


async def test_generate_drops_near_copies_of_exemplars(isolated_artifacts):
    _seed_bank()

    def _copying(prompt):
        return {"questions": [
            {"prompt": HEAPIFY_Q, "type": "short_answer", "answer": "x"},
            {"prompt": "Given the array [3, 9, 2, 1, 4, 5], run heapify to build a max-heap and show the array after each sift down!",
             "type": "short_answer", "answer": "x"},
            {"prompt": "How many sift-down calls does heapify make on an array of 7 elements?",
             "type": "calculation", "answer": "3"},
        ]}

    router = _FakeRouter({
        "past_exam_segment": [_segment_response()],
        "past_exam_tag": [_tag_by_keyword],
        "exam_prep_questions": [_copying],
    })
    skill = PastExamsSkill(_FakeKB([_heap_chunk()]), router)
    await _upload(skill)
    res = await skill.generate("ds", [HEAP_TOPIC], 3, None)
    assert res.data["added"] == {HEAP_TOPIC: 1}
    prompts = [q["prompt"] for q in ep.load_bank("ds")["topics"][0]["questions"]]
    assert prompts == ["How many sift-down calls does heapify make on an array of 7 elements?"]


async def test_generate_without_exemplars_fails(isolated_artifacts):
    _seed_bank()
    skill = PastExamsSkill(_FakeKB([_heap_chunk()]), _FakeRouter({}))
    res = await skill.generate("ds", [HEAP_TOPIC], 3, None)
    assert res.error == "no_exemplars"


async def test_variant_of_past_exam_question_reuses_exemplars(isolated_artifacts):
    _seed_bank()
    router = _FakeRouter({
        "past_exam_segment": [_segment_response()],
        "past_exam_tag": [_tag_by_keyword],
        "exam_prep_questions": [_mimic_response()],
    })
    skill = PastExamsSkill(_FakeKB([_heap_chunk()]), router)
    await _upload(skill)
    await skill.generate("ds", [HEAP_TOPIC], 1, None)
    qid = ep.load_bank("ds")["topics"][0]["questions"][0]["id"]

    router.responses["exam_prep_questions"] = [lambda p: {"questions": [
        {"prompt": "Build a max-heap from [2, 8, 6, 1] with heapify. Final array?", "type": "short_answer", "answer": "[8, 2, 6, 1]"},
    ]}]
    prep = ep.ExamPrepSkill(skill.kb, router)
    await prep.submit_answers("ds", {"answers": {qid: "wrong"}}, None)
    await __import__("asyncio").gather(*list(ep._BACKGROUND_VARIANT_TASKS), return_exceptions=True)

    variant_prompt = router.prompts_for("exam_prep_questions")[-1]
    assert HEAPIFY_Q in variant_prompt
    variants = [q for q in ep.load_bank("ds")["topics"][0]["questions"] if q.get("variant_of") == qid]
    assert variants and variants[0]["origin"] == ep.ORIGIN_PAST_EXAM_STYLE


# ── exam bank integration ─────────────────────────────────────────────


def _q(qid: str, origin: str) -> dict:
    return {
        "id": qid, "type": "short_answer", "prompt": f"prompt {qid}", "options": None,
        "answer": "x", "explanation": "", "difficulty": "medium", "concepts": [],
        "variant_of": None, "created_at": ep._now_iso(), "history": [],
        "consecutive_correct": 0, "mastered": False, "archived": False, "origin": origin,
    }


async def test_next_quiz_origin_filter(isolated_artifacts):
    _seed_bank(questions=[_q("q_gen", ep.ORIGIN_GENERATED), _q("q_past", ep.ORIGIN_PAST_EXAM_STYLE)])
    prep = ep.ExamPrepSkill(_FakeKB([_heap_chunk()]), _FakeRouter({}))

    res = await prep.next_quiz("ds", {"origin": "past_exam_style"}, None)
    assert [q["id"] for q in res.data["questions"]] == ["q_past"]
    assert res.data["view"]["topics"][0]["past_exam_style_count"] == 1

    res = await prep.next_quiz("ds", {}, None)
    assert {q["id"] for q in res.data["questions"]} == {"q_gen", "q_past"}


async def test_next_quiz_past_exam_origin_without_questions_reports_reason(isolated_artifacts):
    _seed_bank(questions=[_q("q_gen", ep.ORIGIN_GENERATED)])
    prep = ep.ExamPrepSkill(_FakeKB([_heap_chunk()]), _FakeRouter({}))
    res = await prep.next_quiz("ds", {"origin": "past_exam_style"}, None)
    assert res.data["questions"] == []
    assert res.data["reason"] == "no_past_exam_questions"


def test_v1_bank_migrates_origin(isolated_artifacts):
    path = ep._bank_path("ds")
    path.parent.mkdir(parents=True)
    q = _q("q_old", ep.ORIGIN_GENERATED)
    del q["origin"]
    path.write_text(json.dumps({"version": 1, "course_id": "ds", "topics": [
        {"id": HEAP_TOPIC, "name": "Heapify", "weight": 0.5, "source_chunks": [], "questions": [q]},
    ]}), encoding="utf-8")
    bank = ep.load_bank("ds")
    assert bank["version"] == ep.EXAM_BANK_VERSION == 2
    assert bank["topics"][0]["questions"][0]["origin"] == ep.ORIGIN_GENERATED


# ── API ───────────────────────────────────────────────────────────────


@pytest.fixture
def past_client(monkeypatch, tmp_path, sample_chunks, fake_embed_fn):
    art = tmp_path / "artifacts"
    (art / "courses" / "testcourse").mkdir(parents=True)
    (art / "courses" / "testcourse" / "chunks.json").write_text(
        json.dumps([c.model_dump() for c in sample_chunks], default=str)
    )
    (art / "courses" / "testcourse" / "course_meta.json").write_text(
        json.dumps({"course_id": "testcourse", "name": "Test Course", "documents": ["d1"]})
    )
    monkeypatch.setenv("ARTIFACTS_DIR", str(art))
    from nano_notebooklm import config
    monkeypatch.setattr(config, "ARTIFACTS_DIR", art)
    from nano_notebooklm.kb import store as kb_store
    monkeypatch.setattr(kb_store, "_get_default_embed_fn", lambda: fake_embed_fn)

    import importlib
    import api.server as server_mod
    importlib.reload(server_mod)
    server_mod.kb.build_index("testcourse")

    topic_id = ep._stable_topic_id("Backpropagation")
    bank = ep._empty_bank("testcourse")
    bank["topics"] = [{
        "id": topic_id, "name": "Backpropagation", "weight": 0.7,
        "source_chunks": [{"chunk_id": "c1", "source_file": "ml.pdf", "location": "PDF p.1"}],
        "questions": [], "created_at": ep._now_iso(),
    }]
    ep.save_bank("testcourse", bank)

    def _tags(prompt):
        out = []
        for qid, text in re.findall(r"^\[(pq_[a-f0-9]{12})\] (.*)$", prompt, flags=re.M):
            hit = "gradient" in text.lower()
            out.append({"question_id": qid, "topic_ids": [topic_id] if hit else [],
                        "match": "primary" if hit else "off_topic", "confidence": 0.9})
        return {"tags": out}

    router = _FakeRouter({
        "past_exam_segment": [{"questions": [
            {"number": "1", "text": "Compute the gradient of the loss w.r.t. w1 for this 2-layer network.", "kind": "calculation"},
            {"number": "2", "text": "Explain the Bellman optimality equation.", "kind": "short_answer"},
        ]}],
        "past_exam_tag": [_tags],
        "exam_prep_questions": [lambda p: {"questions": [
            {"prompt": "For a 1-hidden-layer net with sigmoid units, what is dL/dw2 when ...?", "type": "short_answer", "answer": "a*(y-t)"},
        ]}],
    })
    server_mod.orchestrator.skills["past_exams"].router = router
    server_mod.orchestrator.skills["exam_prep"].router = router

    from fastapi.testclient import TestClient
    with TestClient(server_mod.app) as c:
        c.server_mod = server_mod
        c.topic_id = topic_id
        yield c


def _wait_ready(client, course="testcourse", timeout=10.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/past-exams/{course}").json()
        if not body["processing"]:
            return body
        time.sleep(0.05)
    raise AssertionError("past exam processing did not finish")


def test_api_upload_tag_generate_and_quiz(past_client):
    c = past_client
    r = c.post(
        "/api/past-exams/testcourse/upload",
        files=[("files", ("final_2023.md", b"1. gradient question\n2. bellman question\n", "text/markdown"))],
    )
    assert r.status_code == 200, r.text
    assert len(r.json()["exam_ids"]) == 1

    body = _wait_ready(c)
    exam = body["exams"][0]
    assert exam["status"] == "ready" and exam["question_count"] == 2
    assert {q["match"] for q in exam["questions"]} == {"primary", "off_topic"}
    assert body["topics"][0]["exemplar_count"] == 1

    # Exams stay out of the course KB.
    texts = [ch.text for ch in c.server_mod.kb.get_chunks("testcourse")]
    assert not any("Bellman optimality" in t for t in texts)

    r = c.post("/api/past-exams/testcourse/generate", json={"topic_ids": [c.topic_id], "count": 1})
    assert r.status_code == 200, r.text
    assert r.json()["added"] == {c.topic_id: 1}

    r = c.post("/api/exam-prep/quiz/next", json={"course_id": "testcourse", "origin": "past_exam_style"})
    assert r.status_code == 200
    qs = r.json()["questions"]
    assert len(qs) == 1 and qs[0]["origin"] == "past_exam_style"
    assert qs[0]["exemplar_refs"] == ["final_2023 · Q1"]

    off_q = next(q for q in exam["questions"] if q["match"] == "off_topic")
    r = c.patch(f"/api/past-exams/testcourse/questions/{off_q['id']}",
                json={"topic_ids": [c.topic_id], "match": "partial"})
    assert r.status_code == 200
    assert r.json()["topics"][0]["exemplar_count"] == 2

    r = c.delete(f"/api/past-exams/testcourse/{exam['exam_id']}")
    assert r.status_code == 200 and r.json()["exams"] == []


def test_api_validation(past_client):
    c = past_client
    r = c.post("/api/past-exams/testcourse/upload", files=[("files", ("x.exe", b"MZ", "application/octet-stream"))])
    assert r.status_code == 400
    assert c.get("/api/past-exams/..hack").status_code == 400
    assert c.delete("/api/past-exams/testcourse/not-an-id").status_code == 400
    assert c.delete("/api/past-exams/testcourse/abcdefabcdef").status_code == 404
    assert c.patch("/api/past-exams/testcourse/questions/bad", json={"match": "off_topic"}).status_code == 400
    r = c.post("/api/past-exams/testcourse/generate", json={"topic_ids": [c.topic_id]})
    assert r.status_code == 400 and "no_exemplars" in r.text
    r = c.post("/api/past-exams/testcourse/generate", json={"topic_ids": [c.topic_id], "extra": 1})
    assert r.status_code in (400, 422)
