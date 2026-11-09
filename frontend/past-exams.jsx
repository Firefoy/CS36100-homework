/* global React, API, ExamPrepQuiz, ExamPrepResult */
// Past Exams — upload past exam papers, review how each question was
// matched to this course's Exam Prep topics, and practise with new
// questions generated in the same style.
//
// Views (controlled by `view`):
//   "manage" — upload + exam list with per-question topic tags + practice panel
//   "quiz"   — reuses ExamPrepQuiz on origin=past_exam_style questions
//   "result" — reuses ExamPrepResult
//
// Exams process in a background task on the server; while any exam is
// processing this component polls /api/past-exams/{course_id}.

const { useState: usePE, useEffect: usePEEffect, useRef: usePERef } = React;

const PAST_EXAM_POLL_MS = 3000;
const PAST_EXAM_LOW_CONFIDENCE = 0.5;

function PastExams({ activeCourse, userLang }) {
  const t = (k, vars) => window.I18N.t(k, userLang || "en", vars);
  const [data, setData] = usePE(null);
  const [view, setView] = usePE("manage");
  const [busyLabel, setBusyLabel] = usePE("");
  const [error, setError] = usePE("");
  const [info, setInfo] = usePE("");
  const [engine, setEngine] = usePE("pymupdf");
  const [expanded, setExpanded] = usePE({});
  const [selected, setSelected] = usePE({});
  const [count, setCount] = usePE(3);
  const [quizQuestions, setQuizQuestions] = usePE([]);
  const [answers, setAnswers] = usePE({});
  const [graded, setGraded] = usePE(null);
  const fileRef = usePERef(null);
  // Results that land after the user switched course are dropped.
  const courseRef = usePERef(activeCourse);
  courseRef.current = activeCourse;

  function errText(e, fallbackKey) {
    const token = e && e.message;
    if (token) {
      const key = "pastexam.error." + token;
      const s = t(key);
      if (s !== key) return s;
    }
    return token || t(fallbackKey);
  }

  async function load(courseId, { quiet = false } = {}) {
    try {
      const d = await API.pastExamsView(courseId);
      if (courseRef.current !== courseId) return;
      setData(d);
    } catch (e) {
      if (courseRef.current !== courseId || quiet) return;
      setError(errText(e, "pastexam.error.failed_load"));
    }
  }

  usePEEffect(() => {
    setData(null); setView("manage"); setError(""); setInfo("");
    setExpanded({}); setSelected({}); setQuizQuestions([]); setAnswers({}); setGraded(null);
    if (activeCourse) load(activeCourse);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeCourse]);

  usePEEffect(() => {
    if (!activeCourse || !data || !data.processing) return undefined;
    const courseId = activeCourse;
    const timer = setTimeout(() => load(courseId, { quiet: true }), PAST_EXAM_POLL_MS);
    return () => clearTimeout(timer);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeCourse, data]);

  async function run(labelKey, fn, fallbackErrKey) {
    if (busyLabel) return;
    const courseId = activeCourse;
    setBusyLabel(t(labelKey)); setError(""); setInfo("");
    try {
      await fn(courseId);
    } catch (e) {
      if (courseRef.current === courseId) setError(errText(e, fallbackErrKey));
    } finally {
      setBusyLabel("");
    }
  }

  function onFilesPicked(e) {
    const files = Array.from(e.target.files || []);
    e.target.value = "";
    if (!files.length) return;
    run("pastexam.busy.uploading", async (courseId) => {
      await API.pastExamsUpload(courseId, files, {
        engine, lang: userLang === "zh" ? "ch" : "en", userLang,
      });
      await load(courseId);
    }, "pastexam.error.failed_upload");
  }

  function onRetag() {
    run("pastexam.busy.retagging", async (courseId) => {
      const d = await API.pastExamsRetag(courseId, { userLang });
      if (courseRef.current === courseId) setData(d);
    }, "pastexam.error.failed_retag");
  }

  function onDelete(exam) {
    if (!window.confirm(t("pastexam.confirm.delete", { name: exam.filename }))) return;
    run("pastexam.busy.deleting", async (courseId) => {
      const d = await API.pastExamsDelete(courseId, exam.exam_id);
      if (courseRef.current === courseId) setData(d);
    }, "pastexam.error.failed_delete");
  }

  function onTagChange(q, value) {
    const body = value === "off_topic"
      ? { topicIds: [], match: "off_topic" }
      : { topicIds: [value], match: q.match === "partial" ? "partial" : "primary" };
    run("pastexam.busy.saving", async (courseId) => {
      const d = await API.pastExamsSetTags(courseId, q.id, body);
      if (courseRef.current === courseId) setData(d);
    }, "pastexam.error.failed_save");
  }

  const topics = (data && data.topics) || [];
  const matchedTopics = topics.filter(tp => tp.exemplar_count > 0);
  const selectedIds = matchedTopics.filter(tp => selected[tp.id]).map(tp => tp.id);
  const practiceAvailable = (selectedIds.length ? matchedTopics.filter(tp => selected[tp.id]) : matchedTopics)
    .some(tp => tp.past_exam_style_count > 0);

  function onGenerate() {
    if (!selectedIds.length) return;
    run("pastexam.busy.generating", async (courseId) => {
      const res = await API.pastExamsGenerate(courseId, selectedIds, { count, userLang });
      const added = Object.values(res.added || {}).reduce((s, n) => s + n, 0);
      if (courseRef.current !== courseId) return;
      setInfo(added > 0 ? t("pastexam.info.generated", { n: added }) : t("pastexam.info.generated_none"));
      await load(courseId);
    }, "pastexam.error.failed_generate");
  }

  function onPractice() {
    run("pastexam.busy.sampling", async (courseId) => {
      const res = await API.examPrepNextQuiz(courseId, {
        size: 8,
        topicIds: selectedIds.length ? selectedIds : null,
        userLang,
        origin: "past_exam_style",
      });
      if (courseRef.current !== courseId) return;
      if (!res.questions || res.questions.length === 0) {
        const reason = res.reason || "no_past_exam_questions";
        setError(errText({ message: reason }, "pastexam.error.no_past_exam_questions"));
        return;
      }
      setQuizQuestions(res.questions);
      setAnswers({}); setGraded(null);
      setView("quiz");
    }, "pastexam.error.failed_start");
  }

  function onSubmit() {
    if (!Object.keys(answers).length) return;
    run("pastexam.busy.grading", async (courseId) => {
      const res = await API.examPrepSubmit(courseId, answers, { userLang });
      if (courseRef.current !== courseId) return;
      setGraded(res);
      setView("result");
    }, "pastexam.error.failed_submit");
  }

  if (!activeCourse) {
    return (
      <div className="reader-body exam-prep-wrap">
        <div className="exam-prep-empty">
          <h2>{t("pastexam.title")}</h2>
          <p>{t("pastexam.empty_select_course")}</p>
        </div>
      </div>
    );
  }

  const exams = (data && data.exams) || [];

  return (
    <div className="reader-body exam-prep-wrap past-exams" data-screen-label={t("pastexam.title")}>
      <header className="exam-prep-header">
        <div>
          <h2>{t("pastexam.title")} · {activeCourse}</h2>
          <p className="pe-intro">{t("pastexam.intro")}</p>
        </div>
      </header>

      {error && <div className="error-banner">{error}</div>}
      {info && <div className="pe-info-banner">{info}</div>}
      {busyLabel && <div className="pe-busy mono">{busyLabel}</div>}

      {view === "quiz" && (
        <ExamPrepQuiz
          t={t}
          questions={quizQuestions}
          scope={null}
          answers={answers}
          onAnswer={(qid, ans) => setAnswers(a => ({ ...a, [qid]: ans }))}
          onSubmit={onSubmit}
          onCancel={() => setView("manage")}
          loading={!!busyLabel}
        />
      )}

      {view === "result" && graded && (
        <ExamPrepResult
          t={t}
          graded={graded}
          questions={quizQuestions}
          answers={answers}
          onContinue={() => { setView("manage"); setGraded(null); load(activeCourse); }}
          onAgain={onPractice}
        />
      )}

      {view === "manage" && (
        <React.Fragment>
          <section className="pe-section">
            <div className="pe-upload-row">
              <button className="btn primary" onClick={() => fileRef.current && fileRef.current.click()} disabled={!!busyLabel}>
                {t("pastexam.action.upload")}
              </button>
              <select className="pe-select" value={engine} onChange={e => setEngine(e.target.value)} disabled={!!busyLabel}>
                <option value="pymupdf">{t("pastexam.engine.fast")}</option>
                <option value="mineru">{t("pastexam.engine.ocr")}</option>
              </select>
              <input
                ref={fileRef}
                type="file"
                multiple
                accept=".pdf,.docx,.pptx,.ppt,.md,.txt"
                style={{ display: "none" }}
                onChange={onFilesPicked}
              />
            </div>

            {data && data.needs_retag && (
              <div className="pe-retag-banner">
                <span>{t("pastexam.retag.banner")}</span>
                <button className="btn ghost" onClick={onRetag} disabled={!!busyLabel}>{t("pastexam.action.retag")}</button>
              </div>
            )}

            {data && exams.length === 0 && <p className="pe-empty">{t("pastexam.empty_exams")}</p>}

            <div className="pe-exam-list">
              {exams.map(exam => (
                <PastExamCard
                  key={exam.exam_id}
                  t={t}
                  exam={exam}
                  topics={topics}
                  expanded={!!expanded[exam.exam_id]}
                  onToggle={() => setExpanded(x => ({ ...x, [exam.exam_id]: !x[exam.exam_id] }))}
                  onDelete={() => onDelete(exam)}
                  onTagChange={onTagChange}
                  busy={!!busyLabel}
                />
              ))}
            </div>
          </section>

          {exams.some(e => e.status === "ready") && (
            <section className="pe-section">
              <h3 className="pe-section-title">{t("pastexam.practice.title")}</h3>
              <p className="pe-hint">{t("pastexam.practice.hint")}</p>
              {matchedTopics.length === 0 ? (
                <p className="pe-empty">{t("pastexam.practice.no_topics")}</p>
              ) : (
                <React.Fragment>
                  <div className="pe-topic-grid">
                    {matchedTopics.map(tp => (
                      <label key={tp.id} className={"pe-topic" + (selected[tp.id] ? " selected" : "")}>
                        <input
                          type="checkbox"
                          checked={!!selected[tp.id]}
                          onChange={() => setSelected(s => ({ ...s, [tp.id]: !s[tp.id] }))}
                        />
                        <span className="pe-topic-body">
                          <span className="pe-topic-name">{tp.name}</span>
                          <span className="pe-topic-meta mono">
                            {t("pastexam.topic.exemplars", { n: tp.exemplar_count })}
                            {" · "}
                            {t("pastexam.topic.generated", { n: tp.past_exam_style_count })}
                          </span>
                        </span>
                      </label>
                    ))}
                  </div>
                  {topics.length > matchedTopics.length && (
                    <p className="pe-hint">{t("pastexam.topic.unmatched", { n: topics.length - matchedTopics.length })}</p>
                  )}
                  <div className="pe-practice-row">
                    <label className="pe-count">
                      <span>{t("pastexam.count_label")}</span>
                      <select className="pe-select" value={count} onChange={e => setCount(Number(e.target.value))}>
                        {[1, 2, 3, 5, 8].map(n => <option key={n} value={n}>{n}</option>)}
                      </select>
                    </label>
                    <button className="btn ghost" onClick={onGenerate} disabled={!!busyLabel || !selectedIds.length}
                      title={selectedIds.length ? "" : t("pastexam.tooltip.select_topics")}>
                      {t("pastexam.action.generate", { n: count })}
                    </button>
                    <button className="btn primary" onClick={onPractice} disabled={!!busyLabel || !practiceAvailable}
                      title={practiceAvailable ? "" : t("pastexam.tooltip.generate_first")}>
                      {t("pastexam.action.practice")}
                    </button>
                  </div>
                </React.Fragment>
              )}
            </section>
          )}
        </React.Fragment>
      )}
    </div>
  );
}

function PastExamCard({ t, exam, topics, expanded, onToggle, onDelete, onTagChange, busy }) {
  const qs = exam.questions || [];
  const onTopic = qs.filter(q => q.match === "primary" || q.match === "partial").length;
  const offTopic = qs.filter(q => q.match === "off_topic").length;
  const stageKey = "pastexam.stage." + (exam.stage || "queued");
  return (
    <div className={"pe-exam status-" + exam.status}>
      <div className="pe-exam-head">
        <button className="pe-exam-toggle" onClick={onToggle} disabled={exam.status !== "ready"}>
          <span className="mono">{exam.status === "ready" ? (expanded ? "▾" : "▸") : "·"}</span>
          <span className="pe-exam-name">{exam.filename}</span>
        </button>
        <span className={"pe-chip pe-chip-" + exam.status}>{t("pastexam.status." + exam.status)}</span>
        {exam.status === "ready" && (
          <span className="pe-exam-summary mono">
            {t("pastexam.exam.summary", { n: qs.length, on: onTopic, off: offTopic })}
          </span>
        )}
        {exam.status === "processing" && <span className="pe-exam-summary mono">{t(stageKey)}</span>}
        <button className="btn ghost danger pe-delete" onClick={onDelete} disabled={busy || exam.status === "processing"}>
          {t("pastexam.action.delete")}
        </button>
      </div>
      {exam.status === "failed" && (
        <div className="pe-exam-error">{t("pastexam.error." + (exam.error || "processing_failed"))}</div>
      )}
      {exam.status === "ready" && exam.error === "no_course_topics" && (
        <div className="pe-exam-error">{t("pastexam.error.no_course_topics")}</div>
      )}
      {expanded && exam.status === "ready" && (
        <div className="pe-question-list">
          {qs.map(q => (
            <PastExamQuestion key={q.id} t={t} q={q} topics={topics} onTagChange={onTagChange} busy={busy} />
          ))}
        </div>
      )}
    </div>
  );
}

function PastExamQuestion({ t, q, topics, onTagChange, busy }) {
  const [open, setOpen] = usePE(false);
  const value = q.match === "off_topic" ? "off_topic" : (q.topic_ids && q.topic_ids[0]) || "";
  const lowConf = !q.overridden && (q.match === "primary" || q.match === "partial")
    && Number(q.confidence || 0) < PAST_EXAM_LOW_CONFIDENCE;
  return (
    <div className={"pe-question match-" + q.match + (lowConf ? " low-conf" : "")}>
      <div className="pe-q-head">
        <span className="pe-q-num">Q{q.number}</span>
        <span className="mono dim">{q.kind}</span>
        {q.points != null && <span className="mono dim">{t("pastexam.q.points", { n: q.points })}</span>}
        {q.page != null && <span className="mono dim">{t("pastexam.q.page", { n: q.page })}</span>}
        {q.solution && <span className="pe-badge">{t("pastexam.q.has_solution")}</span>}
        {q.match === "partial" && <span className="pe-badge">{t("pastexam.q.partial")}</span>}
        {lowConf && <span className="pe-badge warn">{t("pastexam.q.low_conf")}</span>}
        {q.overridden && <span className="pe-badge">{t("pastexam.q.edited")}</span>}
        <select
          className="pe-select pe-tag-select"
          value={value}
          onChange={e => onTagChange(q, e.target.value)}
          disabled={busy}
          aria-label={t("pastexam.q.tag_label")}
        >
          {value === "" && <option value="" disabled>{t("pastexam.q.untagged")}</option>}
          {topics.map(tp => <option key={tp.id} value={tp.id}>{tp.name}</option>)}
          <option value="off_topic">{t("pastexam.q.off_topic")}</option>
        </select>
      </div>
      <p className={"pe-q-text" + (open ? " open" : "")} onClick={() => setOpen(o => !o)}>{q.text}</p>
      {q.focus && <p className="pe-q-focus">{t("pastexam.q.focus", { text: q.focus })}</p>}
    </div>
  );
}

Object.assign(window, { PastExams });
