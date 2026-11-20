"""Form/answer parsing and the item-analysis engine.

The engine is deterministic: it takes the current forms, answer rows and human
review decisions, and produces a complete snapshot.  The caller persists that
snapshot in one transaction together with the change that triggered it, so
forms, mapping corrections, review decisions and statistics always become
visible at the same moment.

Memory use is bounded independently of the number of stored cards.  Analysis
walks ``answer_rows`` in card-number order, one bounded batch at a time
(:data:`CARD_BATCH_SIZE`), and hands each batch to a sink that writes the
per-card verdicts straight into the snapshot tables.  Only the per-question
aggregates (questions x options, defined by the forms) are kept in memory.
"""
from __future__ import annotations

import csv
import io
import json
import re
from collections import OrderedDict

# How many cards are analysed and written per batch.  Peak memory is
# O(CARD_BATCH_SIZE * questions per card), so it depends on the form shape but
# never on how many cards the database holds.
CARD_BATCH_SIZE = 500


class ValidationError(ValueError):
    """Raised for bad CSV input or an invalid form/mapping edit."""


# --------------------------------------------------------------------------
# CSV parsing
# --------------------------------------------------------------------------

FORM_HEADERS = {
    "q_no": {"题号", "卷式题号", "印刷题号", "q_no", "qno", "question", "题号\n"},
    "letter": {"选项字母", "字母", "涂选字母", "option", "letter", "选项"},
    "std_code": {"标准题码", "题码", "标准题号", "std_code", "code", "standard_code"},
    "std_option": {"标准选项", "标准答案选项", "std_option", "standard_option"},
}
ANSWER_HEADERS = {
    "card_no": {"卡号", "答题卡号", "卡编号", "card_no", "card", "card_id"},
    "q_no": {"题号", "卷式题号", "印刷题号", "q_no", "qno", "question"},
    "letter": {"涂选字母", "选项字母", "字母", "letter", "option", "作答"},
    "scanned_code": {"实扫题码", "扫描题码", "实扫题号", "scanned_code", "scan_code", "scanned"},
}


def _decode(raw: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ValidationError("无法识别文件编码，请使用 UTF-8 或 GBK 编码的 CSV")


def _norm(value) -> str:
    return (value or "").strip()


def _locate_header(header, spec: dict):
    found = {}
    for idx, cell in enumerate(header):
        key = cell.lower().replace(" ", "")
        if not key:
            continue
        for field, aliases in spec.items():
            if key in {a.lower() for a in aliases} and field not in found:
                found[field] = idx
    return found


def _iter_rows(raw, spec: dict, kind: str):
    """Stream normalised data rows without materialising the whole file.

    Accepts bytes or already-decoded text.  The header (when present) is
    consumed once; headerless files use the canonical column order.
    """
    text = _decode(raw) if isinstance(raw, (bytes, bytearray)) else raw
    reader = csv.reader(io.StringIO(text))
    positions = None
    saw_any = False
    for raw_row in reader:
        row = [_norm(c) for c in raw_row]
        if not any(row):
            continue
        if positions is None:
            header = _locate_header(row, spec)
            if header:
                missing = [f for f in spec if f not in header]
                if missing:
                    raise ValidationError(f"{kind}缺少列：{', '.join(missing)}")
                positions = header
                saw_any = True
                continue
            # Headerless file: accept the canonical column order.
            positions = {field: i for i, field in enumerate(spec)}
        saw_any = True
        yield tuple(row[positions[f]] if positions[f] < len(row) else "" for f in spec)
    if not saw_any:
        raise ValidationError(f"{kind}为空")


def _read_rows(raw: bytes, spec: dict, kind: str):
    """Materialise every data row.  Only for small inputs (form CSVs)."""
    return list(_iter_rows(raw, spec, kind))


def decode_csv(raw: bytes) -> str:
    """Decode an uploaded CSV to text so job payloads stay UTF-8 safe."""
    return _decode(raw)


def validate_form_rows(rows):
    """Validate one form's mapping rows. Returns the same rows on success."""
    seen = set()
    q_to_code: dict[str, str] = {}
    code_to_q: dict[str, str] = {}
    code_letters: dict[str, dict[str, str]] = {}
    for q_no, letter, std_code, std_option in rows:
        if not q_no or not letter:
            raise ValidationError(f"卷式行缺少题号或选项字母：{q_no or '?'}/{letter or '?'}")
        if not std_code or not std_option:
            raise ValidationError(f"卷式行缺少标准题码或标准选项：{q_no}/{letter}")
        if (q_no, letter) in seen:
            raise ValidationError(f"卷式内题号 {q_no} 的选项 {letter} 重复")
        seen.add((q_no, letter))
        if q_to_code.setdefault(q_no, std_code) != std_code:
            raise ValidationError(
                f"卷式内题号 {q_no} 映射到多个标准题码（{q_to_code[q_no]} / {std_code}）"
            )
        if code_to_q.setdefault(std_code, q_no) != q_no:
            raise ValidationError(
                f"标准题码 {std_code} 在本卷出现多次（题号 {code_to_q[std_code]} / {q_no}）"
            )
        letters = code_letters.setdefault(std_code, {})
        if std_option in letters:
            raise ValidationError(
                f"标准题码 {std_code} 的标准选项 {std_option} 重复"
            )
        letters[std_option] = letter
    return rows


def parse_form_csv(raw: bytes):
    """Return [(q_no, letter, std_code, std_option), ...] validated per form."""
    rows = _read_rows(raw, FORM_HEADERS, "卷式 CSV")
    if not rows:
        raise ValidationError("卷式 CSV 没有数据行")
    return validate_form_rows(rows)


def iter_answer_csv(raw):
    """Yield normalised answer rows one at a time (constant memory).

    Shape/emptiness is validated per row; duplicate (card, question) pairs are
    left to the ``answer_rows`` primary key during import so this never needs a
    file-sized set.
    """
    count = 0
    for card_no, q_no, letter, scanned in _iter_rows(raw, ANSWER_HEADERS, "答题 CSV"):
        if not card_no or not q_no:
            raise ValidationError("答题行缺少卡号或题号")
        count += 1
        yield {"card_no": card_no, "q_no": q_no,
               "letter": letter, "scanned_code": scanned}
    if not count:
        raise ValidationError("答题 CSV 没有数据行")


def validate_answer_csv(raw) -> int:
    """Stream-validate an answer CSV, returning its row count."""
    return sum(1 for _ in iter_answer_csv(raw))


def parse_answer_csv(raw: bytes):
    """Return [{card_no, q_no, letter, scanned_code}, ...] (small inputs/tests)."""
    seen = set()
    out = []
    for row in iter_answer_csv(raw):
        key = (row["card_no"], row["q_no"])
        if key in seen:
            raise ValidationError(
                f"卡号 {row['card_no']} 的题号 {row['q_no']} 出现多行"
            )
        seen.add(key)
        out.append(row)
    return out


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------

def _natural_key(text: str):
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", text)]


def build_forms(form_rows) -> "OrderedDict[int, dict]":
    """form_rows: iterable of (form_id, form_name, q_no, letter, std_code, std_option)."""
    forms: "OrderedDict[int, dict]" = OrderedDict()
    for form_id, form_name, q_no, letter, std_code, std_option in form_rows:
        form = forms.setdefault(
            form_id,
            {"id": form_id, "name": form_name, "qmap": {}, "codes": {},
             "options": OrderedDict()},
        )
        entry = form["qmap"].setdefault(
            q_no, {"std_code": std_code, "letters": {}, "rev": {}}
        )
        entry["letters"][letter] = std_option
        entry["rev"][std_option] = letter
        form["codes"][std_code] = q_no
        form["options"].setdefault(std_code, set()).add(std_option)
    return forms


def form_is_compatible(form: dict, card_rows: dict) -> bool:
    """Full-card question numbers match and every non-empty code/letter fits."""
    if set(form["qmap"].keys()) != set(card_rows.keys()):
        return False
    for q_no, row in card_rows.items():
        mapping = form["qmap"][q_no]
        if row["scanned_code"] and mapping["std_code"] != row["scanned_code"]:
            return False
        if row["letter"] and row["letter"] not in mapping["letters"]:
            return False
    return True


class Analysis:
    """Aggregated state of one snapshot.

    ``stats``/``unmatched_count``/``flagged_count`` are always kept.  The
    per-card lists (``card_results``/``flagged``/``unmatched_cards``) are only
    filled by the in-memory :func:`analyse` path; the streaming path writes
    them straight to the snapshot tables so memory does not grow with the
    number of stored cards.
    """

    def __init__(self, collect_details: bool = True):
        self.question_codes: list[str] = []
        self.options: dict[str, list[str]] = {}
        self.stats: dict[str, dict] = {}
        self.card_results: list[dict] = []
        self.unmatched_cards: list[dict] = []
        self.flagged: list[dict] = []
        self.cards_total = 0
        self.cards_matched = 0
        self.unmatched_count = 0
        self.flagged_count = 0
        self.collect_details = collect_details

    def summary(self) -> dict:
        return {
            "cards_total": self.cards_total,
            "cards_matched": self.cards_matched,
            "cards_unmatched": self.unmatched_count,
            "questions": len(self.question_codes),
            "flagged_cells": self.flagged_count,
        }

    def add(self, card_no, std_code, status, std_option, detail):
        """Sink interface: keep details only for small in-memory runs."""
        if not self.collect_details:
            return
        self.card_results.append(
            {"card_no": card_no, "std_code": std_code, "status": status,
             "std_option": std_option, "detail": detail}
        )
        if status == "flagged":
            self.flagged.append(
                {"card_no": card_no, "std_code": std_code, "detail": detail}
            )
        elif status == "unmatched":
            self.unmatched_cards.append({"card_no": card_no, "detail": detail})


class _CardEvaluator:
    """Turns one card's scanned rows into per-question verdicts."""

    def __init__(self, forms: list[dict], analysis: Analysis, decisions: dict):
        self.forms = forms
        self.analysis = analysis
        self.decisions = decisions

    def evaluate(self, card_no: str, rows: dict):
        """Yield (std_code, status, std_option, detail) for one card."""
        result = self.analysis
        compatible = [f for f in self.forms if form_is_compatible(f, rows)]
        if not compatible:
            detail = "无可相容卷式（整卡题号不一致或非空题码/涂选字母不吻合）"
            result.unmatched_count += 1
            yield "", "unmatched", None, detail
            return
        result.cards_matched += 1
        form_names = "/".join(f["name"] for f in compatible)

        for code in result.question_codes:
            stat = result.stats[code]
            decision = self.decisions.get((card_no, code))

            containing = [f for f in compatible if code in f["codes"]]
            if not containing:
                # No compatible form printed this question: skip entirely.
                continue

            answers = []
            for form in containing:
                q_no = form["codes"][code]
                letter = rows[q_no]["letter"]
                if letter:
                    std_option = form["qmap"][q_no]["letters"][letter]
                else:
                    std_option = ""
                answers.append((form["name"], letter, std_option))

            if len(containing) != len(compatible):
                missing = [f["name"] for f in compatible if code not in f["codes"]]
                status, option, detail = (
                    "flagged", None,
                    f"部分相容卷式未出该题：{'/'.join(missing)}",
                )
            else:
                distinct = {a[2] for a in answers}
                if len(distinct) == 1:
                    option = answers[0][2]
                    if option:
                        status, detail = "counted", f"相容卷式：{form_names}"
                    else:
                        status, detail = "blank", f"相容卷式：{form_names}（空答）"
                else:
                    shown = "；".join(
                        f"{name}: {letter or '空'}" for name, letter, _ in answers
                    )
                    status, option, detail = (
                        "flagged", None, f"卷式归一作答分歧：{shown}"
                    )

            if decision and status in ("flagged", "blank"):
                # A human conclusion is only meaningful where the automatic
                # verdict could not be trusted (disagreement / unprinted).
                # Once a correction makes the cell consistent, recalculated
                # statistics win and the stale conclusion stops applying.
                res = decision["resolution"]
                if res == "skip":
                    status, option, detail = "skipped", None, (
                        "人工核对：跳过（不计入）" +
                        (f"；{decision['note']}" if decision.get("note") else "")
                    )
                elif res == "blank":
                    status, option, detail = "blank", None, (
                        "人工核对：漏答" +
                        (f"；{decision['note']}" if decision.get("note") else "")
                    )
                elif res.startswith("opt:") and res[4:] in stat["options"]:
                    chosen = res[4:]
                    status, option, detail = "counted", chosen, (
                        "人工核对：" + chosen +
                        (f"；{decision['note']}" if decision.get("note") else "")
                    )

            if status == "skipped":
                continue
            if status == "counted":
                stat["counted"] += 1
                stat["options"][option] += 1
            elif status == "blank":
                stat["blank"] += 1
            else:
                stat["flagged"] += 1
                result.flagged_count += 1
            yield code, status, option, detail


def _group_card_rows(row_iter):
    """Yield (card_no, {q_no: {letter, scanned_code}}) one card at a time."""
    current = None
    rows: dict = {}
    for r in row_iter:
        card_no = r["card_no"]
        if card_no != current:
            if current is not None:
                yield current, rows
            current = card_no
            rows = {}
        rows[r["q_no"]] = {"letter": r["letter"], "scanned_code": r["scanned_code"]}
    if current is not None:
        yield current, rows


def analyse(forms, card_rows_by_card, decisions, sink=None) -> Analysis:
    """Compute a snapshot from in-memory card rows (used by tests/small data).

    ``sink`` defaults to the returned :class:`Analysis`; pass a
    :class:`_SnapshotSink` to stream instead of buffering per-card details.
    """
    analysis = _new_analysis(forms)

    def rows_iter():
        for card_no in sorted(card_rows_by_card):
            for q_no, row in card_rows_by_card[card_no].items():
                yield {"card_no": card_no, "q_no": q_no,
                       "letter": row["letter"], "scanned_code": row["scanned_code"]}

    _walk(analysis, forms, decisions, _group_card_rows(rows_iter()),
          sink if sink is not None else analysis)
    if sink is None:
        analysis.flagged.sort(
            key=lambda x: (_natural_key(x["card_no"]), _natural_key(x["std_code"]))
        )
    return analysis


def _new_analysis(forms, collect_details: bool = True) -> Analysis:
    analysis = Analysis(collect_details=collect_details)
    codes = set()
    options: dict[str, set[str]] = {}
    for form in forms.values():
        for code, letters in form["options"].items():
            codes.add(code)
            options.setdefault(code, set()).update(letters)
    analysis.question_codes = sorted(codes, key=_natural_key)
    analysis.options = {c: sorted(options[c]) for c in analysis.question_codes}
    for code in analysis.question_codes:
        analysis.stats[code] = {
            "counted": 0, "blank": 0, "flagged": 0,
            "options": {o: 0 for o in analysis.options[code]},
        }
    return analysis


def _walk(analysis: Analysis, forms, decisions, cards, sink) -> None:
    """Evaluate cards in order, updating counters and feeding ``sink``."""
    evaluator = _CardEvaluator(list(forms.values()), analysis, decisions)
    for card_no, rows in cards:
        analysis.cards_total += 1
        for verdict in evaluator.evaluate(card_no, rows):
            # Counters are updated by the evaluator while it yields.
            sink.add(card_no, *verdict)


# --------------------------------------------------------------------------
# DB loading + streaming persistence
# --------------------------------------------------------------------------

def load_forms(conn):
    rows = conn.execute(
        "SELECT f.id, f.name, m.q_no, m.letter, m.std_code, m.std_option "
        "FROM forms f JOIN form_mappings m ON m.form_id = f.id "
        "ORDER BY f.id, m.q_no, m.letter"
    ).fetchall()
    return build_forms(
        [(r["id"], r["name"], r["q_no"], r["letter"], r["std_code"], r["std_option"])
         for r in rows]
    )


def load_card_rows(conn):
    """Only used by small in-process callers; the app streams instead."""
    cards: dict[str, dict] = {}
    for r in conn.execute(
        "SELECT card_no, q_no, letter, scanned_code FROM answer_rows "
        "ORDER BY card_no, q_no"
    ):
        cards.setdefault(r["card_no"], {})[r["q_no"]] = {
            "letter": r["letter"], "scanned_code": r["scanned_code"],
        }
    return cards


def load_decisions(conn, card_nos=None):
    """Human conclusions, optionally only for a batch of card numbers."""
    sql = "SELECT card_no, std_code, resolution, note FROM review_decisions"
    params: list = []
    if card_nos is not None:
        card_nos = list(card_nos)
        if not card_nos:
            return {}
        marks = ",".join("?" * len(card_nos))
        sql += f" WHERE card_no IN ({marks})"
        params = card_nos
    return {
        (r["card_no"], r["std_code"]): {
            "resolution": r["resolution"], "note": r["note"],
        }
        for r in conn.execute(sql, params)
    }


class _SnapshotSink:
    """Writes streamed verdicts into the run tables, one bounded batch at a time.

    The ``runs`` header row is created first (inside the same transaction), so
    the child rows' foreign keys are satisfied while the per-card verdicts are
    flushed batch by batch.  Everything still commits or rolls back together
    with the caller's transaction.
    """

    def __init__(self, conn, run_id: int):
        self.conn = conn
        self.run_id = run_id
        self.cards: list[tuple] = []
        self.unmatched: list[tuple] = []

    def add(self, card_no, std_code, status, std_option, detail):
        if status == "unmatched":
            # The "" row keeps per-card detail queryable for unmatched cards.
            self.cards.append((self.run_id, card_no, "", status, std_option, detail))
            self.unmatched.append((self.run_id, card_no, detail))
        else:
            self.cards.append(
                (self.run_id, card_no, std_code, status, std_option, detail)
            )
        if len(self.cards) >= CARD_BATCH_SIZE:
            self.flush()

    def flush(self):
        if self.cards:
            self.conn.executemany(
                "INSERT INTO run_card_results"
                "(run_id, card_no, std_code, status, std_option, detail) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                self.cards,
            )
            self.cards.clear()
        if self.unmatched:
            self.conn.executemany(
                "INSERT INTO run_unmatched(run_id, card_no, detail) VALUES (?, ?, ?)",
                self.unmatched,
            )
            self.unmatched.clear()


def compute(conn) -> Analysis:
    """In-memory analysis; kept for tests.  The app uses :func:`compute_and_persist`."""
    return analyse(load_forms(conn), load_card_rows(conn), load_decisions(conn))


def _create_run(conn, analysis: Analysis, created_at: str) -> int:
    cur = conn.execute(
        "INSERT INTO runs(created_at, summary_json) VALUES (?, '{}')", (created_at,)
    )
    return cur.lastrowid


def compute_and_persist(conn, created_at: str) -> tuple[int, Analysis]:
    """Analyse every stored card and write one snapshot, in bounded memory.

    ``answer_rows`` is read in chunks of ``max(CARD_BATCH_SIZE, 1)`` distinct
    card numbers; only that many cards' rows and verdicts are alive at once.
    """
    forms = load_forms(conn)
    analysis = _new_analysis(forms, collect_details=False)
    form_list = list(forms.values())

    run_id = _create_run(conn, analysis, created_at)
    sink = _SnapshotSink(conn, run_id)

    cursor = conn.cursor()
    cursor.execute(
        "SELECT DISTINCT card_no FROM answer_rows ORDER BY card_no"
    )
    while True:
        cards = cursor.fetchmany(CARD_BATCH_SIZE)
        if not cards:
            break
        lo = cards[0][0]
        hi = cards[-1][0]
        card_nos = [r[0] for r in cards]
        # Decisions are loaded per batch so memory tracks the batch size, not
        # the number of accumulated review conclusions.
        evaluator = _CardEvaluator(
            form_list, analysis, load_decisions(conn, card_nos)
        )
        rows = conn.execute(
            "SELECT card_no, q_no, letter, scanned_code FROM answer_rows "
            "WHERE card_no >= ? AND card_no <= ? ORDER BY card_no, q_no",
            (lo, hi),
        )
        for card_no, card_rows in _group_card_rows(rows):
            analysis.cards_total += 1
            for code, status, option, detail in evaluator.evaluate(card_no, card_rows):
                sink.add(card_no, code, status, option, detail)
        sink.flush()
    cursor.close()

    conn.execute(
        "UPDATE runs SET summary_json = ? WHERE id = ?",
        (json.dumps(analysis.summary(), ensure_ascii=False), run_id),
    )
    conn.executemany(
        "INSERT INTO run_question_stats(run_id, std_code, counted, blank, flagged) "
        "VALUES (?, ?, ?, ?, ?)",
        [(run_id, code, st["counted"], st["blank"], st["flagged"])
         for code, st in analysis.stats.items()],
    )
    conn.executemany(
        "INSERT INTO run_option_counts(run_id, std_code, std_option, cnt) VALUES (?, ?, ?, ?)",
        [(run_id, code, opt, cnt)
         for code, st in analysis.stats.items() for opt, cnt in st["options"].items()],
    )
    return run_id, analysis


def prune_runs(conn, keep: int = 4) -> None:
    rows = conn.execute(
        "SELECT id FROM runs ORDER BY id DESC LIMIT -1 OFFSET ?", (keep,)
    ).fetchall()
    for r in rows:
        conn.execute("DELETE FROM runs WHERE id = ?", (r["id"],))
