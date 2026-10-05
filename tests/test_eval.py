"""Test phase 3: ROUGE tiếng Việt, chỉ số phụ, judge (parse/cache/sample), chốt API, BERTScore, report."""

import json
from pathlib import Path

import pytest

from vnsum.eval.config import EvalConfigError, load_eval_config, validate_eval_config
from vnsum.eval.judge import (
    JudgeCache,
    JudgeParseError,
    build_judge_prompt,
    judge_sample_ids,
    parse_judge_output,
    run_judge,
)
from vnsum.eval.llm_client import ApiGate, ApiNotAllowedError, LLMClient
from vnsum.eval.metrics import novel_ngram_rate, repetition_rate, syllables
from vnsum.eval.systems import PredictionCache, build_llm_prompt, lead_n

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs" / "eval.yaml"


# ------------------------------ ROUGE tiếng Việt ----------------------------- #


def test_rouge_vietnamese_tokenizer_distinguishes_diacritics():
    """BẮT BUỘC: "Hà Nội" và "Ha Noi" phải cho kết quả khác nhau."""
    pytest.importorskip("rouge_score")
    from vnsum.eval.metrics import rouge_per_sample

    same = rouge_per_sample(["Hà Nội"], ["Hà Nội"])[0]
    diff = rouge_per_sample(["Ha Noi"], ["Hà Nội"])[0]
    assert same["rouge1"] == 100.0
    assert diff["rouge1"] == 0.0
    assert same != diff


def test_default_rouge_tokenizer_destroys_vietnamese():
    """Minh họa lỗi: tokenizer mặc định coi "Hà Nội" và "Hè Nổi" là giống hệt nhau."""
    rouge_scorer = pytest.importorskip("rouge_score.rouge_scorer")
    from vnsum.eval.metrics import rouge_per_sample

    default = rouge_scorer.RougeScorer(["rouge1"])
    assert default.score("Hà Nội", "Hè Nổi")["rouge1"].fmeasure == 1.0  # lỗi của tokenizer mặc định
    assert rouge_per_sample(["Hè Nổi"], ["Hà Nội"])[0]["rouge1"] == 0.0  # tokenizer của mình phân biệt được


def test_syllable_tokenizer_keeps_diacritics_and_splits_punct():
    assert syllables("Hà Nội, ngày 5/10: MƯA to!") == ["hà", "nội", "ngày", "5", "10", "mưa", "to"]


# -------------------------------- chỉ số phụ --------------------------------- #


def test_repetition_rate():
    assert repetition_rate("a b c d e", 3) == 0.0
    # trigram: (a b c) (b c a) (c a b) (a b c) -> 1 bản lặp / 4
    assert repetition_rate("a b c a b c", 3) == pytest.approx(25.0)
    assert repetition_rate("ngắn", 3) == 0.0


def test_novel_ngram_rate():
    doc = "Hà Nội mưa to vào chiều nay"
    assert novel_ngram_rate("Hà Nội mưa to", doc, 1) == 0.0
    assert novel_ngram_rate("Hà Nội nắng to", doc, 1) == pytest.approx(25.0)
    assert novel_ngram_rate("Ha Noi", doc, 1) == 100.0  # mất dấu = n-gram mới


def test_lead_n():
    doc = "Câu một. Câu hai! Câu ba? Câu bốn."
    assert lead_n(doc, 3) == "Câu một. Câu hai! Câu ba?"
    assert lead_n("Chỉ một câu", 3) == "Chỉ một câu"


# ---------------------------------- config ----------------------------------- #


def test_repo_eval_config():
    cfg = load_eval_config(CFG_PATH)
    assert isinstance(cfg["bertscore"]["num_layers"], int)  # chỉ định rõ
    assert cfg["bertscore"]["max_length"] <= 256
    assert cfg["llm"]["judge"]["temperature"] == 0
    assert cfg["judge"]["sample_size"] == 200
    assert cfg["generation"]["dtype"] != "fp16"
    assert "api_key" not in json.dumps(cfg).replace("api_key_env", "")  # không có khóa trong config


def test_config_rejects_fp16_and_bad_bertscore_length():
    cfg = load_eval_config(CFG_PATH)
    cfg["generation"]["dtype"] = "fp16"
    with pytest.raises(EvalConfigError, match="fp16"):
        validate_eval_config(cfg)
    cfg = load_eval_config(CFG_PATH)
    cfg["bertscore"]["max_length"] = 512
    with pytest.raises(EvalConfigError, match="256"):
        validate_eval_config(cfg)


# ---------------------------------- judge ------------------------------------ #


def test_parse_judge_output_variants():
    ok = parse_judge_output('{"score": 4, "unsupported_claims": ["x"], "rationale": "A. B. C."}')
    assert ok == {"score": 4, "unsupported_claims": ["x"], "rationale": "A. B."}  # cắt còn 2 câu
    fenced = parse_judge_output('```json\n{"score": 5.0, "unsupported_claims": [], "rationale": "Ổn."}\n```')
    assert fenced["score"] == 5
    for bad in ("không có json", '{"score": 7, "unsupported_claims": []}', '{"score": "5"}',
                '{"score": 3, "unsupported_claims": "x"}'):
        with pytest.raises(JudgeParseError):
            parse_judge_output(bad)


def test_judge_prompt_has_rubric_and_no_cot_request():
    prompt, truncated = build_judge_prompt("Văn bản gốc.", "Tóm tắt.", max_chars=100)
    assert "5 —" in prompt and "1 —" in prompt and '"score"' in prompt
    assert not truncated
    assert "từng bước" not in prompt.lower() and "step by step" not in prompt.lower()
    _, truncated = build_judge_prompt("x" * 200, "y", max_chars=100)
    assert truncated


def test_judge_sample_is_deterministic_and_order_free():
    ids = [f"id{i}" for i in range(1000)]
    a = judge_sample_ids(ids, 200, seed=42)
    assert a == judge_sample_ids(list(reversed(ids)), 200, seed=42)
    assert len(a) == 200 and len(set(a)) == 200
    assert a != judge_sample_ids(ids, 200, seed=7)
    assert len(judge_sample_ids(ids[:50], 200, seed=42)) == 50
    assert set(judge_sample_ids(ids, 200, seed=42)) <= set(judge_sample_ids(ids, 500, seed=42))  # lồng nhau


class _FakeClient:
    """Thay LLMClient: đếm số lần gọi, không gửi mạng."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0

    def complete(self, user, system=None):
        self.calls += 1
        return self.reply, {}

    def map(self, items, fn):
        return [fn(x) for x in items]


def _judge_cfg():
    cfg = load_eval_config(CFG_PATH)
    return cfg


def test_run_judge_uses_cache_and_respects_api_gate(tmp_path):
    cfg = _judge_cfg()
    items = [{"id": "a", "document": "Doc a.", "prediction": "Pred a."},
             {"id": "b", "document": "Doc b.", "prediction": "Pred b."}]
    cache = JudgeCache(tmp_path / "judge.jsonl")

    # Không có --allow_api: không gọi, trả None, ghi nhận kế hoạch.
    gate = ApiGate(allowed=False)
    fake = _FakeClient('{"score": 5, "unsupported_claims": [], "rationale": "Đúng."}')
    assert run_judge("sys", items, cfg, cache=cache, gate=gate, client=fake) is None
    assert fake.calls == 0 and gate.planned == {"judge:sys": 2}

    # Có --allow_api: gọi 2 lần, ghi cache.
    gate = ApiGate(allowed=True)
    out = run_judge("sys", items, cfg, cache=cache, gate=gate, client=fake)
    assert fake.calls == 2 and [r["score"] for r in out] == [5, 5]

    # Chạy lại (kể cả cache đọc từ file): không gọi thêm.
    reloaded = JudgeCache(tmp_path / "judge.jsonl")
    out2 = run_judge("sys", items, cfg, cache=reloaded, gate=ApiGate(allowed=False), client=fake)
    assert fake.calls == 2 and [r["id"] for r in out2] == ["a", "b"]

    # Prediction đổi -> key đổi -> cần chấm lại.
    changed = [dict(items[0], prediction="Khác.")]
    assert run_judge("sys", changed, cfg, cache=reloaded, gate=ApiGate(allowed=False), client=fake) is None


def test_run_judge_records_parse_errors(tmp_path):
    cfg = _judge_cfg()
    fake = _FakeClient("xin lỗi, tôi không chấm được")
    out = run_judge("sys", [{"id": "a", "document": "d", "prediction": "p"}], cfg,
                    cache=JudgeCache(tmp_path / "j.jsonl"), gate=ApiGate(allowed=True), client=fake)
    assert out[0]["score"] is None and out[0]["error"]


def test_llm_client_blocked_without_allow_api():
    cfg = load_eval_config(CFG_PATH)
    client = LLMClient(cfg["llm"]["summarizer"], label="x", gate=ApiGate(allowed=False), session=object())
    with pytest.raises(ApiNotAllowedError):
        client.complete("hi")


def test_llm_prompt_inserts_document_without_format_errors():
    tpl = "Tóm tắt:\n{document}"
    assert build_llm_prompt(tpl, "a {b} c", max_chars=100) == "Tóm tắt:\na {b} c"
    assert build_llm_prompt(tpl, "abcdef", max_chars=3) == "Tóm tắt:\nabc"


# ------------------------------ cache dự đoán -------------------------------- #


def test_prediction_cache_reuses_only_same_fingerprint(tmp_path):
    c1 = PredictionCache(tmp_path, "sys", fp="v1")
    c1.add({"a": "x", "b": "y"})
    assert PredictionCache(tmp_path, "sys", fp="v1").missing(["a", "b", "c"]) == ["c"]
    assert PredictionCache(tmp_path, "sys", fp="v2").missing(["a", "b"]) == ["a", "b"]


# ------------------------------- BERTScore ------------------------------------ #


def test_greedy_match_matches_manual_computation():
    torch = pytest.importorskip("torch")
    from vnsum.eval.bertscore import greedy_match

    hyp = torch.nn.functional.normalize(torch.tensor([[[1.0, 0.0], [0.0, 1.0], [9.0, 9.0]]]), dim=-1)
    ref = torch.nn.functional.normalize(torch.tensor([[[1.0, 0.0], [1.0, 1.0]]]), dim=-1)
    hyp_mask = torch.tensor([[1, 1, 0]])  # token thứ 3 là đặc biệt/padding -> bỏ
    ref_mask = torch.tensor([[1, 1]])
    p, r, f = greedy_match(hyp, hyp_mask, ref, ref_mask)
    s = 2**-0.5
    exp_p = (1.0 + s) / 2  # hyp[0]->ref[0]=1, hyp[1]->ref[1]=0.707
    exp_r = (1.0 + s) / 2  # ref[0]->hyp[0]=1, ref[1]->max(0.707,0.707)
    assert p.item() == pytest.approx(exp_p, abs=1e-6)
    assert r.item() == pytest.approx(exp_r, abs=1e-6)
    assert f.item() == pytest.approx(2 * exp_p * exp_r / (exp_p + exp_r), abs=1e-6)
    empty_p, _, empty_f = greedy_match(hyp, torch.zeros_like(hyp_mask), ref, ref_mask)
    assert empty_p.item() == 0.0 and empty_f.item() == 0.0


# --------------------------------- report ------------------------------------ #


def test_aggregate_and_report_use_only_given_numbers(tmp_path):
    pd = pytest.importorskip("pandas")
    from vnsum.eval.report import aggregate, write_qualitative, write_report

    cfg = load_eval_config(CFG_PATH)
    per_sample = pd.DataFrame({
        "system": ["s1", "s1", "s2", "s2"], "id": ["a", "b", "a", "b"],
        "dataset": ["vietnews", "wikilingua_vi"] * 2, "is_long": [True, False] * 2,
        "summary": ["r"] * 4, "prediction": ["p"] * 4,
        "rouge1": [10.0, 20.0, 30.0, 50.0], "rouge2": [1.0, 2.0, 3.0, 4.0], "rougeL": [5.0, 6.0, 7.0, 8.0],
        "length_syllables": [10.0, 20.0, 30.0, 40.0], "repetition_3gram": [0.0] * 4,
        "novel_1gram": [1.0] * 4, "novel_2gram": [2.0] * 4,
        "faithfulness": [5, None, 2, 4], "unsupported_claims": [[], None, ["sai số liệu"], []],
        "judge_rationale": ["ok", None, "sai", "ok"], "judge_error": [None] * 4,
    })
    metrics = aggregate(per_sample)
    s1 = metrics[(metrics.system == "s1") & (metrics.subset == "all")].iloc[0]
    assert s1["n"] == 2 and s1["rouge1"] == pytest.approx(15.0)
    assert s1["faithfulness_n"] == 1 and s1["faithfulness_mean"] == 5.0
    s2_long = metrics[(metrics.system == "s2") & (metrics.subset == "long")].iloc[0]
    assert s2_long["n"] == 1 and s2_long["rouge1"] == 30.0

    out = tmp_path / "report.md"
    write_report(per_sample, metrics, cfg, out, system_info={"s1": "a", "s2": "b"}, skipped={"s3": "cần --allow_api"},
                 notes=[], reference_aux={"length_syllables": 33.0, "repetition_3gram": 0.0, "novel_1gram": 9.0, "novel_2gram": 19.0})
    text = out.read_text(encoding="utf-8")
    assert "| `s1` | 2 | 15.00 | 1.50 | 5.50 |" in text
    assert "| `s2` | 2 | 40.00 | 3.50 | 7.50 |" in text
    assert "RAW" in text and "`s3` (cần --allow_api)" in text

    eval_df = pd.DataFrame({"id": ["a", "b"], "document": ["Doc A dài.", "Doc B."], "summary": ["r", "r"]})
    cfg["qualitative"]["system"] = "s2"
    q = tmp_path / "q.md"
    write_qualitative(per_sample, eval_df, cfg, q)
    qtext = q.read_text(encoding="utf-8")
    assert "faithfulness 4/5" in qtext and "faithfulness 2/5" in qtext and "sai số liệu" in qtext


def test_display_name_changes_report_but_not_prediction_cache(tmp_path):
    pd = pytest.importorskip("pandas")
    from vnsum.eval.report import aggregate, display_name, write_report
    from vnsum.eval.systems import system_fingerprint

    cfg = load_eval_config(CFG_PATH)
    sys_cfg = dict(cfg["systems"]["vit5_lora"], display_name=None)
    fp_before = system_fingerprint("vit5_lora", sys_cfg, cfg, "m")
    renamed = dict(sys_cfg, display_name="VnLongSum", description="khác")
    assert system_fingerprint("vit5_lora", renamed, cfg, "m") == fp_before  # đổi tên không mất cache

    cfg["systems"]["vit5_lora"]["display_name"] = "VnLongSum"
    assert display_name(cfg, "vit5_lora") == "VnLongSum"
    assert display_name(cfg, "lead3") == "lead3"
    per_sample = pd.DataFrame({
        "system": ["vit5_lora"], "id": ["a"], "dataset": ["vietnews"], "is_long": [False],
        "summary": ["r"], "prediction": ["p"], "rouge1": [10.0], "rouge2": [1.0], "rougeL": [5.0],
        "length_syllables": [3.0], "faithfulness": [None],
    })
    metrics = aggregate(per_sample, cfg)
    assert metrics.iloc[0]["display_name"] == "VnLongSum"
    out = tmp_path / "r.md"
    write_report(per_sample, metrics, cfg, out, system_info={"vit5_lora": "x"}, skipped={}, notes=[])
    assert "| **VnLongSum** | 1 |" in out.read_text(encoding="utf-8")


class _FakeResp:
    def __init__(self, status, payload):
        self.status_code, self._payload, self.text = status, payload, str(payload)

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self):
        self.posts, self.gets = [], []

    def post(self, url, headers=None, json=None, timeout=None):
        self.posts.append((url, json))
        return _FakeResp(200, {"choices": [{"message": {"content": " Tóm tắt. "}}], "usage": {"total_tokens": 5}})

    def get(self, url, headers=None, timeout=None):
        self.gets.append(url)
        return _FakeResp(200, {"data": [{"id": "gpt-6-luna"}, {"id": "gpt-6.1-sol"}]})


def test_openai_request_body_fields():
    monkeypatch_env = {"OPENAI_API_KEY": "test"}
    import os

    for k, v in monkeypatch_env.items():
        os.environ[k] = v
    cfg = load_eval_config(CFG_PATH)
    ccfg = dict(cfg["llm"]["summarizer"])
    sess = _FakeSession()
    client = LLMClient(ccfg, label="s", gate=ApiGate(allowed=True), session=sess)
    text, _ = client.complete("hi")
    url, body = sess.posts[0]
    assert text == "Tóm tắt."
    assert url == "https://api.openai.com/v1/chat/completions"
    assert body["max_completion_tokens"] == ccfg["max_tokens"] and "max_tokens" not in body
    assert body["reasoning_effort"] == "none" and body["temperature"] == 0.0

    ccfg["temperature"] = None  # model không nhận temperature -> không gửi
    LLMClient(ccfg, label="s", gate=ApiGate(allowed=True), session=sess).complete("hi")
    assert "temperature" not in sess.posts[1][1]
    assert client.list_models() == ["gpt-6-luna", "gpt-6.1-sol"]
    assert sess.gets[0] == "https://api.openai.com/v1/models"


# ----------------------------------------------------------------- bootstrap CI / so sánh cặp
def test_bootstrap_ci_contains_mean_and_is_deterministic():
    np = pytest.importorskip("numpy")
    from vnsum.eval.stats import bootstrap_mean_ci

    x = np.random.default_rng(0).normal(30, 10, size=500)
    mean, lo, hi = bootstrap_mean_ci(x, 1000, 0.95, 42)
    assert lo < mean < hi and mean == pytest.approx(x.mean())
    assert bootstrap_mean_ci(x, 1000, 0.95, 42) == (mean, lo, hi)


def test_paired_bootstrap_detects_shift_and_not_noise():
    np = pytest.importorskip("numpy")
    from vnsum.eval.stats import paired_bootstrap

    rng = np.random.default_rng(1)
    b = rng.normal(20, 8, size=400)
    same = paired_bootstrap(b, b, 1000, 0.95, 42)
    assert same["diff"] == 0 and same["p_value"] == 1.0
    shifted = paired_bootstrap(b + 2 + rng.normal(0, 1, size=400), b, 1000, 0.95, 42)
    assert shifted["ci_low"] > 0 and shifted["p_value"] < 0.01
    with pytest.raises(ValueError):
        paired_bootstrap([1, 2], [1], 10, 0.95, 0)


def test_compute_significance_pairs_on_common_samples_and_report(tmp_path):
    pd = pytest.importorskip("pandas")
    from vnsum.eval.report import aggregate, write_report
    from vnsum.eval.stats import compute_significance

    cfg = load_eval_config(CFG_PATH)
    ids = [str(i) for i in range(60)]
    rows = []
    for system, base in (("vit5_lora", 30.0), ("lead3", 20.0)):
        for k, i in enumerate(ids):
            rows.append({"system": system, "id": i, "dataset": "vietnews", "is_long": False, "summary": "r",
                         "prediction": "p", "rouge1": base + k % 5, "rouge2": 1.0, "rougeL": 1.0,
                         "length_syllables": 30.0, "repetition_3gram": 0.0, "novel_1gram": 0.0, "novel_2gram": 0.0,
                         "faithfulness": (4 if system == "vit5_lora" else 5) if k < 20 else None})
    per_sample = pd.DataFrame(rows)
    ci_df, paired_df = compute_significance(per_sample, cfg)
    r1 = paired_df[(paired_df.system == "lead3") & (paired_df.metric == "rouge1")].iloc[0]
    assert r1["n"] == 60 and r1["diff"] == pytest.approx(10.0) and r1["ci_low"] > 0
    f = paired_df[(paired_df.system == "lead3") & (paired_df.metric == "faithfulness")].iloc[0]
    assert f["n"] == 20 and f["diff"] == pytest.approx(-1.0)
    assert set(ci_df[ci_df.metric == "faithfulness"]["n"]) == {20}

    out = tmp_path / "report.md"
    write_report(per_sample, aggregate(per_sample), cfg, out, system_info={}, skipped={}, notes=[], reference_aux=None,
                 significance=(ci_df, paired_df))
    text = out.read_text(encoding="utf-8")
    assert "Khoảng tin cậy 95%" in text and "So sánh cặp" in text and "có ý nghĩa" in text


# ----------------------------------------------------------------- human eval
def _human_per_sample(pd):
    rows = []
    for system in ("lead3", "vit5_lora", "vit5_public", "llm_zeroshot"):
        for k in range(30):
            rows.append({"system": system, "id": f"d{k}", "dataset": "vietnews", "prediction": f"{system} tóm tắt {k}",
                         "faithfulness": (k % 5) + 1 if k < 20 else None, "judge_rationale": f"lý do {k}"})
    return pd.DataFrame(rows)


def test_human_eval_select_is_blind_balanced_and_deterministic(tmp_path):
    pd = pytest.importorskip("pandas")
    from vnsum.eval.human_eval import export_items, select_items

    ps = _human_per_sample(pd)
    systems = ["vit5_lora", "vit5_public", "llm_zeroshot"]
    items = select_items(ps, systems, 50, 42)
    assert len(items) == 50 and items["item_id"].is_unique
    assert items["system"].value_counts().to_dict() == {"vit5_lora": 17, "vit5_public": 17, "llm_zeroshot": 16}
    assert "lead3" not in set(items["system"])
    assert set(items["id"].str[1:].astype(int)) <= set(range(20))     # chỉ mẫu đã được judge chấm
    assert items.equals(select_items(ps.sample(frac=1, random_state=3), systems, 50, 42))
    with pytest.raises(ValueError):
        select_items(ps, systems, 90, 42)                                  # mỗi hệ thống chỉ 20 mẫu đã chấm

    docs = pd.Series({f"d{k}": f"Văn bản gốc số {k}.\nĐoạn hai." for k in range(30)})
    paths = export_items(items, docs, tmp_path, 10)
    md = paths["items"].read_text(encoding="utf-8")
    assert "H001" in md and "Thang điểm" in md and "cắt còn 10" in md
    for name in ("vit5_lora", "vit5_public", "llm_zeroshot", "lý do"):
        assert name not in md.replace("vit5_lora tóm tắt", "").replace("vit5_public tóm tắt", "").replace("llm_zeroshot tóm tắt", "")
    sheet = pd.read_csv(paths["sheet"], encoding="utf-8-sig")
    assert list(sheet.columns) == ["item_id", "human_score", "human_notes"] and sheet["human_score"].isna().all()
    with pytest.raises(FileExistsError):
        export_items(items, docs, tmp_path, 10)                            # không ghi đè điểm đã chấm


def test_quadratic_weighted_kappa_and_spearman():
    np = pytest.importorskip("numpy")
    from vnsum.eval.human_eval import quadratic_weighted_kappa, spearman

    a = np.array([1, 2, 3, 4, 5, 5, 4, 3])
    assert quadratic_weighted_kappa(a, a) == pytest.approx(1.0)
    assert spearman(a, a) == pytest.approx(1.0)
    # Tính tay: obs = [[0,1],[1,0]] trên nhãn {1,2} -> kappa = 1 - 2/1 = -1
    assert quadratic_weighted_kappa(np.array([1, 2]), np.array([2, 1]), labels=(1, 2)) == pytest.approx(-1.0)
    assert np.isnan(spearman(np.array([3, 3, 3]), np.array([1, 2, 3])))
    # Lệch 1 bậc bị phạt nhẹ hơn lệch 4 bậc
    near = quadratic_weighted_kappa(np.array([1, 2, 3, 4, 5]), np.array([2, 2, 3, 4, 5]))
    far = quadratic_weighted_kappa(np.array([1, 2, 3, 4, 5]), np.array([5, 2, 3, 4, 5]))
    assert near > far


def test_human_eval_analyze_reads_semicolon_sheet_and_skips_blank(tmp_path):
    pd = pytest.importorskip("pandas")
    from vnsum.eval.human_eval import analyze, read_sheet

    key = pd.DataFrame({"item_id": ["H001", "H002", "H003", "H004"], "system": ["a", "a", "b", "b"],
                        "id": ["1", "2", "3", "4"], "dataset": ["vietnews"] * 4,
                        "judge_score": [5, 4, 2, 1], "judge_rationale": ["ok", "ok", "sai tên", "bịa"]})
    p = tmp_path / "sheet.csv"
    p.write_text("item_id;human_score;human_notes\nH001;5;\nH002;4;\nH003;5;judge quá khắt khe\nH004;;\n", encoding="utf-8-sig")
    sheet = read_sheet(p)
    text, merged = analyze(sheet, key)
    assert "Đã chấm 3/4 mục" in text
    assert "| Trùng khớp tuyệt đối | 66.7% |" in text
    assert "H003" in text and "judge quá khắt khe" in text and "## Các mục lệch từ 2 điểm trở lên (1)" in text
    bad = tmp_path / "bad.csv"
    bad.write_text("item_id,human_score\nH001,7\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_sheet(bad)


def test_run_judge_retries_parse_errors_only_when_asked(tmp_path):
    cfg = _judge_cfg()
    cache = JudgeCache(tmp_path / "c.jsonl")
    items = [{"id": "a", "document": "Văn bản.", "prediction": "Tóm tắt."}]

    class Bad:
        def complete(self, prompt, system=None):
            return 'Tôi không chắc, khoảng 5 điểm.', {}
        def map(self, todo, fn):
            return [fn(x) for x in todo]

    class Good(Bad):
        def complete(self, prompt, system=None):
            return '{"score": 4, "unsupported_claims": [], "rationale": "ok"}', {}

    assert run_judge("s", items, cfg, cache=cache, gate=ApiGate(True), client=Bad())[0]["error"]
    gate = ApiGate(False)                                   # lỗi đã cache: không chấm lại, không cần API
    assert run_judge("s", items, cfg, cache=cache, gate=gate)[0]["score"] is None and not gate.planned
    dry = ApiGate(False)                                    # chưa cho phép API: chỉ ước tính, vẫn trả kết quả cũ
    assert run_judge("s", items, cfg, cache=cache, gate=dry, retry_errors=True)[0]["error"] and dry.planned
    out = run_judge("s", items, cfg, cache=cache, gate=ApiGate(True), client=Good(), retry_errors=True)
    assert out[0]["score"] == 4 and out[0]["error"] is None
    assert JudgeCache(tmp_path / "c.jsonl").get(next(iter(cache.data)))["score"] == 4   # dòng mới thắng khi nạp lại


def test_recover_judge_output_from_unescaped_quotes_and_truncation():
    from vnsum.eval.judge import recover_judge_output, resolve_cached

    bad = ('{"score": 4, "unsupported_claims": ["Sau hơn 3 giờ hạ cánh" — văn bản gốc nói \'sau hơn 3 giờ khắc phục\'], '
           '"rationale": "Chi tiết thời gian bị diễn đạt sai. Ý chính vẫn đúng."}')
    with pytest.raises(JudgeParseError):
        parse_judge_output(bad)
    rec = recover_judge_output(bad)
    assert rec["score"] == 4 and rec["recovered"] and "3 giờ hạ cánh" in rec["unsupported_claims"][0]
    assert rec["rationale"].startswith("Chi tiết thời gian")
    cut = '{"score": 1, "unsupported_claims": ["Võ Văn Hoan được gọi là Phó giám đốc (thực ra'
    assert recover_judge_output(cut)["score"] == 1
    for unsure in ('{"unsupported_claims": [], "score": 3', '{"score": 7, "x": 1', "điểm 4", '{"score": "4"'):
        assert recover_judge_output(unsure) is None          # không đoán khi score không rõ ràng
    row = {"key": "k", "raw": bad, "score": None, "error": "JSON lỗi"}
    fixed = resolve_cached(row)
    assert fixed["score"] == 4 and fixed["error"] is None and fixed["original_error"] == "JSON lỗi"
    assert row["score"] is None                               # không sửa dòng cache gốc
