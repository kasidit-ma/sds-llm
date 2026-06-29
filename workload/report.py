"""Generate workload_report.docx: overview + method + datasets + 3 figures + summary tables."""
import json
from pathlib import Path

from docx import Document
from docx.shared import Inches

import speedup

HERE = Path(__file__).parent
RESULTS = HERE / "results"


def _add_table(doc, header, rows_data):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Light Grid Accent 1"
    for c, h in zip(t.rows[0].cells, header):
        c.text = h
    for rd in rows_data:
        cells = t.add_row().cells
        for c, v in zip(cells, rd):
            c.text = str(v)
    return t


def _week6_speedup(doc, rows):
    bench_path = HERE / "bench_results.json"
    if not bench_path.exists():
        return
    bench = json.loads(bench_path.read_text())
    pred = speedup.predict(rows, bench)

    doc.add_heading("6. week6 — Speedup & curve fitting", 1)

    doc.add_paragraph(
        "Flow ของ speedup (STAGE A): corpus → tokenize (Qwen2.5) → วัด feature 2 ตัว "
        "(Heaps' β และ PBE P90@1/2/4) พร้อมรัน n-gram spec-decode แบบ simulation "
        "(actual_speedup = baseline_steps / spec_steps = token เฉลี่ยที่ accept ต่อรอบ "
        "verifier). simulation ตั้ง drafter cost = 0 และ accept อย่างน้อย 1 token/รอบ "
        "→ ค่า >1 เสมอเชิงกลไก จึงเชื่อ *การเรียงลำดับข้าม dataset* ไม่ใช่ค่าสัมบูรณ์.")
    doc.add_paragraph(
        "Flow ของ curve fitting (STAGE B): เอา feature + actual_speedup ทั้ง 12 dataset มา "
        "fit linear least squares ได้  pred_speedup = w0 + w1·β + w2·P90@1 + w3·P90@2 + "
        "w4·P90@4. feature เข้าแบบแบนๆ (ไม่ใช่ product chain) จึงไม่ต้องพึ่ง depth ที่ไม่ได้วัด "
        "(เช่น @3). ปลายทางที่นำไปใช้: workload ใหม่แค่วัด feature (ถูก) แล้วทำนาย speedup "
        "ได้เลยโดยไม่ต้องรัน benchmark เต็ม.")

    fp = HERE / "speedup_flow.png"
    if fp.exists():
        doc.add_picture(str(fp), width=Inches(6.3))
        cap = doc.add_paragraph(
            "Flow 2 stage: (A) วัด feature + simulation ได้ actual_speedup, "
            "(B) fit regression แล้วนำไปทำนาย workload ใหม่โดยไม่ต้อง benchmark.")
        cap.style = "Caption"

    doc.add_paragraph("ผลทำนายเทียบค่าจริง (residual = pred − actual):")
    _add_table(
        doc, ["dataset", "family", "β", "pred", "actual", "residual"],
        [(r["dataset_id"], r["family"], f"{r['heaps']['beta']:.3f}",
          f"{pred[r['dataset_id']]:.2f}",
          f"{bench[r['dataset_id']]['actual_speedup']:.2f}",
          f"{pred[r['dataset_id']] - bench[r['dataset_id']]['actual_speedup']:+.2f}")
         for r in rows if r["dataset_id"] in bench])

    doc.add_paragraph("ผลการ fit (nested models, linear least squares, in-sample):")
    fit_p = doc.add_paragraph(speedup.fit_actual(rows, bench))
    fit_p.style = "No Spacing"

    fp2 = HERE / "speedup_plot.png"
    if fp2.exists():
        doc.add_picture(str(fp2), width=Inches(6.3))
        cap = doc.add_paragraph(
            "(1) β vs actual + เส้น fit, (2) predicted vs actual (เส้นทแยง = ทำนายเป๊ะ), "
            "(3) bar chart ต่อ dataset เรียงตาม actual_speedup.")
        cap.style = "Caption"

    for b in [
        "β เป็น feature ที่อธิบาย speedup ได้มากสุด — โมเดล β เดี่ยวก็ R² สูงแล้ว.",
        "dialogue/nl (ซ้ำเยอะ เดาง่าย) speedup สูงสุด ~3.8–4.0; math (ยาก) ต่ำสุด ~2.2.",
        "ขั้นถัดไป: รัน wall-clock จริง (มี drafter cost) ทับ simulation แล้ว fit ใหม่ — "
        "โค้ดไม่ต้องแก้ แค่ bench_results.json เปลี่ยน.",
    ]:
        doc.add_paragraph(b, style="List Bullet")


def build(out_name="workload_report.docx"):
    rows = [json.loads(f.read_text()) for f in sorted(RESULTS.glob("*.json"))]
    rows.sort(key=lambda r: (r["family"], r["dataset_id"]))
    doc = Document()

    doc.add_heading("Workload Characterization — SDS-LLM", 0)
    doc.add_paragraph(
        "วัดคุณลักษณะของ 12 datasets ว่าเอื้อต่อ n-gram speculative decoding แค่ไหน "
        "ผ่าน 2 ตัวชี้วัด: Heaps' β (ความหลากหลายของ vocab) และ Prefix Branching "
        f"Entropy (PBE; ความไม่แน่นอนของ token ถัดไป). Tokenizer = {rows[0]['tokenizer']}.")

    doc.add_heading("1. วิธีวัด", 1)
    doc.add_paragraph(
        "Heaps' β: ไล่อ่าน corpus แล้ว fit V = K·N^β บนสเกล log-log (V = token ไม่ซ้ำ, "
        "N = token ที่อ่าน). β สูง = เจอคำใหม่เรื่อยๆ vocab หลากหลาย.")
    doc.add_paragraph(
        "PBE: สำหรับ prefix (context) ยาว 3 tokens นับว่ามี token ตามมาได้กี่แบบ (branching). "
        "branching=1 = เดาถูกแน่ → drafter ทำงานได้ดี. @1/@2/@4 = ดู token ที่ตำแหน่ง +1/+2/+4 "
        "(ยิ่งไกลยิ่งไม่แน่นอน). รายงานทั้ง P90 (เคสแย่ปกติ robust ต่อ outlier) และ mean "
        "(ค่าเฉลี่ย ไวต่อ tail ของ context เดายากสุดโต่ง).")

    doc.add_heading("2. Datasets (12 ตัว)", 1)
    doc.add_paragraph(
        "ครอบคลุม 4 family: code, natural language (nl), math, dialogue. มีการสลับ 3 ตัวจาก spec "
        "เดิม — the-stack (gated) → python_code_instructions, code_contests (stream ช้าเกิน) → "
        "MetaMathQA, SWE-chat (tree) → alpaca — และแก้ hf_path ของ MATH/samsum เป็น parquet mirror.")
    _add_table(doc, ["dataset", "family", "tokens"],
               [(r["dataset_id"], r["family"], f"{r['token_count']:,}") for r in rows])

    doc.add_heading("3. ผลลัพธ์ (กราฟ)", 1)
    figs = [
        ("workload_character.png", "Scatter: Heaps' β (Y) × PBE mean (X) แยก 3 panel @1/@2/@4. "
         "มุมซ้ายบน = เหมาะกับ spec decoding (vocab หลากหลายแต่เดา local ง่าย)."),
        ("pbe_ecdf.png", "ECDF ของ branching count ต่อ context. เส้นสูง/ชิดซ้าย = เดาง่าย, "
         "เส้นประ y=0.9 = P90."),
        ("heaps_curve.png", "เส้นโค้ง vocab growth (V vs N, log-log). ความชัน = β."),
    ]
    for fname, cap in figs:
        fp = HERE / fname
        if fp.exists():
            doc.add_picture(str(fp), width=Inches(6.3))
            c = doc.add_paragraph(cap)
            c.style = "Caption"

    doc.add_heading("4. ตารางสรุป", 1)
    doc.add_paragraph("PBE P90 (+ β):")
    _add_table(doc, ["dataset", "family", "β", "P90@1", "P90@2", "P90@4"],
               [(r["dataset_id"], r["family"], f"{r['heaps']['beta']:.3f}",
                 *[f"{r['pbe'][f'pbe@{n}']['p90']:.0f}" for n in (1, 2, 4)]) for r in rows])
    doc.add_paragraph("PBE mean:")
    _add_table(doc, ["dataset", "family", "mean@1", "mean@2", "mean@4"],
               [(r["dataset_id"], r["family"],
                 *[f"{r['pbe'][f'pbe@{n}']['mean']:.2f}" for n in (1, 2, 4)]) for r in rows])

    doc.add_heading("5. ข้อสังเกตหลัก", 1)
    for b in [
        "dialogue (dolly, alpaca) เดาง่ายสุด — P90@1=1, mean ขยับน้อยแม้ draft ไกล.",
        "math (MATH, MetaMathQA) ยากสุด — P90/mean พุ่งขึ้นชัดเมื่อ @1→@4 (tail หนา).",
        "code เสถียร P90=2 ทุก step; mean จับความต่างย่อยที่ P90 มองไม่เห็น.",
        "β กับ PBE สวนทางกัน (β สูง ↔ PBE ต่ำ): vocab หลากหลายแต่ local structure เดาง่าย.",
        "HumanEval/MBPP มี token น้อย (~20–30k) → β อาจไม่นิ่งเท่าตัวอื่น.",
    ]:
        doc.add_paragraph(b, style="List Bullet")

    _week6_speedup(doc, rows)

    doc.save(HERE / out_name)
    print(f"saved {out_name}")


if __name__ == "__main__":
    build()
