# Workload Characterization — บันทึกวิเคราะห์

วัด 12 datasets (4 family) ด้วย tokenizer `Qwen/Qwen2.5-7B` เพื่อดูว่าตัวไหนเอื้อต่อ n-gram
speculative decoding ตัวชี้วัด: **Heaps' β** (ความหลากหลาย vocab) และ **PBE** (branching count
ต่อ context ยาว 3 tokens — ยิ่งต่ำ token ถัดไปยิ่งเดาง่าย)

## ข้อค้นพบหลัก

**1. β กับ PBE สวนทางกันแรงมาก** — corr(β, PBE mean@1) = **−0.92**, corr(β, การโต @1→@4) = **−0.88**

แปลว่า dataset ที่ vocab หลากหลาย (β สูง) กลับมี local structure ที่เดาง่าย (PBE ต่ำ) อย่างเป็น
ระบบ ไม่ใช่เรื่องบังเอิญ → ในทางปฏิบัติ **β ตัวเดียวพอเดา PBE ได้คร่าวๆ** สองแกนนี้ไม่ได้ให้
ข้อมูลอิสระต่อกันเท่าที่คิดตอนออกแบบ (ดู scatter ที่จุดเรียงเป็นแนวทแยง)

**2. จัดอันดับความเหมาะกับ spec decoding** (เดาง่าย → ยาก, ใช้ mean@1 + สัดส่วน context ที่ branching=1):

| อันดับ | dataset | β | mean@1 | % context เดาตรง (br=1) |
|---|---|---|---|---|
| เดาง่ายสุด | dialogue_dolly | 0.768 | 1.16 | 92.5% |
| | dialogue_alpaca | 0.756 | 1.20 | 90.2% |
| | nl_wikitext103 | 0.746 | 1.21 | 91.2% |
| กลางๆ | code (humaneval/mbpp/python) | ~0.69 | 1.27–1.36 | ~85–87% |
| | nl (openwebtext/cnndailymail) | ~0.68 | 1.30–1.35 | ~87–89% |
| ยากสุด | math_metamath | 0.603 | 1.68 | 76.5% |
| | math_math | 0.576 | 1.88 | 73.8% |

> "% context เดาตรง" = สัดส่วน context ที่ branching=1 → คือ **เพดาน acceptance** ของ n-gram drafter
> แบบ greedy ที่ @1 (ถ้า context แตกได้ทางเดียว drafter เดาถูกแน่)

**3. draft ยาวยิ่งแย่ลงไม่เท่ากันในแต่ละ family** — ดูการเพิ่มของ branching จาก @1 ไป @4 (Δ):

- math เสื่อมหนักสุด: `math_math` Δ=+0.83 (1.88→2.71), `metamath` Δ=+0.65
- dialogue เสถียรสุด: `dolly` Δ=+0.06, `alpaca` Δ=+0.09 — เดาไกลแทบไม่แย่ลง
- code/nl อยู่กลาง Δ≈0.1–0.26

→ implication: **width-draft (หลาย sequence สั้น) น่าจะคุ้มกับ math** ส่วน **depth-draft (sequence
ยาวเส้นเดียว) เวิร์กกับ dialogue/nl** เพราะเดาไกลยังแม่น

## แยกตาม family

- **dialogue** — เดาง่ายสุด + β สูง (vocab หลากหลายแต่ประโยคมี pattern ซ้ำเยอะ "instruction/response")
- **code** — เสถียร P90@น=2 คงที่ทุก step; mean จับความต่างย่อยได้ (python_instruct 1.36 > humaneval 1.27)
- **nl** — wikitext เดาง่าย/β สูง (สารานุกรมเป็นทางการ) ส่วน cnndailymail/openwebtext กลางๆ
- **math** — ยากสุดทุกมิติ: β ต่ำ (เลข/สัญลักษณ์วนซ้ำ vocab โตช้า) แต่ token ถัดไปไม่แน่นอน
  (`= \n` ตามด้วยเลขอะไรก็ได้ → branching สูงสุดถึง 235)

## P90 vs mean (ทำไมเก็บทั้งคู่)

- **P90** robust ต่อ outlier — บอก "เคสแย่ปกติ" เช่น math_math P90@1=3 (90% ของ context แตก ≤3)
- **mean** ไวต่อ tail — math_math mean@1=1.88 ถูกดันจาก context ส่วนน้อย (~2%) ที่แตก ≥10 ทาง
- ดู distribution เต็มได้ที่ `pbe_ecdf.png`

## ข้อควรระวัง

- HumanEval/MBPP มี token น้อย (~20–30k) → β/PBE อาจไม่นิ่งเท่าตัวอื่น (cap ที่ max_samples ทำให้
  corpus ไม่ใหญ่)
- the-stack (gated) ถูกข้าม — code family เหลือ instruction-style 3 ตัว อาจไม่แทน raw source code
- ทุกค่าผูกกับ tokenizer Qwen2.5 — เปลี่ยน tokenizer ตัวเลขเปลี่ยน (สลับด้วย `--tokenizer`)

## ไฟล์ผลลัพธ์

- `results/*.json` — ค่าดิบต่อ dataset (มี `pbe@n.hist` + `heaps.curve` ให้วาดซ้ำได้ไม่ต้อง re-tokenize)
- `workload_character.png` (scatter) · `pbe_ecdf.png` (ECDF) · `heaps_curve.png` (vocab growth)
- `workload_summary.md` (ตาราง P90 + mean) · `workload_report.docx` (รายงานรวม)
