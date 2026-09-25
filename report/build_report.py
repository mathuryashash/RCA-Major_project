"""Build the RVITM/VTU-format major project report as a .docx.

Follows IQAC-VTU_Guidelines-for-Major-Project-work.pdf (page setup, font
sizes, report flow order) and IQAC-RVITM_Major Project Report format_Cover
pages.docx (cover/certificate/declaration wording, institution details).

VTU page setup: A4, 1.5/double spacing, margins L=1.25in R=1in T=0.75in
B=0.75in. Font sizes: chapter title 18pt centered, chapter heading 16pt left
justified, section/subsection heading 14pt left, body text 12pt justified.
"""
from __future__ import annotations

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.shared import Inches, Pt, RGBColor
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TEAM = [
    ("Yashash Mathur", "1RFXXISXXX"),
    ("Aditya Prakash", "1RFXXISXXX"),
    ("Utsav Upadhyay", "1RFXXISXXX"),
    ("Shresth Modi", "1RFXXISXXX"),
]
GUIDE = "Dr. Niharika P Kumar"
GUIDE_DESIGNATION = "Associate Professor, Dept. of ISE, RVITM"
HOD = "Dr. Vinoth Kumar M"
PRINCIPAL = "Dr. Nagashettappa Biradar"
TITLE = "LocalRCA: On-Device Root Cause Analysis for Windows Endpoints Without Ground Truth"
YEAR = "2026-27"


def set_base_style(doc):
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)
    pf = normal.paragraph_format
    pf.line_spacing = 1.5
    pf.space_after = Pt(6)
    rPr = normal.element.rPr
    if rPr is None:
        rPr = OxmlElement("w:rPr")
        normal.element.insert(0, rPr)
    rFonts = OxmlElement("w:rFonts")
    rFonts.set(qn("w:eastAsia"), "Times New Roman")
    rPr.append(rFonts)


def set_margins(section):
    section.page_width = Inches(8.27)
    section.page_height = Inches(11.69)
    section.top_margin = Inches(0.75)
    section.bottom_margin = Inches(0.75)
    section.left_margin = Inches(1.25)
    section.right_margin = Inches(1.0)


def add_centered(doc, text, size=12, bold=False, space_after=6, space_before=0, all_caps=False):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.space_before = Pt(space_before)
    run = p.add_run(text.upper() if all_caps else text)
    run.bold = bold
    run.font.size = Pt(size)
    run.font.name = "Times New Roman"
    return p


def add_chapter_title(doc, number, title):
    """18pt centered chapter title, per VTU font table."""
    doc.add_page_break()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(18)
    run = p.add_run(f"CHAPTER {number}")
    run.bold = True
    run.font.size = Pt(16)
    p2 = doc.add_paragraph()
    p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p2.paragraph_format.space_after = Pt(24)
    run2 = p2.add_run(title.upper())
    run2.bold = True
    run2.font.size = Pt(18)


def add_section(doc, text, level=1):
    """16pt left for section headings, 14pt for subsections."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(8)
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(16 if level == 1 else 14)
    return p


def add_body(doc, text, justify=True):
    p = doc.add_paragraph()
    if justify:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_after = Pt(8)
    run = p.add_run(text)
    run.font.size = Pt(12)
    run.font.name = "Times New Roman"
    return p


def add_bullets(doc, items):
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.line_spacing = 1.5
        run = p.runs[0] if p.runs else p.add_run("")
        run.text = item
        run.font.size = Pt(12)


def add_table(doc, header, rows, widths_in=None):
    table = doc.add_table(rows=0, cols=len(header))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr_cells = table.add_row().cells
    for i, text in enumerate(header):
        hdr_cells[i].text = str(text)
        for para in hdr_cells[i].paragraphs:
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in para.runs:
                run.bold = True
                run.font.size = Pt(11)
    for row in rows:
        cells = table.add_row().cells
        for i, text in enumerate(row):
            cells[i].text = str(text)
            for para in cells[i].paragraphs:
                para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in para.runs:
                    run.font.size = Pt(10.5)
    if widths_in:
        for row in table.rows:
            for i, w in enumerate(widths_in):
                row.cells[i].width = Inches(w)
    return table


def add_caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(12)
    run = p.add_run(text)
    run.italic = True
    run.font.size = Pt(10.5)


def add_figure(doc, path, caption, width_in=6.0):
    """Centered image plus caption; path is relative to the repository root."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.add_run().add_picture(str(ROOT / path), width=Inches(width_in))
    add_caption(doc, caption)


def add_toc_field(doc):
    p = doc.add_paragraph()
    run = p.add_run()
    fldChar1 = OxmlElement("w:fldChar")
    fldChar1.set(qn("w:fldCharType"), "begin")
    instrText = OxmlElement("w:instrText")
    instrText.set(qn("xml:space"), "preserve")
    instrText.text = r'TOC \o "1-3" \h \z \u'
    fldChar2 = OxmlElement("w:fldChar")
    fldChar2.set(qn("w:fldCharType"), "separate")
    fldChar3 = OxmlElement("w:t")
    fldChar3.text = "Table of contents — right-click and 'Update Field' in Word to populate."
    fldChar4 = OxmlElement("w:fldChar")
    fldChar4.set(qn("w:fldCharType"), "end")
    r_element = run._r
    r_element.append(fldChar1)
    r_element.append(instrText)
    r_element.append(fldChar2)
    t_run = OxmlElement("w:r")
    t_run.append(fldChar3)
    r_element.addnext(t_run)
    t_run.addnext(fldChar4)


def add_page_number_footer(section):
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    fld1 = OxmlElement("w:fldChar")
    fld1.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    fld2 = OxmlElement("w:fldChar")
    fld2.set(qn("w:fldCharType"), "end")
    run._r.append(fld1)
    run._r.append(instr)
    run._r.append(fld2)


def build():
    doc = Document()
    set_base_style(doc)
    section = doc.sections[0]
    set_margins(section)
    add_page_number_footer(section)

    # ---------------- COVER / TITLE PAGE ----------------
    add_centered(doc, "VISVESVARAYA TECHNOLOGICAL UNIVERSITY", size=16, bold=True, space_after=2, space_before=20)
    add_centered(doc, "Belagavi, Karnataka, India", size=12, space_after=30)

    add_centered(doc, "A PROJECT REPORT ON", size=13, bold=True, space_after=16)
    add_centered(doc, f'"{TITLE}"', size=15, bold=True, space_after=16)
    add_centered(doc, "Submitted in partial fulfilment for the award of Degree of", size=12, space_after=2)
    add_centered(doc, "Bachelor of Engineering", size=13, bold=True, space_after=2)
    add_centered(doc, "IN", size=12, space_after=2)
    add_centered(doc, "Information Science & Engineering", size=13, bold=True, space_after=20)

    add_centered(doc, "Submitted by", size=12, space_after=10)
    for name, usn in TEAM:
        add_centered(doc, f"{name}   {usn}", size=12, space_after=2)

    add_centered(doc, "Under the Guidance of", size=12, space_before=16, space_after=6)
    add_centered(doc, GUIDE, size=13, bold=True, space_after=2)
    add_centered(doc, GUIDE_DESIGNATION, size=12, space_after=20)

    add_centered(doc, "R V INSTITUTE OF TECHNOLOGY AND MANAGEMENT \u00ae", size=14, bold=True, space_before=10, space_after=2)
    add_centered(doc, "BANGALORE \u2013 560076", size=12, space_after=2)
    add_centered(doc, YEAR, size=12, space_after=0)

    # ---------------- CERTIFICATE ----------------
    doc.add_page_break()
    add_centered(doc, "R V INSTITUTE OF TECHNOLOGY AND MANAGEMENT", size=14, bold=True, space_before=10)
    add_centered(doc, "Department of Information Science & Engineering", size=12, space_after=2)
    add_centered(doc, "Bangalore \u2013 560076", size=12, space_after=20)
    add_centered(doc, "CERTIFICATE", size=15, bold=True, space_after=16)

    names_str = ", ".join(f"{n} ({u})" for n, u in TEAM[:-1]) + f" and {TEAM[-1][0]} ({TEAM[-1][1]})"
    add_body(doc,
        f'Certified that the project work entitled "{TITLE}" is carried out by '
        f"{names_str}, who are bonafide students of RV Institute of Technology and "
        f"Management, Bangalore, in partial fulfilment for the award of degree of "
        f"Bachelor of Engineering in Information Science and Engineering of the "
        f"Visvesvaraya Technological University, Belagavi during the year {YEAR}. "
        "It is certified that all corrections/suggestions indicated for the internal "
        "assessment have been incorporated in the report deposited in the departmental "
        "library. The project report has been approved as it satisfies the academic "
        "requirements in respect of project work prescribed for the said degree.")

    doc.add_paragraph()
    sig_table = add_table(doc, ["Signature of Guide", "Signature of HoD", "Signature of the Principal"],
                           [[GUIDE, HOD, PRINCIPAL]])
    doc.add_paragraph()
    add_body(doc, "External Viva:", justify=False)
    add_table(doc, ["Sl. No.", "Name of Examiners", "Signature with Date"], [["1", "", ""], ["2", "", ""]])

    # ---------------- DECLARATION ----------------
    doc.add_page_break()
    add_centered(doc, "DECLARATION", size=15, bold=True, space_before=10, space_after=16)
    names_decl = ", ".join(n for n, _ in TEAM[:-1]) + f" and {TEAM[-1][0]}"
    add_body(doc,
        f'We, {names_decl}, students of the seventh/eighth semester B.E., Department '
        f'of Information Science and Engineering, hereby declare that the project '
        f'titled "{TITLE}" has been carried out by us and submitted in partial '
        "fulfilment for the award of degree of Bachelor of Engineering in Information "
        "Science and Engineering. We declare that this work has not been carried out "
        "by any other students for the award of a degree in any other branch, and that "
        "it does not infringe upon anyone's copyright, patents or other intellectual "
        "property rights.")
    doc.add_paragraph()
    add_body(doc, "Place: Bangalore", justify=False)
    add_body(doc, f"Date: __ / __ / {YEAR.split('-')[0]}", justify=False)
    doc.add_paragraph()
    add_body(doc, "Names:                                                          Signature", justify=False)
    for i, (name, _) in enumerate(TEAM, 1):
        add_body(doc, f"{i}. {name}", justify=False)

    # ---------------- ACKNOWLEDGEMENT ----------------
    doc.add_page_break()
    add_centered(doc, "ACKNOWLEDGEMENT", size=15, bold=True, space_before=10, space_after=16)
    add_body(doc,
        "We would like to express our sincere gratitude to our guide, "
        f"{GUIDE}, {GUIDE_DESIGNATION}, for her continuous guidance, valuable "
        "suggestions and encouragement throughout the course of this project. Her "
        "insistence on measuring every claim against real, collected data rather than "
        "assumption shaped the direction of this work.")
    add_body(doc,
        f"We thank {HOD}, Head of the Department of Information Science and "
        f"Engineering, and {PRINCIPAL}, Principal, RV Institute of Technology and "
        "Management, for providing the infrastructure and academic environment "
        "necessary to carry out this project.")
    add_body(doc,
        "We also thank the Project Review Committee and all faculty members of the "
        "Department of ISE for their periodic review and constructive feedback, and "
        "our families and friends for their continued support.")

    # ---------------- ABSTRACT ----------------
    doc.add_page_break()
    add_centered(doc, "ABSTRACT", size=15, bold=True, space_before=10, space_after=16)
    add_body(doc,
        "Post-hoc diagnosis of a personal or workstation-class machine is poorly "
        "served by existing tooling: cloud observability stacks assume a fleet, a "
        "network path and a willingness to export telemetry, while the Windows Event "
        "Log records that a fault occurred but rarely why. This project presents "
        "LocalRCA, a desktop system that records system telemetry on a single "
        "machine, learns that machine's normal behaviour with an LSTM autoencoder, "
        "and attempts to explain incidents using Granger causality constrained by "
        "false-discovery-rate correction, an effect-size floor and a subsystem "
        "topology prior. The system runs entirely on the endpoint; the collector "
        "opens no sockets, and the only network capability is an opt-in, off-by-"
        "default check for a newer release that transmits nothing. Because a "
        "personal machine offers no labelled incidents, the system is evaluated in "
        "two ways: seven controlled fault injections, which supply ground truth by "
        "construction, and a population survey that replays the pipeline over 596 "
        "incidents discovered in 30 days of real collected history. The system "
        "detects and correctly attributes injected CPU, disk and memory faults, and "
        "in one case produced a correct causal explanation of a fault it was never "
        "told about. Across real history it produces a supported causal chain for "
        "106 of 596 incidents (17.8%), and the hand-written subsystem prior rejects "
        "22.7% of statistically accepted pairs. Preparing the system for use by "
        "non-specialists added two requirements: the learned model is encrypted "
        "at rest under the user's Windows credentials, and the interface leads "
        "with a single plain-language verdict per screen while keeping every raw "
        "measurement one switch away. The most useful property of such a "
        "system is argued to be its willingness to report that it cannot explain an "
        "incident, and this rate is quantified across the full survey population.")

    # ---------------- TOC / LOF / LOT ----------------
    doc.add_page_break()
    add_centered(doc, "TABLE OF CONTENTS", size=15, bold=True, space_before=10, space_after=16)
    add_toc_field(doc)

    doc.add_page_break()
    add_centered(doc, "LIST OF FIGURES", size=14, bold=True, space_before=10, space_after=12)
    figs = [
        "Fig 3.1  Process architecture: collector, database, desktop application",
        "Fig 4.1  Inference pipeline: detection, causal testing, terminal states",
        "Fig 6.1  Causal yield against incident window length",
        "Fig 7.1  Subsystem topology prior and rejected causal directions",
        "Fig 8.1  Captured Data tab, default view",
        "Fig 8.2  Captured Data tab, advanced view",
        "Fig 8.3  Baseline & Training tab, default view",
        "Fig 8.4  Run RCA Inference tab, default view",
    ]
    for f in figs:
        add_body(doc, f, justify=False)

    doc.add_page_break()
    add_centered(doc, "LIST OF TABLES", size=14, bold=True, space_before=10, space_after=12)
    tabs = [
        "Table 6.1  Controlled fault injections and outcomes",
        "Table 6.2  Granger max-lag sweep: tested population vs. explanation rate",
        "Table 7.1  Topology prior: kept, pruned, and cycle-broken causal pairs",
        "Table 8.1  What the default and advanced views show",
        "Table 9.1  Test suite and static-analysis verification summary",
    ]
    for t in tabs:
        add_body(doc, t, justify=False)

    doc.add_page_break()
    add_centered(doc, "ABBREVIATIONS", size=14, bold=True, space_before=10, space_after=12)
    abbrs = [
        ("RCA", "Root Cause Analysis"),
        ("LSTM", "Long Short-Term Memory (recurrent neural network)"),
        ("FDR", "False Discovery Rate"),
        ("ADF", "Augmented Dickey\u2013Fuller (unit-root test)"),
        ("WAL", "Write-Ahead Logging (SQLite journal mode)"),
        ("GUI", "Graphical User Interface"),
        ("DPAPI", "Windows Data Protection API"),
        ("EFS", "Encrypting File System (Windows)"),
        ("IEEE", "Institute of Electrical and Electronics Engineers"),
        ("CIE", "Continuous Internal Evaluation"),
        ("SEE", "Semester End Examination"),
    ]
    add_table(doc, ["Abbreviation", "Expansion"], abbrs)

    # ==========================================================
    # CHAPTER 1 - INTRODUCTION
    # ==========================================================
    add_chapter_title(doc, 1, "Introduction")

    add_section(doc, "1.1  Background")
    add_body(doc,
        "Diagnosing why a personal computer became slow, stalled or crashed is "
        "typically a manual exercise: an engineer lines up resource graphs against "
        "the Windows Event Log and guesses. Fleet-oriented observability platforms "
        "(Datadog, New Relic, Prometheus-based stacks) automate this, but assume "
        "infrastructure a single machine does not have \u2014 a collector network, a "
        "time-series backend, and consent to export telemetry describing which "
        "applications a person runs. No comparable tool exists that runs entirely on "
        "one Windows endpoint, keeps every byte of telemetry local, and still "
        "attempts a causal (not merely correlational) explanation of a fault.")

    add_section(doc, "1.2  Problem Statement")
    add_body(doc,
        "Build a desktop system for Windows that (a) collects system telemetry with "
        "zero network egress, (b) learns what \u201cnormal\u201d behaviour looks like for the "
        "specific machine it runs on, without any labelled training data, (c) detects "
        "when the machine departs from that learned normal, and (d) attempts to "
        "explain the departure with a statistically defensible causal argument rather "
        "than a black-box severity score \u2014 while being honest, in the user interface "
        "and in the reported metrics, about how often it cannot produce an "
        "explanation at all.")

    add_section(doc, "1.3  Design Constraints")
    add_body(doc, "Three constraints follow from targeting a single endpoint, and they shape the entire design.")
    add_bullets(doc, [
        "No egress. Telemetry describing application usage is sensitive. The system "
        "is built so that nothing is transmitted; the privacy property is structural, "
        "enforced by the absence of network code, rather than by policy.",
        "No ground truth. A personal machine has no labelled incidents. The model "
        "cannot be trained to recognise known faults and cannot be evaluated against "
        "a labelled test set. The response adopted here is to manufacture ground "
        "truth by causing faults deliberately (Chapter 6).",
        "Cold start is unavoidable. \u201cNormal\u201d must be learned from the machine itself, "
        "so the tool is useless until it has observed enough normal behaviour \u2014 "
        "approximately 21 hours of clean collection in the deployed configuration.",
    ])

    add_section(doc, "1.4  Objectives")
    add_bullets(doc, [
        "Design and implement an end-to-end on-device root-cause-analysis pipeline "
        "combining reconstruction-error anomaly detection with statistically "
        "constrained Granger causality.",
        "Build a fault-injection harness that manufactures ground truth on a machine "
        "that has none, and use it to verify the pipeline can identify a cause it "
        "was not told about.",
        "Measure, at population scale rather than on single runs, how often the "
        "causal layer produces an answer at all.",
        "Audit the hand-written subsystem topology prior against real causal-pair "
        "evidence and quantify what it forbids.",
        "Deliver a working Windows desktop application (not only a research script) "
        "with a background collector service, a training UI, and an inference UI.",
        "Make that application usable by a non-specialist without hiding what it "
        "collects, and protect the learned model at rest.",
    ])

    add_section(doc, "1.5  Scope")
    add_body(doc,
        "The system targets a single Windows 10/11 endpoint. It is evaluated on one "
        "physical machine over roughly 30\u201340 days of real collected history plus "
        "seven controlled fault injections. It does not attempt fleet-scale or "
        "microservice root cause analysis, does not use a service call graph, and "
        "does not claim statistical generality beyond the host it was evaluated on "
        "\u2014 this limitation is treated explicitly in Chapter 9 rather than concealed.")

    # ==========================================================
    # CHAPTER 2 - LITERATURE REVIEW
    # ==========================================================
    add_chapter_title(doc, 2, "Literature Review")

    add_section(doc, "2.1  Anomaly Detection in Multivariate Time Series")
    add_body(doc,
        "Reconstruction-based detectors using LSTM autoencoders are a standard "
        "approach when labelled anomalies are unavailable (Malhotra et al., 2016; "
        "Hundman et al., 2018). Training on nominal data only, and thresholding "
        "reconstruction error, avoids requiring labelled faults \u2014 which is precisely "
        "the constraint an endpoint imposes, since a personal machine has no incident "
        "labels to train against.")

    add_section(doc, "2.2  Causal Discovery from Observational Data")
    add_body(doc,
        "Granger causality (Granger, 1969) tests whether one time series improves "
        "prediction of another beyond the second series' own history. It is "
        "correlational under a temporal constraint rather than truly interventional, "
        "and is well known to over-report on correlated series without correction. "
        "This project applies Benjamini\u2013Hochberg false-discovery-rate control "
        "(Benjamini & Hochberg, 1995) across the full set of ordered metric pairs, "
        "an effect-size floor, and a domain prior over subsystems to suppress this "
        "over-reporting tendency.")

    add_section(doc, "2.3  Root Cause Analysis in Distributed Systems")
    add_body(doc,
        "Microservice RCA work typically exploits service call graphs or "
        "distributed traces to constrain candidate causes (Kim, Sumbaly & Shah, "
        "2013; Chen et al., 2014). No such topology exists on a single endpoint, "
        "which motivates the explicit, inspectable subsystem map used in this "
        "project \u2014 and, as Chapter 7 shows, makes the quality of that map a "
        "first-order concern rather than an implementation detail.")

    add_section(doc, "2.4  Benchmarked Causal RCA at Fleet Scale")
    add_body(doc,
        "Recent work evaluates causal discovery for root cause analysis at a scale "
        "this project does not attempt. Ikram et al. (NeurIPS 2022) discover root "
        "causes across thousands of microservice nodes and report top-k recall "
        "against ground truth gathered from a real production incident corpus. Pham, "
        "Ha & Zhang (ASE 2024) benchmark six causal-RCA methods (including "
        "CausalRCA and BARO) across microservice architectures and find that none "
        "dominates uniformly, that performance is highly sensitive to the underlying "
        "call graph, and that published recall numbers do not transfer cleanly "
        "between systems. Both works assume a call graph or trace corpus that a "
        "single endpoint does not have; the corresponding gap in this project is "
        "that its 596-incident survey has no comparable ground truth, so Chapter 6 "
        "reports how often the system speaks, not how often it is correct against a "
        "labelled corpus.")

    add_section(doc, "2.5  Positioning of This Work")
    add_body(doc,
        "The distinguishing constraints of this project are the single-machine "
        "scope, the absence of ground truth, and a deliberate design commitment to "
        "reporting uncertainty rather than producing a ranking regardless of "
        "evidence. Existing literature either assumes fleet-scale infrastructure "
        "(Section 2.3\u20132.4) or does not address the causal-explanation problem at "
        "all (Section 2.1). This project sits in the gap between the two.")

    # ==========================================================
    # CHAPTER 3 - SYSTEM ARCHITECTURE
    # ==========================================================
    add_chapter_title(doc, 3, "System Architecture")

    add_section(doc, "3.1  Process Architecture")
    add_body(doc,
        "The system comprises two processes sharing a single SQLite database "
        "(Fig. 3.1). RCA-Collector.exe runs headless at logon and is the only "
        "process that writes; RCA-Desktop.exe is the PySide6/Qt6 graphical "
        "application and only reads. This separation exists because collection must "
        "survive the user interface being closed \u2014 a user closing the diagnostic "
        "window should not silently stop data collection. A PowerShell supervisor "
        "script (supervise.ps1) restarts the collector if it exits unexpectedly, "
        "because creating a Task Scheduler logon task that restarts on failure "
        "requires elevated privileges that a diagnostic tool should not demand of "
        "its user.")
    add_body(doc,
        "The collector samples 29 numeric features every 30 seconds, the top "
        "processes by CPU and resident memory every 300 seconds (tightening to 30 "
        "seconds automatically under detected load, to catch short-lived spikes), "
        "and an allow-listed subset of Windows Event Log records. Retention is "
        "tiered by sensitivity: numeric system readings are kept for 365 days, "
        "per-process detail for 30 days, and the foreground application name \u2014 the "
        "single most personal field collected \u2014 for only 30 days, with "
        "SQLite's secure_delete option enabled so that purged rows do not survive "
        "in unallocated database pages.")
    add_body(doc,
        "At the time of writing the evaluation host had accumulated 124,674 system "
        "samples, 1,835,250 process samples and 6,600 Event Log records over 59 "
        "days, occupying 227 MB. Coverage of that span is 73.1%: 157 breaks in "
        "collection account for 383 hours not recorded. The population survey in "
        "Chapter 6 draws on the most recent 30 days of this history.")

    add_section(doc, "3.2  Technology Stack")
    add_table(doc, ["Layer", "Technology", "Purpose"], [
        ["Desktop UI", "PySide6 / Qt 6", "Training and inference GUI, results display"],
        ["Background collector", "Python, WMI/psutil, pywin32", "Zero-network telemetry sampling"],
        ["Persistence", "SQLite (WAL mode)", "Single-file local database, no server process"],
        ["Anomaly model", "PyTorch (LSTM autoencoder)", "Learns per-machine \u201cnormal\u201d behaviour"],
        ["Causal inference", "statsmodels (Granger causality, ADF test)", "Constrained causal graph construction"],
        ["Protection at rest", "DPAPI (pywin32), EFS", "Per-user encryption of the model; best-effort folder encryption"],
        ["Packaging", "PyInstaller, signtool", "Signed Windows executables"],
    ], widths_in=[1.6, 2.4, 2.5])
    add_caption(doc, "Table 3.1  Technology stack by architectural layer.")

    add_section(doc, "3.3  Data Flow")
    add_body(doc,
        "At runtime the collector writes rows to telemetry.db in Write-Ahead "
        "Logging (WAL) mode, which allows the desktop application to read "
        "concurrently without blocking the writer. The desktop application never "
        "writes to the production database during normal operation; a trained model "
        "is persisted separately as a single PyTorch artifact (telemetry_model.pt) "
        "containing the network weights, per-metric thresholds, the feature scaler, "
        "and metadata such as training timestamp and reference reconstruction "
        "error, used later to detect when the model has gone stale. Since v1.5.1 "
        "this artifact is written encrypted (Section 5.5).")

    # ==========================================================
    # CHAPTER 4 - METHODOLOGY
    # ==========================================================
    add_chapter_title(doc, 4, "Methodology")

    add_section(doc, "4.1  Objective of the Study")
    add_body(doc,
        "Given a machine's own telemetry history and no external labels, evaluate "
        "(a) how accurately the system detects and attributes deliberately injected "
        "faults of known type, and (b) how often, across real (non-injected) "
        "incident history, the causal-inference layer produces a statistically "
        "supported explanation rather than a bare anomaly flag.")

    add_section(doc, "4.2  Measurement Parameters")
    add_bullets(doc, [
        "Detection: number of the 29 tracked metrics flagged anomalous per incident.",
        "Causal edges: number of directed metric-to-metric edges surviving Granger "
        "testing, false-discovery-rate correction, the effect-size floor, and the "
        "subsystem topology filter.",
        "Explanation rate: fraction of incidents (population-wide, not just injected "
        "ones) for which at least one causal edge survives.",
        "False-positive rate: fraction of idle-machine samples incorrectly flagged "
        "as anomalous, measured with a dedicated negative-control injection.",
    ])

    add_section(doc, "4.3  Measurement Method")
    add_body(doc,
        "Scaling and windowing: a min\u2013max scaler is fitted on the clean baseline "
        "only; incident windows are transformed with the baseline's parameters and "
        "clipped to [0, 1]. Fitting on the incident window itself would normalise "
        "away the deviation being sought. Training windows are built within "
        "contiguous collection segments only, never spanning a gap in collection.")
    add_body(doc,
        "Anomaly detection: an LSTM autoencoder compresses a window of T samples "
        "over n = 29 features to a 32-dimensional latent vector and reconstructs "
        "it. Only the encoder's final hidden state reaches the bottleneck, so the "
        "entire window is represented by 32 numbers before reconstruction; that "
        "compression is the detection mechanism itself. Reconstruction error is "
        "reduced over time but not over features, which makes the output "
        "attributable to a specific metric rather than to a window as a whole. "
        "Per-metric thresholds are set at the 99th percentile of validation error, "
        "because reconstructability varies widely across channels \u2014 a near-constant "
        "channel reconstructs almost perfectly, while a bursty one does not, and a "
        "single global threshold would flag the bursty channel permanently.")
    add_body(doc,
        "Constrained causal inference: for each ordered pair of anomalous metrics, "
        "both series are differenced until an Augmented Dickey\u2013Fuller test rejects "
        "a unit root, then tested for Granger causality at lags 1 through L (default "
        "L = 5). A pair is skipped entirely unless the aligned sample count N "
        "satisfies N \u2265 3L + 2, which explains the majority of this system's "
        "negative results. Surviving p-values are corrected with Benjamini\u2013Hochberg "
        "at \u03b1 = 0.05, and an effect-size proxy (F / (F + N) \u2265 0.10) is additionally "
        "required, since negligible improvements become statistically significant "
        "given a long enough history. Accepted pairs form a directed graph; cycles "
        "are removed and remaining edges are filtered against the subsystem "
        "topology prior (Chapter 7). Candidates are then ranked by a weighted "
        "composite of causal outflow, temporal priority, inflow, severity and event "
        "correlation, combined with PageRank on the reversed graph.")

    add_section(doc, "4.4  Tools and Equipment")
    add_bullets(doc, [
        "One Windows 11 workstation, used both to generate the 30\u201340 day telemetry "
        "history used in the survey and to run all controlled fault injections.",
        "PyTorch for the LSTM autoencoder; statsmodels for Granger causality and "
        "the Augmented Dickey\u2013Fuller test; SQLite for persistence.",
        "A purpose-built fault-injection harness (CPU burn, disk write flood, "
        "memory allocation, idle-machine negative control) to manufacture ground "
        "truth, since a personal machine offers none.",
        "pytest for the automated test suite; ruff for static lint analysis.",
    ])

    add_section(doc, "4.5  Error Sources and Mitigation")
    add_bullets(doc, [
        "Collection gaps (sleep, hibernation, crash) can corrupt a training window "
        "that spans the gap; mitigated by only building windows from maximal "
        "contiguous collection segments.",
        "A short incident window cannot support Granger testing at all (the sample "
        "floor); mitigated by explicitly reporting \u201cnot tested\u201d as a distinct outcome "
        "from \u201ctested, no causal chain found\u201d rather than conflating the two.",
        "Granger causality alone over-reports on correlated series; mitigated with "
        "false-discovery-rate correction, an effect-size floor, and a domain "
        "topology prior, each of which is independently audited in Chapters 6\u20137.",
        "A trained model can go stale as machine usage patterns drift; mitigated by "
        "comparing recent median reconstruction error against the value recorded at "
        "training time and surfacing a staleness warning in the UI.",
    ])

    # ==========================================================
    # CHAPTER 5 - IMPLEMENTATION
    # ==========================================================
    add_chapter_title(doc, 5, "Implementation")

    add_section(doc, "5.1  Collector Service")
    add_body(doc,
        "The collector (src/telemetry/collector.py) runs a single-threaded loop "
        "that ticks system sampling every 30 seconds, process sampling on a "
        "load-dependent cadence, Windows Event Log polling, and a retention purge "
        "pass. Consent is enforced independently at two layers: a PermissionError is "
        "raised inside run_once() if consent has not been granted, so even a direct "
        "command-line invocation without going through the GUI dialog cannot "
        "collect, and the command-line entry point additionally blocks install/run "
        "before the GUI has ever shown a consent dialog.")

    add_section(doc, "5.2  Persistence Layer")
    add_body(doc,
        "telemetry.db uses SQLite in WAL mode with synchronous=NORMAL and "
        "secure_delete=ON. The samples table's ts column is the INTEGER PRIMARY "
        "KEY, making every timestamp range query (used throughout the RCA pipeline) "
        "an indexed seek rather than a full-table scan. During this project's "
        "verification pass (Chapter 9), the query layer was reworked so that every "
        "RCA-time read (detect_incidents, window_between, recent_real_window) "
        "pushes its timestamp filter into the SQL WHERE clause instead of loading "
        "the entire table into pandas and filtering afterward \u2014 a change that keeps "
        "query cost proportional to the requested window rather than to total "
        "collection history as retention grows toward the 365-day ceiling.")

    add_section(doc, "5.3  Desktop Application")
    add_body(doc,
        "The desktop application is organised into three tabs matching the natural "
        "workflow: Captured Data (collection statistics), Baseline & Training "
        "(configure and run LSTM training), and Run RCA Inference (select or define "
        "an incident window and run the causal pipeline). Training and inference "
        "each run on a background QThread worker so the UI remains responsive; "
        "progress and log output are streamed back to the main thread via Qt "
        "signals. A single Advanced switch in the window header, persisted with "
        "QSettings, chooses between a plain-language default view and the full "
        "technical view (Chapter 8).")

    add_section(doc, "5.4  Fault-Injection Harness")
    add_body(doc,
        "Because a personal machine provides no labelled incidents, a harness was "
        "built that causes a specific, known disturbance (CPU load, disk write "
        "flood, memory allocation, or a deliberate idle period as a negative "
        "control), waits for samples to land at the real collection cadence, runs "
        "the production inference pipeline unmodified over the injection window, "
        "and scores the result against the known injected cause. This is the "
        "project's principal answer to the \u201cno ground truth\u201d constraint from "
        "Section 1.3.")

    add_section(doc, "5.5  Protection at Rest and Signed Binaries")
    add_body(doc,
        "The trained model holds the scaler's per-metric bounds and thresholds, "
        "which together form a compact statistical profile of how the machine is "
        "used. It is wrapped with the Windows Data Protection API (DPAPI), which "
        "binds the ciphertext to the user's logon credentials, so a copy taken to "
        "another account or machine cannot be read. The write is atomic, via a "
        "temporary file and a rename, so an interrupted save cannot leave a "
        "truncated model. Artifacts from earlier versions carry no header, still "
        "load, and are encrypted the next time the model is trained. If the "
        "user's credentials are reset and the key is lost, the application says "
        "the model must be retrained rather than surfacing a raw Windows error.")
    add_body(doc,
        "The database and logs, which DPAPI does not wrap, are covered by a second, "
        "best-effort layer: at start-up the collector asks the Encrypting File "
        "System (EFS) to encrypt its data folder. EFS does not exist on Windows "
        "Home editions. The evaluation host runs Windows 11 Home, the request "
        "fails on every start, and its 227 MB database remains in plaintext, "
        "protected only by file-system permissions. The failure is logged rather "
        "than silent, but the protection a user receives depends on their edition "
        "of Windows. Likewise, the model on the evaluation host was trained before "
        "encryption was added and stays plaintext until it is retrained.")
    add_body(doc,
        "Independent review of the encryption change found that training wrote an "
        "intermediate checkpoint in plaintext and deleted it only on success, so "
        "an interrupted run left unencrypted weights on disk. The checkpoint is "
        "now removed in a finally block, and any leftover from an earlier crash is "
        "purged when training starts.")
    add_body(doc,
        "Both executables are Authenticode-signed with an RFC 3161 timestamp, and "
        "each release publishes a SHA-256 checksum. The certificate is "
        "self-signed: it shows that a binary is unchanged since it was built, not "
        "who built it, so Windows SmartScreen still warns on first run. Removing "
        "the warning requires a certificate from an authority Windows trusts.")

    # ==========================================================
    # CHAPTER 6 - RESULTS AND DISCUSSION (fault injection + population)
    # ==========================================================
    add_chapter_title(doc, 6, "Results and Discussion")

    add_section(doc, "6.1  Evaluation by Fault Injection")
    add_body(doc,
        "Seven controlled fault injections were run on the evaluation machine, each "
        "in a 30-minute window unless noted, and scored against the known injected "
        "cause.")
    add_table(doc, ["Fault", "Samples", "Flagged", "Edges", "Outcome"], [
        ["CPU, 7 min", "14", "6/29", "\u2014", "not tested (below floor)"],
        ["CPU", "60", "6/29", "6", "explained correctly"],
        ["Disk", "60", "4/29", "0", "detected, unexplained"],
        ["Memory", "60", "2/29", "0", "wrong process named"],
        ["Memory (refixed)", "60", "3/29", "0", "detected, attributed"],
        ["Memory (clean)", "60", "0/29", "\u2014", "not measurable on host"],
        ["Idle", "60", "1/29", "\u2014", "3.4% false positives"],
    ], widths_in=[1.5, 0.9, 0.9, 0.7, 2.0])
    add_caption(doc, "Table 6.1  Controlled fault injections and outcomes.")

    add_body(doc,
        "The CPU fault at 30 minutes produced the project's strongest single "
        "result: half the CPU cores were burned with no information given to the "
        "pipeline, and the ranking named cpu_pct first at a score of 1.000, with the "
        "surviving causal edges oriented away from CPU \u2014 the signature of a genuine "
        "root cause, since nothing upstream drives it. The same fault run for only "
        "7 minutes (14 samples) fell below the Granger sample floor of 17 and was "
        "correctly reported as \u201cnot tested\u201d rather than as a failed analysis, "
        "demonstrating that the causal layer distinguishes starvation from a "
        "negative result.")
    add_body(doc,
        "The memory injection exposed a genuine defect: the anomaly was detected "
        "correctly, but attribution ranked candidate processes by mean CPU usage "
        "while resident memory \u2014 the metric actually driving the fault \u2014 was "
        "measured but never sorted on. A process that allocates memory and then "
        "sleeps could therefore never be named as the cause, a defect that had been "
        "concealed until this point because the CPU and disk faults were both "
        "CPU-heavy and had been attributed correctly by accident of their shape. "
        "This was fixed and re-verified as \u201cMemory (refixed)\u201d in Table 6.1.")

    add_section(doc, "6.2  Population-Scale Evaluation")
    add_body(doc,
        "Fault injection establishes correctness on a handful of cases but yields "
        "single observations. To measure frequency, the production pipeline was "
        "replayed, read-only and without any injection, over every incident "
        "discoverable in 30 days of real collected history on the evaluation "
        "machine: 596 incidents.")
    add_body(doc,
        "Of the 596 incidents discovered, 215 (36%) fall below the Granger sample "
        "floor and cannot be tested at any lag setting. Of the 381 that can be "
        "tested, 106 produce a supported causal chain. The end-to-end figure a user "
        "experiences is therefore 106 / 596 = 17.8%: roughly one incident in six "
        "receives a causal explanation. The funnel is reported rather than only the "
        "survivor-filtered rate: the share of analysable incidents explained is "
        "27.8%, which is the more flattering number and describes fewer "
        "situations.")

    add_table(doc, ["Max Lag (L)", "Incidents Found", "Below Floor", "Tested", "Explained", "% of All", "% of Tested"], [
        ["3", "598", "0", "598", "110", "18.4%", "18.4%"],
        ["4", "597", "193", "404", "106", "17.8%", "26.2%"],
        ["5 (default)", "596", "215", "381", "106", "17.8%", "27.8%"],
    ], widths_in=[1.1, 1.1, 1.0, 0.9, 1.1, 0.9, 1.1])
    add_caption(doc, "Table 6.2  Granger max-lag sweep over an identical 30-day, ~597-incident population.")

    add_body(doc,
        "The sample floor is a parameter artefact rather than a fundamental limit: "
        "short incidents are widened before analysis to exactly the model's window "
        "size (15 samples), while the floor at L = 5 requires 17 \u2014 two constants "
        "chosen independently in the same pipeline. Re-running the survey at L in "
        "{3, 4, 5} over an identical population (Table 6.2) confirms the cliff "
        "disappears entirely at L \u2264 3 (0 incidents below the floor, versus 215 at "
        "L = 5), but explanations rise only modestly, from 106 to 110, while the "
        "tested population nearly triples. The gain from lowering L is honest "
        "reporting \u2014 \u201ctested, nothing found\u201d instead of \u201cnot tested\u201d \u2014 rather than "
        "additional insight.")
    add_body(doc,
        "Causal yield rises with incident window length, confirming at scale the "
        "starvation effect first observed in the single 7-minute-versus-30-minute "
        "CPU injection pair: incidents of 0\u201330 minutes explain at 14.7%, rising "
        "roughly monotonically to 63.6% for incidents exceeding 6 hours.")

    # ==========================================================
    # CHAPTER 7 - TOPOLOGY PRIOR AUDIT
    # ==========================================================
    add_chapter_title(doc, 7, "Auditing the Subsystem Topology Prior")

    add_section(doc, "7.1  Motivation")
    add_body(doc,
        "Since no service call graph exists on a single endpoint (unlike the "
        "microservice RCA literature reviewed in Chapter 2), the system encodes "
        "domain knowledge as an explicit, hand-written map of permissible subsystem "
        "influence: power \u2192 cpu \u2192 memory \u2192 disk \u2192 network, plus process as a source "
        "feeding every subsystem. This map had never been checked against evidence "
        "before this project's verification pass.")

    add_section(doc, "7.2  Audit Method and Findings")
    add_body(doc,
        "Every statistically accepted causal pair across the 30-day survey "
        "population was classified as kept (survives the map), pruned (rejected by "
        "the map), or cycle-broken (rejected by cycle removal, independent of the "
        "map). Of 538 total accepted-pair evaluations, 337 (62.6%) were kept, 122 "
        "(22.7%) were pruned by the topology map, and 79 (14.7%) were removed by "
        "cycle-breaking \u2014 making the hand-written map the single largest filter in "
        "the causal pipeline after the statistical significance gates themselves.")
    add_table(doc, ["Outcome", "Pair Evaluations", "% of Total"], [
        ["Kept (survives to graph)", "337", "62.6%"],
        ["Pruned by topology map", "122", "22.7%"],
        ["Removed by cycle-breaking", "79", "14.7%"],
    ], widths_in=[2.6, 1.6, 1.2])
    add_caption(doc, "Table 7.1  Disposition of every statistically accepted causal pair, 30-day survey.")

    add_body(doc,
        "Every rejected transition has zero survivors, because the map forbids "
        "whole classes of direction rather than individual edges. Judged "
        "individually, several rejected directions are almost certainly real "
        "physical mechanisms rather than statistical noise: disk \u2192 memory (the "
        "Windows standby file cache expanding under write pressure) is the "
        "strongest single rejected relationship measured in this audit, at a "
        "strength of 0.654; disk \u2192 cpu plausibly reflects interrupt and DPC "
        "handling costing CPU time during heavy disk activity; and network \u2192 disk "
        "(a download arriving over the network and being written to disk) is "
        "present, though smaller in this run than in an earlier measurement, at 4 "
        "occurrences. Conversely, the smallest rejected classes (network \u2192 process, "
        "disk \u2192 process, at 1\u20132 occurrences each) are more plausibly spurious "
        "statistical artefacts than real, wrongly forbidden mechanisms.")
    add_body(doc,
        "The prior is therefore not so much wrong as one-directional: it models "
        "resource pressure flowing \u201cdownhill\u201d through the subsystem chain and has "
        "no vocabulary for the feedback that makes I/O itself expensive. Adding the "
        "reverse edges would make the topology graph cyclic, where cycle-breaking "
        "already removes 14.7% of pairs \u2014 so the two correction mechanisms would "
        "begin to compete with each other. This is recorded as an open design "
        "question for future work rather than patched without further evidence.")

    # ==========================================================
    # CHAPTER 8 - APPLICATION WALKTHROUGH
    # ==========================================================
    add_chapter_title(doc, 8, "Application Walkthrough")

    add_body(doc,
        "A review of the running application found that its first screen was a "
        "table of every captured channel with its raw latest value, and that "
        "training readiness was expressed four different ways (clean samples, "
        "longest run, current run, time remaining) that a non-specialist had to "
        "reconcile alone. Hiding this information would contradict the project's "
        "premise, since a user cannot meaningfully consent to collection they "
        "cannot inspect. The released application (v1.5.1) therefore offers two "
        "presentations of the same computation, selected by one Advanced switch in "
        "the window header. The switch is off by default and remembered between "
        "sessions. Nothing is computed differently in either view.")

    add_section(doc, "8.1  Captured Data Tab")
    add_body(doc,
        "The default view shows one sentence whose colour encodes the state of "
        "collection. It reads Healthy only when coverage is at least 90% with at "
        "most three breaks, and Partial data from 50%; below that it says "
        "collection is just getting started. The evaluation host reads Partial "
        "data at 73% coverage, which is accurate if unflattering (Fig. 8.1). Pause, "
        "refresh and update-check controls are visible in both views.")
    add_figure(doc, "report/assets/fig8_1_data_simple.png",
               "Fig 8.1  Captured Data tab, default view.")
    add_body(doc,
        "With Advanced on, the same tab adds the raw store statistics \u2014 sample "
        "counts, coverage, breaks, retention periods, size on disk and database "
        "path \u2014 and the full table of captured channels with their latest values "
        "and whether the model uses each one (Fig. 8.2).")
    add_figure(doc, "report/assets/fig8_2_data_advanced.png",
               "Fig 8.2  Captured Data tab, advanced view.")

    add_section(doc, "8.2  Baseline & Training Tab")
    add_body(doc,
        "The default view states whether a model can be trained and, if not, "
        "roughly how long remains, followed by the age of the current model and "
        "the Train button (Fig. 8.3). The advanced view restores the four "
        "readiness counters and the training hyperparameters \u2014 LSTM epochs and "
        "window size \u2014 with an estimate of training time calibrated from the "
        "last run on the machine. Training runs on a background thread, and its "
        "log console appears only once training produces output.")
    add_figure(doc, "report/assets/fig8_3_training_simple.png",
               "Fig 8.3  Baseline & Training tab, default view.")

    add_section(doc, "8.3  Run RCA Inference Tab")
    add_body(doc,
        "The default view keeps the incident picker, the time estimate and the Run "
        "button, and opens results on the plain-language Report tab (Fig. 8.4). "
        "The advanced view adds the custom start and end times, the Granger "
        "maximum lag, and the Score and Outflow columns of the ranked candidate "
        "table, and opens results on that table. The Causal Graph and Anomaly "
        "Timeline tabs are available in both views.")
    add_figure(doc, "report/assets/fig8_4_inference_simple.png",
               "Fig 8.4  Run RCA Inference tab, default view.")

    add_table(doc, ["Tab", "Default view", "Advanced view adds"], [
        ["Captured Data", "One coloured status sentence; pause/refresh",
         "Store statistics, retention, database path, channel table"],
        ["Baseline & Training", "Readiness sentence; model age; Train",
         "Four readiness counters; epochs, window size, time estimate"],
        ["Run RCA Inference", "Incident picker; Run; Report tab first",
         "Custom range, Granger max lag, Score/Outflow columns"],
    ], widths_in=[1.5, 2.3, 2.6])
    add_caption(doc, "Table 8.1  What the default and advanced views show.")

    add_body(doc,
        "This design was checked by one expert review and by driving the live "
        "application; it has not been measured with users. Whether the default "
        "view helps a non-specialist reach a correct conclusion is therefore "
        "unestablished, and is listed as future work.")

    # ==========================================================
    # CHAPTER 9 - TESTING AND VERIFICATION
    # ==========================================================
    add_chapter_title(doc, 9, "Testing and Verification")

    add_section(doc, "9.1  Automated Test Suite")
    add_body(doc,
        "The project carries an automated pytest suite covering the collector, "
        "persistence layer, causal engine, and desktop UI logic. At the time of "
        "this report the suite comprises 150 tests, all passing, with static "
        "analysis (ruff) reporting zero outstanding issues across the source and "
        "test trees. A methodological finding from this project's own verification "
        "process is worth recording explicitly: nine real defects were found in "
        "code that already carried a passing test suite of 130 tests, by "
        "adversarial, role-specific code review rather than by writing additional "
        "tests. In nearly every case the existing test had asked a question "
        "adjacent to the one that actually mattered \u2014 a database value instead of "
        "the bytes on disk, one file instead of the full storage footprint, a "
        "function instead of the code path that calls it.")

    add_section(doc, "9.2  Multi-Perspective Verification")
    add_body(doc,
        "The system was independently reviewed from three specialist perspectives "
        "against the live application, the real 226 MB production database, and "
        "the actual running desktop UI (not documentation or source alone):")
    add_bullets(doc, [
        "Security review: SQL-injection surface, consent enforcement, subprocess/"
        "command-injection safety, and the self-update mechanism's TLS handling all "
        "passed with no exploitable defects found; data-at-rest encryption and log "
        "redaction were flagged as best-effort limitations appropriate for a "
        "single-user local tool, documented rather than silently assumed complete.",
        "Database review: two real two-process race conditions were found and "
        "fixed \u2014 one in corruption-recovery when both the collector and desktop "
        "process open a damaged database simultaneously, one in concurrent schema "
        "migration on first launch after an upgrade \u2014 plus a query-performance fix "
        "pushing timestamp filters into SQL instead of loading full tables into "
        "memory before filtering.",
        "UX/UI review: field-level tooltips were added for statistical/ML "
        "parameters (Granger Max Lag, LSTM Window Size) that had no in-app "
        "explanation, an inconsistent empty-state was restyled to match the rest "
        "of the interface, and a progress bar that ambiguously represented both "
        "\u201cdata collected so far\u201d and \u201ctraining run in progress\u201d was split into two "
        "distinct, unambiguous representations.",
    ])
    add_body(doc,
        "Every fix identified by this review process was implemented directly "
        "(not merely reported) and re-verified against the full test suite, which "
        "remained at 142/142 passing after all changes.")
    add_body(doc,
        "A second round of security and database review examined the encryption "
        "and signing changes. It found one real defect \u2014 the plaintext training "
        "checkpoint described in Section 5.5, found independently by both "
        "reviewers in code whose tests all passed because they exercised only the "
        "successful path \u2014 and four lesser issues: a silent fallback to "
        "plaintext when DPAPI is unavailable (now logged), a raw Windows error on "
        "decryption failure (now a clear message), an undocumented choice of "
        "timestamp server, and a confusing prompt when the signing script is "
        "re-run. All five were fixed, and the suite grew to 150 tests.")

    add_section(doc, "9.3  Verification Summary")
    add_table(doc, ["Check", "Result"], [
        ["Automated tests (pytest)", "150 / 150 passing"],
        ["Static analysis (ruff)", "0 issues"],
        ["Security review", "No exploitable defects; 3 documented low-severity limitations"],
        ["Database review", "2 race conditions found and fixed; 1 performance fix applied"],
        ["UX/UI review", "3 usability findings, all fixed and re-verified"],
        ["Encryption/signing review", "1 defect and 4 minor issues, all fixed"],
        ["Design review of live UI", "Default/advanced split implemented and checked in the running app"],
        ["Release build", "v1.5.1, signed, SHA-256 published"],
        ["Real-data L-sweep (Ch. 6)", "596-incident population, 3 lag settings, ~44 min real compute"],
        ["Topology audit (Ch. 7)", "538 causal-pair evaluations classified and reported"],
    ], widths_in=[2.6, 3.1])
    add_caption(doc, "Table 9.1  Verification activities completed for this report.")

    # ==========================================================
    # CHAPTER 10 - CONCLUSION AND FUTURE WORK
    # ==========================================================
    add_chapter_title(doc, 10, "Conclusion and Future Work")

    add_section(doc, "10.1  Conclusion")
    add_body(doc,
        "This project presented an on-device root cause analysis system for "
        "Windows endpoints operating without ground truth, network access or fleet "
        "context. Under controlled injection it detects and attributes CPU, disk "
        "and memory faults, and has produced a correct causal explanation of a "
        "fault it was never told about. Across 596 real historical incidents it "
        "explains roughly one incident in six, declines to explain the rest, and \u2014 "
        "deliberately \u2014 distinguishes \u201cnot tested\u201d from \u201ctested and nothing "
        "survived\u201d rather than collapsing the two into a single failure state. It "
        "is argued that this last property matters more than the raw explanation "
        "rate: a tool that honestly reports \u201cno causal chain was supported\u201d when "
        "the data genuinely cannot support one is more useful than a confident "
        "ranking that changes with the chosen analysis window, and in this system "
        "the engineering required to make it say so honestly exceeded the "
        "engineering required to compute an answer at all.")
    add_body(doc,
        "The released application also protects the learned model under the "
        "user's Windows credentials, ships signed binaries, and leads with a plain "
        "verdict while keeping every measurement available to anyone who asks for "
        "it.")

    add_section(doc, "10.2  Limitations")
    add_bullets(doc, [
        "All measurements come from one machine. Nothing here establishes "
        "generality across hardware, and one measurement \u2014 memory-fault detection "
        "\u2014 is provably unobtainable on the evaluation host, making a second machine "
        "a prerequisite for generalising this work rather than an optional "
        "enhancement.",
        "Correctness is not established at population scale. The 596-incident "
        "survey has no independently known cause for each incident, so the 17.8% "
        "explanation rate reports how often the causal layer speaks, never whether "
        "it is right. The rate is a ceiling on usefulness, not a measure of "
        "accuracy.",
        "No injected fault tested so far has a cause that is not also the loudest "
        "metric by severity, so a correct causal answer and a correct severity-"
        "ranking answer remain indistinguishable in the fault-injection evidence "
        "collected to date.",
        "Granger causality is correlation under a temporal constraint, not true "
        "intervention. The false-discovery-rate correction, effect floor and "
        "subsystem prior exist to suppress the resulting over-reporting, and "
        "Chapter 7 shows the prior itself is measurably imperfect.",
        "Protection at rest is uneven. The database relies on EFS, which Windows "
        "Home editions lack, and the binaries are self-signed, so SmartScreen "
        "still warns every new user.",
        "The default view has not been evaluated with users.",
    ])

    add_section(doc, "10.3  Future Work")
    add_bullets(doc, [
        "Repeat the full evaluation (fault injection and population survey) on a "
        "second, independent machine to establish whether the explanation rate and "
        "topology findings generalise.",
        "Design and run an injected fault whose cause is deliberately not the "
        "loudest severity metric, to separate causal correctness from severity "
        "ranking in the evidence.",
        "Resolve whether the subsystem topology prior should admit the feedback "
        "directions (disk \u2192 memory, disk \u2192 cpu) it currently forbids, and if so, "
        "how to reconcile that with the existing cycle-breaking mechanism.",
        "Extend automated test coverage toward the categories of defect this "
        "project's adversarial review found (storage footprint, cross-process "
        "behaviour, and code paths rather than functions in isolation), since "
        "these were systematically under-tested relative to unit-level logic.",
        "Encrypt the database itself (for example with SQLCipher) so that Windows "
        "Home users receive the same protection as Pro users.",
        "Measure with users whether the default view leads them to correct "
        "conclusions faster than the advanced view.",
    ])

    # ==========================================================
    # REFERENCES
    # ==========================================================
    doc.add_page_break()
    add_centered(doc, "REFERENCES", size=15, bold=True, space_before=10, space_after=16)
    refs = [
        "[1]  P. Malhotra, A. Ramakrishnan, G. Anand, L. Vig, P. Agarwal, and G. Shroff, "
        "\u201cLSTM-based encoder-decoder for multi-sensor anomaly detection,\u201d in Proc. ICML "
        "Anomaly Detection Workshop, 2016.",
        "[2]  K. Hundman, V. Constantinou, C. Laporte, I. Colwell, and T. Soderstrom, "
        "\u201cDetecting spacecraft anomalies using LSTMs and nonparametric dynamic "
        "thresholding,\u201d in Proc. ACM SIGKDD, 2018, pp. 387\u2013395.",
        "[3]  C. W. J. Granger, \u201cInvestigating causal relations by econometric models and "
        "cross-spectral methods,\u201d Econometrica, vol. 37, no. 3, pp. 424\u2013438, 1969.",
        "[4]  Y. Benjamini and Y. Hochberg, \u201cControlling the false discovery rate: a "
        "practical and powerful approach to multiple testing,\u201d J. Roy. Statist. Soc. B, "
        "vol. 57, no. 1, pp. 289\u2013300, 1995.",
        "[5]  M. Kim, R. Sumbaly, and S. Shah, \u201cRoot cause detection in a service-oriented "
        "architecture,\u201d in Proc. ACM SIGMETRICS, 2013, pp. 93\u2013104.",
        "[6]  P. Chen, Y. Qi, P. Zheng, and D. Hou, \u201cCauseInfer: automatic and distributed "
        "performance diagnosis with hierarchical causality graph in large distributed "
        "systems,\u201d in Proc. IEEE INFOCOM, 2014, pp. 1887\u20131895.",
        "[7]  A. Ikram, S. Chakraborty, S. Mitra, S. Saini, S. Bagchi, and M. Kocaoglu, "
        "\u201cRoot cause analysis of failures in microservices through causal discovery,\u201d "
        "in Advances in Neural Information Processing Systems (NeurIPS), vol. 35, "
        "2022, pp. 31158\u201331170.",
        "[8]  L. Pham, H. Ha, and H. Zhang, \u201cRoot cause analysis for microservice system "
        "based on causal inference: How far are we?\u201d in Proc. 39th IEEE/ACM Int. Conf. "
        "on Automated Software Engineering (ASE), 2024.",
        "[9]  L. Page, S. Brin, R. Motwani, and T. Winograd, \u201cThe PageRank citation "
        "ranking: bringing order to the web,\u201d Stanford InfoLab, Tech. Rep. 1999-66, 1999.",
        "[10] D. A. Dickey and W. A. Fuller, \u201cDistribution of the estimators for "
        "autoregressive time series with a unit root,\u201d J. Amer. Statist. Assoc., vol. 74, "
        "no. 366, pp. 427\u2013431, 1979.",
    ]
    for r in refs:
        add_body(doc, r, justify=False)

    return doc


if __name__ == "__main__":
    doc = build()
    doc.save("report/localrca_major_project_report.docx")
    print("wrote report/localrca_major_project_report.docx")
