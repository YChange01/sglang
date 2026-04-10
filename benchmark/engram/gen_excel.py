#!/usr/bin/env python3
"""Generate benchmark results Excel from raw data."""

from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter

wb = Workbook()
bold = Font(bold=True)
header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
header_font = Font(bold=True, color="FFFFFF")
ubsmem_fill = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
thin_border = Border(
    left=Side(style="thin"), right=Side(style="thin"),
    top=Side(style="thin"), bottom=Side(style="thin"),
)
center = Alignment(horizontal="center", vertical="center")


def style_header(ws, row, ncols):
    for c in range(1, ncols + 1):
        cell = ws.cell(row=row, column=c)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = center
        cell.border = thin_border


def style_data(ws, row_start, row_end, ncols, highlight_col=None):
    for r in range(row_start, row_end + 1):
        for c in range(1, ncols + 1):
            cell = ws.cell(row=r, column=c)
            cell.border = thin_border
            cell.alignment = center
            if highlight_col and c == 1 and ws.cell(row=r, column=1).value and \
               "UBS-MEM" in str(ws.cell(row=r, column=1).value):
                for cc in range(1, ncols + 1):
                    ws.cell(row=r, column=cc).fill = ubsmem_fill


def auto_width(ws):
    for col in ws.columns:
        max_len = 0
        col_letter = get_column_letter(col[0].column)
        for cell in col:
            if cell.value:
                max_len = max(max_len, len(str(cell.value)))
        ws.column_dimensions[col_letter].width = max(max_len + 3, 12)


# ============================================================
# Sheet 1: Raw Data
# ============================================================
ws1 = wb.active
ws1.title = "Raw Data"

ws1.append(["Engram Cross-Node Benchmark — Raw Data"])
ws1.merge_cells("A1:F1")
ws1.cell(1, 1).font = Font(bold=True, size=14)

ws1.append([])
ws1.append(["Environment"])
ws1.append(["Hardware", "Node1 (141.61.84.245) → Node2 (192.168.84.247), UB fabric, udma2"])
ws1.append(["Table", "10000 rows × 341 dim × float32 = 1364 bytes/row"])
ws1.append(["shmem size", "128 MB"])
ws1.append(["Iterations", "500 per benchmark"])
ws1.append(["Date", "2026-04-10"])
ws1.append([])

# --- Single float load ---
r = ws1.max_row + 1
ws1.append(["Single Float Load (4 bytes)"])
ws1.cell(r, 1).font = Font(bold=True, size=12)
ws1.append(["Mode", "Avg (ns)", "Min (ns)", "Max (ns)"])
style_header(ws1, ws1.max_row, 4)
ws1.append(["LOCAL DRAM", 35.9, 20.0, 170.0])
ws1.append(["UBS-MEM (cache)", 195.0, 20.0, 510.0])
ws1.append([])

# --- Single-row read ---
r = ws1.max_row + 1
ws1.append(["Single-Row Read (1364 bytes)"])
ws1.cell(r, 1).font = Font(bold=True, size=12)
ws1.append(["Mode", "Avg (us)", "Min (us)", "Max (us)", "Throughput (MB/s)"])
style_header(ws1, ws1.max_row, 5)
ws1.append(["LOCAL DRAM", 0.173, 0.040, 6.550, 7894.4])
ws1.append(["TCP", 63.433, 56.980, 471.250, 21.5])
ws1.append(["URMA (RDMA READ)", 1.973, 1.870, 12.150, 691.3])
ws1.append(["UBS-MEM (cache)", 0.583, 0.040, 1.350, 2339.0])
ws1.append([])

# --- Batch read ---
r = ws1.max_row + 1
ws1.append(["Batch Read"])
ws1.cell(r, 1).font = Font(bold=True, size=12)
ws1.append(["Mode", "Batch", "Data (KB)", "Total (us)", "Per-row (us)", "Throughput (MB/s)"])
style_header(ws1, ws1.max_row, 6)

batch_data = [
    # LOCAL DRAM
    ("LOCAL DRAM", 32, 42.6, 3.58, 0.112, 11641.8),
    ("LOCAL DRAM", 64, 85.2, 4.80, 0.075, 17353.8),
    ("LOCAL DRAM", 128, 170.5, 8.04, 0.063, 20697.2),
    ("LOCAL DRAM", 256, 341.0, 16.07, 0.063, 20723.3),
    # TCP
    ("TCP", 32, 42.6, 239.89, 7.496, 173.5),
    ("TCP", 64, 85.2, 400.82, 6.263, 207.7),
    ("TCP", 128, 170.5, 677.63, 5.294, 245.7),
    ("TCP", 256, 341.0, 1149.17, 4.489, 289.8),
    # URMA
    ("URMA (RDMA READ)", 32, 42.6, 4.80, 0.150, 8664.5),
    ("URMA (RDMA READ)", 64, 85.2, 7.03, 0.110, 11848.2),
    ("URMA (RDMA READ)", 128, 170.5, 11.82, 0.092, 14090.2),
    ("URMA (RDMA READ)", 256, 341.0, 21.27, 0.083, 15654.6),
    # UBS-MEM
    ("UBS-MEM (cache)", 32, 42.6, 11.70, 0.366, 3557.6),
    ("UBS-MEM (cache)", 64, 85.2, 11.69, 0.183, 7118.7),
    ("UBS-MEM (cache)", 128, 170.5, 10.57, 0.083, 15759.4),
    ("UBS-MEM (cache)", 256, 341.0, 16.53, 0.065, 20151.5),
]
for row in batch_data:
    ws1.append(list(row))
ws1.append([])

# --- Engram prefetch ---
r = ws1.max_row + 1
ws1.append(["Engram Prefetch (12 tables × N tokens)"])
ws1.cell(r, 1).font = Font(bold=True, size=12)
ws1.append(["Mode", "Tokens", "Reads", "Data (MB)", "Avg (ms)", "Throughput (MB/s)"])
style_header(ws1, ws1.max_row, 6)

engram_data = [
    ("LOCAL DRAM", 32, 384, 0.5, 0.0246, 20271.0),
    ("LOCAL DRAM", 64, 768, 1.0, 0.0517, 19314.7),
    ("LOCAL DRAM", 128, 1536, 2.0, 0.1117, 17895.6),
    ("LOCAL DRAM", 256, 3072, 4.0, 0.2261, 17673.7),
    ("TCP", 32, 384, 0.5, 1.6545, 301.9),
    ("TCP", 64, 768, 1.0, 3.1285, 319.3),
    ("TCP", 128, 1536, 2.0, 6.1643, 324.1),
    ("TCP", 256, 3072, 4.0, 12.1913, 327.8),
    ("URMA (RDMA READ)", 32, 384, 0.5, 0.0309, 16177.6),
    ("URMA (RDMA READ)", 64, 768, 1.0, 0.0592, 16867.7),
    ("URMA (RDMA READ)", 128, 1536, 2.0, 0.1160, 17229.2),
    ("URMA (RDMA READ)", 256, 3072, 4.0, 0.2282, 17513.7),
    ("UBS-MEM (cache)", 32, 384, 0.5, 0.0250, 19973.0),
    ("UBS-MEM (cache)", 64, 768, 1.0, 0.0518, 19275.2),
    ("UBS-MEM (cache)", 128, 1536, 2.0, 0.1182, 16906.5),
    ("UBS-MEM (cache)", 256, 3072, 4.0, 0.2511, 15916.0),
]
for row in engram_data:
    ws1.append(list(row))
ws1.append([])

# --- Paper Engram-27B ---
r = ws1.max_row + 1
ws1.append(["Paper: Engram-27B (8 segs × 320B per token, sparse)"])
ws1.cell(r, 1).font = Font(bold=True, size=12)
ws1.append(["Mode", "Batch", "Reads", "Data (KB)", "Latency", "Unit", "Throughput (MB/s)"])
style_header(ws1, ws1.max_row, 7)

paper_data = [
    ("LOCAL DRAM", 1, 8, 2.5, 0.13, "us", 19148.3),
    ("LOCAL DRAM", 4, 32, 10.0, 0.49, "us", 19854.5),
    ("LOCAL DRAM", 16, 128, 40.0, 2.02, "us", 19337.3),
    ("LOCAL DRAM", 64, 512, 160.0, 7.97, "us", 19612.1),
    ("LOCAL DRAM", 256, 2048, 640.0, 33.35, "us", 18742.1),
    ("LOCAL DRAM", 1024, 8192, 2560.0, 147.0, "us", 17007.7),
    ("TCP", 1, 8, 2.5, 210.92, "us", 11.6),
    ("TCP", 4, 32, 10.0, 214.49, "us", 45.5),
    ("TCP", 16, 128, 40.0, 533.31, "us", 73.2),
    ("TCP", 64, 512, 160.0, 1.69, "ms", 92.2),
    ("TCP", 256, 2048, 640.0, 6.22, "ms", 100.5),
    ("TCP", 1024, 8192, 2560.0, 24.59, "ms", 101.7),
    ("URMA (RDMA READ)", 1, 8, 2.5, 2.75, "us", 888.4),
    ("URMA (RDMA READ)", 4, 32, 10.0, 4.54, "us", 2149.0),
    ("URMA (RDMA READ)", 16, 128, 40.0, 11.65, "us", 3352.7),
    ("URMA (RDMA READ)", 64, 512, 160.0, 39.90, "us", 3916.4),
    ("URMA (RDMA READ)", 256, 2048, 640.0, 152.80, "us", 4090.3),
    ("URMA (RDMA READ)", 1024, 8192, 2560.0, None, "skip", None),
    ("UBS-MEM (cache)", 1, 8, 2.5, 0.13, "us", 18246.7),
    ("UBS-MEM (cache)", 4, 32, 10.0, 0.54, "us", 18102.6),
    ("UBS-MEM (cache)", 16, 128, 40.0, 2.11, "us", 18542.0),
    ("UBS-MEM (cache)", 64, 512, 160.0, 7.81, "us", 20010.0),
    ("UBS-MEM (cache)", 256, 2048, 640.0, 31.59, "us", 19782.3),
    ("UBS-MEM (cache)", 1024, 8192, 2560.0, 169.30, "us", 14766.4),
]
for row in paper_data:
    ws1.append(list(row))
ws1.append([])

# --- Cold/Hot ---
r = ws1.max_row + 1
ws1.append(["Cold/Hot Analysis (1364 bytes)"])
ws1.cell(r, 1).font = Font(bold=True, size=12)
ws1.append(["Mode", "Row", "Cold (us)", "Hot (us)"])
style_header(ws1, ws1.max_row, 4)

cold_hot_data = [
    ("LOCAL DRAM", 0, 0.200, 0.050),
    ("LOCAL DRAM", 2500, 0.090, 0.040),
    ("LOCAL DRAM", 5000, 0.090, 0.040),
    ("LOCAL DRAM", 7500, 0.100, 0.030),
    ("LOCAL DRAM", 9999, 0.070, 0.040),
    ("UBS-MEM (cache)", 0, 0.140, 0.050),
    ("UBS-MEM (cache)", 2500, 0.090, 0.040),
    ("UBS-MEM (cache)", 5000, 0.080, 0.040),
    ("UBS-MEM (cache)", 7500, 0.080, 0.050),
    ("UBS-MEM (cache)", 9999, 0.080, 0.040),
]
for row in cold_hot_data:
    ws1.append(list(row))

auto_width(ws1)

# ============================================================
# Sheet 2: Summary Comparison
# ============================================================
ws2 = wb.create_sheet("Summary")

ws2.append(["Cross-Node Benchmark Summary"])
ws2.merge_cells("A1:F1")
ws2.cell(1, 1).font = Font(bold=True, size=14)
ws2.append([])

# --- Paper 27B comparison (main result) ---
ws2.append(["Paper Engram-27B Latency Comparison (us)"])
ws2.cell(ws2.max_row, 1).font = Font(bold=True, size=12)
ws2.append(["Batch", "LOCAL DRAM", "UBS-MEM", "URMA", "TCP", "UBS-MEM vs TCP"])
style_header(ws2, ws2.max_row, 6)

summary_paper = [
    (1, 0.13, 0.13, 2.75, 210.92, "1622x"),
    (4, 0.49, 0.54, 4.54, 214.49, "397x"),
    (16, 2.02, 2.11, 11.65, 533.31, "253x"),
    (64, 7.97, 7.81, 39.90, 1690, "217x"),
    (256, 33.35, 31.59, 152.80, 6220, "197x"),
    (1024, 147.0, 169.30, "N/A", 24590, "145x"),
]
for row in summary_paper:
    ws2.append(list(row))
ws2.append([])

# --- Speed comparison table ---
r = ws2.max_row + 1
ws2.append(["Speed Ratio (vs TCP = 1x)"])
ws2.cell(r, 1).font = Font(bold=True, size=12)
ws2.append(["Metric", "TCP", "URMA", "UBS-MEM", "LOCAL DRAM"])
style_header(ws2, ws2.max_row, 5)
ws2.append(["Single-row read", "1x", "32x", "109x", "366x"])
ws2.append(["128-tok Engram", "1x", "53x", "52x", "55x"])
ws2.append(["Paper batch=1", "1x", "77x", "1622x", "1622x"])
ws2.append(["Paper batch=64", "1x", "42x", "217x", "212x"])
ws2.append(["Paper batch=256", "1x", "41x", "197x", "187x"])
ws2.append([])

# --- vs CXL paper ---
r = ws2.max_row + 1
ws2.append(["Comparison with CXL Paper (arXiv:2603.10087)"])
ws2.cell(r, 1).font = Font(bold=True, size=12)
ws2.append(["Metric", "CXL (paper)", "UB UBS-MEM (ours)", "UB URMA (ours)"])
style_header(ws2, ws2.max_row, 4)
ws2.append(["Single read latency", "~0.2 us", "0.13 us", "1.97 us"])
ws2.append(["Batch=64 latency", "~100 us", "7.81 us", "39.9 us"])
ws2.append(["128-tok prefetch", "<1 ms", "0.118 ms", "0.116 ms"])
ws2.append(["Peak throughput", "~12 GB/s", "~20 GB/s", "~17.5 GB/s"])

auto_width(ws2)

# Highlight UBS-MEM rows in data sheet
for ws in [ws1, ws2]:
    for row in ws.iter_rows(min_row=1, max_row=ws.max_row):
        if row[0].value and "UBS-MEM" in str(row[0].value):
            for cell in row:
                cell.fill = ubsmem_fill

# Save
output = "/Users/lvpengfei/Projects/sglang-UB/sglang/benchmark/engram/benchmark_results.xlsx"
wb.save(output)
print(f"Saved to {output}")
