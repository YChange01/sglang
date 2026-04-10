#!/usr/bin/env python3
"""Generate benchmark results Excel from raw data (2026-04-10 v2)."""

from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter

wb = Workbook()
bold = Font(bold=True)
header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
header_font = Font(bold=True, color="FFFFFF")
cache_fill = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
nc_fill = PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid")
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


def auto_width(ws):
    for col in ws.columns:
        max_len = 0
        col_letter = get_column_letter(col[0].column)
        for cell in col:
            if cell.value:
                max_len = max(max_len, len(str(cell.value)))
        ws.column_dimensions[col_letter].width = max(max_len + 3, 12)


def add_borders(ws, r_start, r_end, ncols):
    for r in range(r_start, r_end + 1):
        for c in range(1, ncols + 1):
            ws.cell(row=r, column=c).border = thin_border
            ws.cell(row=r, column=c).alignment = center


# ============================================================
# Sheet 1: Raw Data
# ============================================================
ws1 = wb.active
ws1.title = "Raw Data"

ws1.append(["Engram Cross-Node Benchmark — Raw Data (2026-04-10 v2)"])
ws1.merge_cells("A1:G1")
ws1.cell(1, 1).font = Font(bold=True, size=14)

ws1.append([])
ws1.append(["Environment"])
ws1.append(["Hardware", "Node1 (141.61.84.245) → Node2 (192.168.84.247), UB fabric, udma2"])
ws1.append(["Table", "10000 rows × 341 dim × float32, 128 MB shmem"])
ws1.append(["Iterations", "500 per benchmark"])
ws1.append(["Date", "2026-04-10"])
ws1.append([])

# --- Single float load ---
r = ws1.max_row + 1
ws1.append(["Single Float Load (4 bytes)"])
ws1.cell(r, 1).font = Font(bold=True, size=12)
ws1.append(["Mode", "Avg (ns)", "Min (ns)", "Max (ns)"])
style_header(ws1, ws1.max_row, 4)
ws1.append(["LOCAL DRAM", 33.3, 20.0, 180.0])
ws1.append(["UBS-MEM (cache)", 39.2, 20.0, 200.0])
ws1.append(["UBS-MEM (noncache)", 165.1, 160.0, 200.0])
ws1.append([])

# --- Single-seg read ---
r = ws1.max_row + 1
ws1.append(["Single-Seg Read (320 bytes)"])
ws1.cell(r, 1).font = Font(bold=True, size=12)
ws1.append(["Mode", "Avg (us)", "Min (us)", "Max (us)", "Throughput (MB/s)"])
style_header(ws1, ws1.max_row, 5)
ws1.append(["LOCAL DRAM", 0.079, 0.020, 0.290, 4044.5])
ws1.append(["UBS-MEM (cache)", 0.088, 0.020, 0.270, 3628.1])
ws1.append(["UBS-MEM (noncache)", 1.429, 1.340, 10.920, 223.9])
ws1.append(["URMA (RDMA READ)", 1.920, 1.830, 14.700, 166.6])
ws1.append(["TCP", 63.597, 58.650, 580.920, 5.0])
ws1.append([])

# --- Engram-27B ---
r = ws1.max_row + 1
ws1.append(["Engram-27B (8 segs × 320B per token, sparse)"])
ws1.cell(r, 1).font = Font(bold=True, size=12)
ws1.append(["Mode", "Batch", "Reads", "Data (KB)", "Latency", "Unit", "Throughput (MB/s)"])
style_header(ws1, ws1.max_row, 7)

# batch sizes
batches = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

local_lat = [0.40, 0.67, 1.09, 1.82, 3.05, 4.73, 8.82, 17.78, 37.07, 78.82, 172.98, 373.55, 814.97, 1660, 3430]
local_unit = ["us"]*13 + ["us", "us"]
local_thr = [6097.7, 7337.3, 8953.5, 10750.9, 12817.1, 16506.4, 17724.3, 17573.4, 16861.9, 15859.5, 14452.9, 13385.2, 12270.3, 12022.6, 11647.5]

cache_lat = [0.38, 0.64, 1.11, 1.80, 2.79, 4.16, 7.46, 14.90, 30.36, 62.46, 130.51, 286.58, 775.29, 1820, 3190]
cache_thr = [6379.4, 7626.1, 8808.3, 10845.0, 14006.0, 18763.5, 20936.9, 20972.0, 20588.6, 20013.7, 19156.1, 17447.0, 12898.4, 10987.0, 12546.8]

nc_lat = [10.16, 20.21, 40.26, 80.85, 160.53, 322.33, 644.95, 1290, 2580, 5150, 10300, 20560, 41080, 82080, 165740]
nc_thr = [240.3, 241.6, 242.6, 241.6, 243.3, 242.4, 242.3, 242.5, 242.5, 242.6, 242.7, 243.2, 243.4, 243.7, 241.3]

urma_lat = [2.80, 3.38, 4.59, 6.96, 11.77, 21.39, 40.44, 78.40, 154.47, 301.66, 603.25, 1210, 2410, 4830, 9650]
urma_thr = [870.9, 1445.6, 2126.8, 2804.6, 3317.4, 3652.2, 3863.3, 3986.0, 4046.1, 4143.8, 4144.2, 4145.3, 4145.4, 4143.2, 4143.5]

tcp_lat = [105.22, 142.08, 221.66, 380.22, 696.74, 1200, 1760, 3340, 6490, 13090, 25820, 51750, 104180, 200060, 353610]
tcp_thr = [23.2, 34.4, 44.1, 51.4, 56.1, 65.1, 89.0, 93.6, 96.3, 95.5, 96.8, 96.6, 96.0, 100.0, 113.1]

modes = [
    ("LOCAL DRAM", local_lat, local_thr),
    ("UBS-MEM (cache)", cache_lat, cache_thr),
    ("UBS-MEM (noncache)", nc_lat, nc_thr),
    ("URMA (RDMA READ)", urma_lat, urma_thr),
    ("TCP", tcp_lat, tcp_thr),
]

for mode_name, lats, thrs in modes:
    for i, b in enumerate(batches):
        reads = b * 8
        data_kb = reads * 320 / 1024.0
        lat = lats[i]
        unit = "ms" if lat >= 1000 else "us"
        lat_val = lat / 1000.0 if lat >= 1000 else lat
        ws1.append([mode_name, b, reads, round(data_kb, 1), round(lat_val, 2), unit, round(thrs[i], 1)])

ws1.append([])

# --- Cold/Hot ---
r = ws1.max_row + 1
ws1.append(["Cold/Hot Analysis"])
ws1.cell(r, 1).font = Font(bold=True, size=12)
ws1.append(["Mode", "Row", "Cold (us)", "Hot (us)"])
style_header(ws1, ws1.max_row, 4)

cold_hot = [
    ("LOCAL DRAM", 0, 0.250, 0.050),
    ("LOCAL DRAM", 5000, 0.200, 0.030),
    ("LOCAL DRAM", 9999, 0.300, 0.040),
    ("UBS-MEM (cache)", 0, 0.320, 0.030),
    ("UBS-MEM (cache)", 5000, 0.220, 0.040),
    ("UBS-MEM (cache)", 9999, 0.200, 0.040),
    ("UBS-MEM (noncache)", 0, 6.940, 6.860),
    ("UBS-MEM (noncache)", 5000, 7.160, 7.270),
    ("UBS-MEM (noncache)", 9999, 6.910, 6.940),
]
for row in cold_hot:
    ws1.append(list(row))

auto_width(ws1)

# ============================================================
# Sheet 2: Summary
# ============================================================
ws2 = wb.create_sheet("Summary")

ws2.append(["Cross-Node Benchmark Summary (2026-04-10 v2)"])
ws2.merge_cells("A1:F1")
ws2.cell(1, 1).font = Font(bold=True, size=14)
ws2.append([])

# --- Single-seg comparison ---
r = ws2.max_row + 1
ws2.append(["Single-Seg Read Latency (320 bytes)"])
ws2.cell(r, 1).font = Font(bold=True, size=12)
ws2.append(["Mode", "Latency (us)", "Throughput (MB/s)", "vs TCP"])
style_header(ws2, ws2.max_row, 4)
ws2.append(["LOCAL DRAM", 0.079, 4044.5, "805x"])
ws2.append(["UBS-MEM (cache)", 0.088, 3628.1, "723x"])
ws2.append(["UBS-MEM (noncache)", 1.429, 223.9, "44x"])
ws2.append(["URMA (RDMA READ)", 1.920, 166.6, "33x"])
ws2.append(["TCP", 63.597, 5.0, "1x"])
ws2.append([])

# --- Engram-27B key batches ---
r = ws2.max_row + 1
ws2.append(["Engram-27B Latency Comparison (us)"])
ws2.cell(r, 1).font = Font(bold=True, size=12)
ws2.append(["Batch", "LOCAL", "UBS-MEM cache", "UBS-MEM noncache", "URMA", "TCP"])
style_header(ws2, ws2.max_row, 6)

key_batches = [0, 3, 6, 8, 10, 12, 14]  # indices for 1,8,64,256,1024,4096,16384
for idx in key_batches:
    b = batches[idx]
    ws2.append([
        b,
        round(local_lat[idx], 2),
        round(cache_lat[idx], 2),
        round(nc_lat[idx], 2),
        round(urma_lat[idx], 2),
        round(tcp_lat[idx], 2),
    ])
ws2.append([])

# --- Speed ratios ---
r = ws2.max_row + 1
ws2.append(["Speed Ratio (vs TCP = 1x)"])
ws2.cell(r, 1).font = Font(bold=True, size=12)
ws2.append(["Metric", "TCP", "URMA", "UBS-MEM nc", "UBS-MEM cache", "LOCAL"])
style_header(ws2, ws2.max_row, 6)
ws2.append(["Single-seg 320B", "1x", "33x", "44x", "723x", "805x"])
ws2.append(["Engram batch=1", "1x", "38x", "10x", "277x", "263x"])
ws2.append(["Engram batch=64", "1x", "44x", "2.7x", "236x", "200x"])
ws2.append(["Engram batch=1024", "1x", "43x", "2.5x", "198x", "150x"])
ws2.append(["Engram batch=16384", "1x", "37x", "2.1x", "111x", "103x"])
ws2.append([])

# --- Key findings ---
r = ws2.max_row + 1
ws2.append(["Key Findings"])
ws2.cell(r, 1).font = Font(bold=True, size=12)
ws2.append(["1. UBS-MEM cache ≈ LOCAL DRAM (cache-hot data, cross-node transparent)"])
ws2.append(["2. UBS-MEM noncache = true cross-node latency: ~1.27us/seg, 243 MB/s constant"])
ws2.append(["3. URMA single: 1.92us; batch amortized to ~4.1 GB/s (post+poll overhead)"])
ws2.append(["4. Noncache Cold/Hot identical (6.9us) — confirms no CPU cache involvement"])
ws2.append(["5. Cache Cold/Hot: 0.32us → 0.03us — 10x improvement from TLB/cache"])
ws2.append(["6. Noncache cannot batch-parallelize; URMA overtakes at batch>=8"])
ws2.append([])

# --- vs CXL paper ---
r = ws2.max_row + 1
ws2.append(["Comparison with CXL Paper (arXiv:2603.10087)"])
ws2.cell(r, 1).font = Font(bold=True, size=12)
ws2.append(["Metric", "CXL (paper)", "UB cache", "UB noncache", "UB URMA"])
style_header(ws2, ws2.max_row, 5)
ws2.append(["Single seg", "~0.2 us", "0.088 us", "1.43 us", "1.92 us"])
ws2.append(["Batch=64", "~100 us", "7.46 us", "645 us", "40.4 us"])
ws2.append(["Batch=1024", "~1 ms", "0.13 ms", "10.3 ms", "0.60 ms"])
ws2.append(["Peak throughput", "~12 GB/s", "~21 GB/s", "243 MB/s", "~4.1 GB/s"])

auto_width(ws2)

# Highlight rows
for ws in [ws1, ws2]:
    for row in ws.iter_rows(min_row=1, max_row=ws.max_row):
        val = str(row[0].value) if row[0].value else ""
        if "cache)" in val and "noncache" not in val:
            for cell in row:
                cell.fill = cache_fill
        elif "noncache" in val:
            for cell in row:
                cell.fill = nc_fill

output = "/Users/lvpengfei/Projects/sglang-UB/sglang/benchmark/engram/benchmark_results.xlsx"
wb.save(output)
print(f"Saved to {output}")
