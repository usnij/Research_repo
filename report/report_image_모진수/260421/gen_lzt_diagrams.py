"""
LZT Forward/Backward 핵심 2장
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os
from matplotlib import font_manager

font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
_prop = font_manager.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams["font.family"] = _prop.get_name()

OUT = os.path.dirname(os.path.abspath(__file__))

# 색상 팔레트
C_F      = "#4A90D9"   # Fragment F — 파랑
C_B      = "#E07B39"   # Fragment B — 주황
C_Z2     = "#9B6FBD"   # Zone2 (겹침) — 보라
C_MERGE  = "#5BAD72"   # merged result — 초록
C_WARN   = "#CC4444"   # stop gradient
C_YELLOW = "#F5C518"


def rbox(ax, cx, cy, w, h, text, fc, ec="#999", fs=10, bold=False, lw=1.4, tc="black"):
    rect = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                          boxstyle="round,pad=0.12",
                          facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3)
    ax.add_patch(rect)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal",
            color=tc, zorder=4, linespacing=1.5)


def arr(ax, x1, y1, x2, y2, color="#555", lw=1.6, rad=0.0):
    cs = f"arc3,rad={rad}" if rad != 0 else "arc3,rad=0"
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=14,
                                connectionstyle=cs),
                zorder=5)


# ═══════════════════════════════════════════════════════════════════════
#  Figure A — Forward: Zone 분할 + 블렌딩
# ═══════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(13, 8), facecolor="white")

# 레이아웃: 위=공간 배치, 아래 왼쪽=zone 값, 아래 오른쪽=블렌딩 트리
ax_ray  = fig.add_axes([0.03, 0.60, 0.94, 0.32])   # 위
ax_val  = fig.add_axes([0.03, 0.04, 0.38, 0.50])   # 아래 왼쪽
ax_tree = fig.add_axes([0.45, 0.04, 0.52, 0.50])   # 아래 오른쪽

for ax in (ax_ray, ax_val, ax_tree):
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[:].set_visible(False)
    ax.set_facecolor("white")

# ── 위: 공간 배치 ──────────────────────────────────────────────────────
ax_ray.set_xlim(0, 12); ax_ray.set_ylim(0, 3.2)

# ray 선
ax_ray.annotate("", xy=(11.8, 1.5), xytext=(0.2, 1.5),
                arrowprops=dict(arrowstyle="-|>", color="#222", lw=2))
ax_ray.text(0.08, 1.5, "Ray", va="center", ha="right", fontsize=11)

t1F, t2F, t1B, t2B = 1.2, 6.8, 4.2, 10.0

# zone 배경색
ax_ray.axvspan(t1F, t1B, ymin=0.12, ymax=0.88, color=C_F,  alpha=0.10)
ax_ray.axvspan(t1B, t2F, ymin=0.12, ymax=0.88, color=C_Z2, alpha=0.15)
ax_ray.axvspan(t2F, t2B, ymin=0.12, ymax=0.88, color=C_B,  alpha=0.10)

# F bar
ax_ray.broken_barh([(t1F, t2F - t1F)], (1.75, 0.6),
                   facecolor=C_F, alpha=0.55, edgecolor="#1a55aa", lw=1.8)
ax_ray.text((t1F + t2F)/2, 2.06, "F", ha="center", va="center",
            fontsize=16, fontweight="bold", color="white")

# B bar
ax_ray.broken_barh([(t1B, t2B - t1B)], (0.65, 0.6),
                   facecolor=C_B, alpha=0.55, edgecolor="#aa4400", lw=1.8)
ax_ray.text((t1B + t2B)/2, 0.96, "B", ha="center", va="center",
            fontsize=16, fontweight="bold", color="white")

# 경계선 + 레이블
for x, lbl in [(t1F,"t1_F"), (t1B,"t1_B"), (t2F,"t2_F"), (t2B,"t2_B")]:
    ax_ray.axvline(x, color="#aaa", lw=1.2, ls="--", ymin=0.08, ymax=0.92)
    ax_ray.text(x, 0.18, lbl, ha="center", fontsize=10, color="#555")

# zone 이름 + 비율 공식
zones = [
    ((t1F+t1B)/2, "Zone 1", "r_F1=(t1B-t1F)/dF", C_F),
    ((t1B+t2F)/2, "Zone 2", "r_F2=(t2F-t1B)/dF\nr_B2=(t2F-t1B)/dB", C_Z2),
    ((t2F+t2B)/2, "Zone 3", "r_B3=(t2B-t2F)/dB", C_B),
]
for xc, zname, formula, col in zones:
    ax_ray.text(xc, 2.95, zname, ha="center", fontsize=11,
                fontweight="bold", color=col)
    ax_ray.text(xc, 2.60, formula, ha="center", fontsize=8.5,
                color="#666", linespacing=1.4)

ax_ray.set_title("Figure A : LZT Forward — Zone 분할 및 블렌딩",
                 fontsize=13, fontweight="bold", pad=6)

# ── 아래 왼쪽: 각 Zone의 alpha / color 값 ─────────────────────────────
ax_val.set_xlim(0, 5); ax_val.set_ylim(0, 5.5)
ax_val.text(2.5, 5.1, "Zone별 alpha · color 값", ha="center",
            fontsize=11, fontweight="bold", color="#333")

rows = [
    # (zone, alpha식, color식, 배경색)
    ("Zone 1", "A_F1 = r_F1 · A_F", "V_F1 = r_F1 · V_F", "#d8eaff"),
    ("Zone 2\n(F쪽)", "A_F2 = r_F2 · A_F", "V_F2 = r_F2 · V_F", "#ede0f8"),
    ("Zone 2\n(B쪽)", "A_B2 = r_B2 · A_B", "V_B2 = r_B2 · V_B", "#ede0f8"),
    ("Zone 3", "A_B3 = r_B3 · A_B", "V_B3 = r_B3 · V_B", "#ffe4d0"),
]
col_x = [0.55, 2.2, 3.85]
col_labels = ["Zone", "Alpha", "Color (premul)"]
col_colors = ["#333", "#333", "#333"]

# 헤더
for cx, lbl in zip(col_x, col_labels):
    ax_val.text(cx, 4.65, lbl, ha="center", fontsize=10,
                fontweight="bold", color="#444")
ax_val.axhline(4.45, color="#ccc", lw=1, xmin=0.02, xmax=0.98)

for i, (zn, alph, col, bg) in enumerate(rows):
    y = 3.6 - i * 0.92
    rect = FancyBboxPatch((0.05, y - 0.35), 4.9, 0.72,
                          boxstyle="round,pad=0.06",
                          facecolor=bg, edgecolor="#ccc", lw=1, zorder=2)
    ax_val.add_patch(rect)
    ax_val.text(col_x[0], y, zn,  ha="center", va="center", fontsize=9.5)
    ax_val.text(col_x[1], y, alph, ha="center", va="center", fontsize=9.5)
    ax_val.text(col_x[2], y, col,  ha="center", va="center", fontsize=9.5)

ax_val.text(2.5, 0.2, "V = c · A  (premultiplied color)",
            ha="center", fontsize=9, color="#777", style="italic")

# ── 아래 오른쪽: 블렌딩 트리 ─────────────────────────────────────────
ax_tree.set_xlim(0, 7); ax_tree.set_ylim(0, 5.5)
ax_tree.text(3.5, 5.1, "over-operator 블렌딩 순서", ha="center",
             fontsize=11, fontweight="bold", color="#333")

BW, BH = 3.0, 0.65

# 입력 3개
rbox(ax_tree, 1.0, 4.2, 1.6, BH, "F1\n(Zone1)", "#d8eaff", ec=C_F, fs=9.5)
rbox(ax_tree, 3.5, 4.2, 2.5, BH, "F2 OVER B2\n(Zone2 합성)", "#ede0f8", ec=C_Z2, fs=9.5)
rbox(ax_tree, 6.0, 4.2, 1.6, BH, "B3\n(Zone3)", "#ffe4d0", ec=C_B, fs=9.5)

# Zone2 합성 공식
rbox(ax_tree, 3.5, 3.0, 5.5, BH,
     "A_z2 = A_F2 + A_B2·(1−A_F2)     V_z2 = V_F2 + V_B2·(1−A_F2)",
     "#ede0f8", ec=C_Z2, fs=9.5)

# Zone1 + Zone2
rbox(ax_tree, 3.0, 1.9, 5.5, BH,
     "A_12 = A_F1 + A_z2·(1−A_F1)     V_12 = V_F1 + V_z2·(1−A_F1)",
     "#d0d8f8", ec="#4455bb", fs=9.5)

# merged
rbox(ax_tree, 3.5, 0.75, 6.5, BH,
     "A_merged = A_12 + A_B3·(1−A_12)     V_merged = V_12 + V_B3·(1−A_12)",
     "#d8f0d8", ec=C_MERGE, fs=9.5, bold=True)

# arrows
arr(ax_tree, 3.5, 3.87, 3.5, 3.33)                      # 입력 중앙 → Zone2합성
arr(ax_tree, 1.0, 3.87, 2.2, 2.23)                       # F1 → Zone1+2
arr(ax_tree, 3.5, 2.67, 3.3, 2.23)                       # Zone2합성 → Zone1+2
arr(ax_tree, 3.0, 1.57, 3.2, 1.08)                       # Zone1+2 → merged
arr(ax_tree, 6.0, 3.87, 5.5, 1.08)                       # B3 → merged

# over-operator 힌트
ax_tree.text(3.5, 0.18, "패턴:  A_front + A_back·(1−A_front)",
             ha="center", fontsize=9.5, color="#555", style="italic")

fig.savefig(f"{OUT}/figA_forward.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("figA saved")


# ═══════════════════════════════════════════════════════════════════════
#  Figure B — Backward: 근사 전파 vs 정확한 전파
# ═══════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(13, 7), facecolor="white")

ax_l = fig.add_axes([0.02, 0.05, 0.44, 0.88])   # 왼쪽: T / dL/dC 비교
ax_r = fig.add_axes([0.50, 0.05, 0.48, 0.88])   # 오른쪽: 파라미터 전파 경로

for ax in (ax_l, ax_r):
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[:].set_visible(False)
    ax.set_facecolor("white")

# ── 왼쪽: T와 dL/dC 비교 ─────────────────────────────────────────────
ax_l.set_xlim(0, 10); ax_l.set_ylim(0, 9)
ax_l.text(5.0, 8.6, "Vanilla vs LZT Backward 비교",
          ha="center", fontsize=12, fontweight="bold")

# 컬럼 제목
ax_l.text(2.2, 8.0, "Vanilla", ha="center", fontsize=11,
          fontweight="bold", color=C_F)
ax_l.text(7.5, 8.0, "LZT (근사)", ha="center", fontsize=11,
          fontweight="bold", color=C_B)
ax_l.axvline(5.0, color="#ddd", lw=1.5, ls="--", ymin=0.02, ymax=0.92)

GYS = [6.5, 5.0, 3.5]   # G0, G1, G2 y좌표
BW_S, BH_S = 3.2, 0.65

# Vanilla: T 순차 감소
rbox(ax_l, 2.2, 7.3, BW_S, BH_S, "T_before", "#d8eaff", ec=C_F, fs=10)
T_labels_v = ["T = T_before", "T·(1−a0)", "T·(1−a0)·(1−a1)"]
for i, yi in enumerate(GYS):
    rbox(ax_l, 2.2, yi, BW_S, BH_S, f"G{i}", "#e8f4e8", ec="#5BAD72", fs=11)
    ax_l.text(4.0, yi, T_labels_v[i], va="center", fontsize=8.5, color="#555")
    if i == 0:
        arr(ax_l, 2.2, 7.3 - BH_S/2, 2.2, yi + BH_S/2, color=C_F)
    else:
        arr(ax_l, 2.2, GYS[i-1] - BH_S/2, 2.2, yi + BH_S/2, color=C_F)

ax_l.text(2.2, 2.55, "T_i 순차 감소\n뒤 Gaussian일수록 작은 gradient",
          ha="center", fontsize=9, color=C_F,
          bbox=dict(boxstyle="round,pad=0.3", fc="#eef4ff", ec=C_F, lw=1))

# LZT: T_before 공통 + dL/dC 공통
rbox(ax_l, 7.5, 7.3, BW_S, BH_S, "T_before  (공통)", "#ffe4d0", ec=C_B, fs=10, bold=True)
for i, yi in enumerate(GYS):
    rbox(ax_l, 7.5, yi, BW_S, BH_S, f"G{i}", "#fff4e8", ec=C_B, fs=11)
    rads = [-0.25, 0.0, 0.25]
    ax_l.annotate("", xy=(7.5, yi + BH_S/2),
                  xytext=(7.5, 7.3 - BH_S/2),
                  arrowprops=dict(arrowstyle="-|>", color=C_B, lw=1.6,
                                  connectionstyle=f"arc3,rad={rads[i]}"))

ax_l.text(7.5, 2.55, "T_before 동일\noverlap 크기 무관하게 같은 gradient",
          ha="center", fontsize=9, color=C_B,
          bbox=dict(boxstyle="round,pad=0.3", fc="#fff4e8", ec=C_B, lw=1))

# dL/dC도 공통 — 중앙 하단
ax_l.text(5.0, 1.55, "+ dL/dC (featuresBackward) 도 모든 Gaussian에 동일",
          ha="center", fontsize=10, color="#333",
          bbox=dict(boxstyle="round,pad=0.35", fc="#fffff0", ec="#aaa", lw=1.2))

# zone 비율 무시 표시
ax_l.text(5.0, 0.6, "zone 비율 r  →  stop gradient (무시)",
          ha="center", fontsize=10, color=C_WARN, fontweight="bold",
          bbox=dict(boxstyle="round,pad=0.35", fc="#fff0f0", ec=C_WARN, lw=1.5))

# ── 오른쪽: 파라미터 전파 경로 ───────────────────────────────────────
ax_r.set_xlim(0, 6); ax_r.set_ylim(0, 9)
ax_r.text(3.0, 8.6, "파라미터 전파 경로  (Gaussian i 기준)",
          ha="center", fontsize=12, fontweight="bold")

BW_R, BH_R = 4.5, 0.7

rbox(ax_r, 3.0, 7.7, 2.0, BH_R, "Loss  dL", "#ffdddd", ec=C_WARN, fs=11, bold=True)

rbox(ax_r, 3.0, 6.5, BW_R, BH_R,
     "dL/dC,  dL/dT  (ray 상태)", "#ffeedd", ec="#cc8844", fs=10)

rbox(ax_r, 3.0, 5.2, BW_R, BH_R,
     "featuresIntegrateBwd\ndL/d(feat_i)    dL/d(alpha_i)", "#d8eaff", ec=C_F, fs=10)

rbox(ax_r, 1.5, 3.9, 2.0, BH_R,
     "feat grad\n→ featGradBuf", "#c8dff8", ec="#2255aa", fs=9.5)

rbox(ax_r, 4.5, 3.9, 2.5, BH_R,
     "densityProcess\nHitBwdToBuffer", "#ffe4d0", ec=C_B, fs=9.5)

rbox(ax_r, 4.5, 2.6, 2.5, BH_R,
     "alpha = f(mu, S, q)\n[ray-ellipsoid]", "#fff4e8", ec=C_B, fs=9.5)

rbox(ax_r, 3.0, 1.3, BW_R, BH_R,
     "dL/d(mu, scale, quat)  →  param grad buffer",
     "#d8f0d8", ec=C_MERGE, fs=10, bold=True)

arr(ax_r, 3.0, 7.35, 3.0, 6.85)
arr(ax_r, 3.0, 6.15, 3.0, 5.55)
arr(ax_r, 2.2, 4.85, 1.5, 4.25)
arr(ax_r, 3.8, 4.85, 4.5, 4.25)
arr(ax_r, 4.5, 3.55, 4.5, 2.95)
arr(ax_r, 4.0, 2.25, 3.5, 1.65)

fig.suptitle("Figure B : LZT Backward — 근사 Gradient 전파",
             fontsize=13, fontweight="bold", y=0.98)
fig.savefig(f"{OUT}/figB_backward.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("figB saved")

print("\n완료:", OUT)
