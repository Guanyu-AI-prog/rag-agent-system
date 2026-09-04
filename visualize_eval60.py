#!/usr/bin/env python3
"""
60题配对评测可视化（纯RAG vs Agent）
数据源：evaluation/results/eval60_scored.json（2026-09-04 凌晨跑完）
输出：docs/images/06~10 五张图，dpi=150
用法：python visualize_eval60.py
"""

import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DATA_PATH = 'evaluation/results/eval60_scored.json'
OUTPUT_DIR = 'docs/images'

COLOR_A = '#2563eb'   # 纯RAG 蓝
COLOR_B = '#f97316'   # Agent 橙
DIMENSIONS = ['recall', 'completeness', 'hallucination', 'accuracy']
DIM_LABELS = ['召回率', '完整性', '幻觉控制', '准确率']
QTYPE_ORDER = ['单点查询', '对比型', '多跳推理', '流程型', '场景型', '边界/异常']


def load_data():
    with open(DATA_PATH, encoding='utf-8') as f:
        raw = json.load(f)
    records = []  # 每题一条: {qid, group, type, a:{...}, b:{...}}
    for qid in raw['A']:
        ra, rb = raw['A'][qid], raw['B'][qid]
        records.append({
            'qid': int(qid),
            'group': ra.get('group', 'A'),
            'type': ra.get('type', '未知'),
            'a': ra,
            'b': rb,
        })
    records.sort(key=lambda r: r['qid'])
    return raw, records


def dim_avg(records, pipe, dim):
    vals = [r[pipe]['scores'][dim] for r in records if r[pipe].get('scores')]
    return float(np.mean(vals)) if vals else 0.0


def composite_avg(records, pipe):
    vals = [r[pipe]['scores']['composite'] for r in records if r[pipe].get('scores')]
    return float(np.mean(vals)) if vals else 0.0


# ──────────────────────────────────────────────
# 图06：五维总体对比（总体 / A组 / B组 三面板）
# ──────────────────────────────────────────────
def plot_overall_5dim(raw, records, out):
    fig, axes = plt.subplots(1, 3, figsize=(19, 6), sharey=True)
    panels = [
        ('总体（60题）', records),
        ('A组（原30题）', [r for r in records if r['group'] == 'A']),
        ('B组（新增30题·更难）', [r for r in records if r['group'] == 'B']),
    ]
    x = np.arange(len(DIMENSIONS))
    width = 0.35
    for ax, (title, recs) in zip(axes, panels):
        a_vals = [dim_avg(recs, 'a', d) for d in DIMENSIONS]
        b_vals = [dim_avg(recs, 'b', d) for d in DIMENSIONS]
        ax.bar(x - width / 2, a_vals, width, label='纯RAG', color=COLOR_A, alpha=0.85)
        ax.bar(x + width / 2, b_vals, width, label='Agent', color=COLOR_B, alpha=0.85)
        for xi, (av, bv) in enumerate(zip(a_vals, b_vals)):
            ax.text(xi - width / 2, av + 1, f'{av:.0f}', ha='center', fontsize=8,
                    fontweight='bold', color=COLOR_A)
            ax.text(xi + width / 2, bv + 1, f'{bv:.0f}', ha='center', fontsize=8,
                    fontweight='bold', color='#c2410c')
        ax.set_xticks(x)
        ax.set_xticklabels(DIM_LABELS, fontsize=9)
        ax.set_ylim(0, 112)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    axes[0].set_ylabel('得分（0-100）', fontsize=11)
    axes[0].legend(loc='lower right', fontsize=10)
    a_all, b_all = composite_avg(records, 'a'), composite_avg(records, 'b')
    fig.suptitle(f'纯RAG vs Agent 五维对比（综合：纯RAG {a_all:.1f} vs Agent {b_all:.1f}）',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✅ 五维对比图：{out}')


# ──────────────────────────────────────────────
# 图07：能力雷达图（五维，两管道叠加）
# ──────────────────────────────────────────────
def plot_radar(records, out):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='polar')
    values_a = [dim_avg(records, 'a', d) for d in DIMENSIONS]
    values_b = [dim_avg(records, 'b', d) for d in DIMENSIONS]
    for vals, color, label in [(values_a, COLOR_A, '纯RAG'), (values_b, COLOR_B, 'Agent')]:
        v = vals + vals[:1]
        angles = np.linspace(0, 2 * np.pi, len(DIMENSIONS), endpoint=False).tolist()
        angles += angles[:1]
        ax.plot(angles, v, 'o-', linewidth=2, color=color, label=label)
        ax.fill(angles, v, alpha=0.12, color=color)
        for ang, val in zip(angles[:-1], vals):
            ax.annotate(f'{val:.1f}', xy=(ang, val), xytext=(0, 10),
                        textcoords='offset points', ha='center', fontsize=9,
                        color=color, fontweight='bold')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(DIM_LABELS, fontsize=12)
    ax.set_ylim(0, 105)
    ax.set_title('综合能力雷达图（60题均值）', fontsize=14, fontweight='bold', pad=25)
    ax.legend(loc='upper right', bbox_to_anchor=(1.28, 1.08))
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✅ 雷达图：{out}')


# ──────────────────────────────────────────────
# 图08：分题型综合分对比 + 路由分布副标题
# ──────────────────────────────────────────────
def plot_qtype(records, out):
    type_recs = {t: [r for r in records if r['type'] == t] for t in QTYPE_ORDER}
    a_vals = [composite_avg(type_recs[t], 'a') for t in QTYPE_ORDER]
    b_vals = [composite_avg(type_recs[t], 'b') for t in QTYPE_ORDER]

    routes = defaultdict(int)
    for r in records:
        routes[r['b'].get('route', '?')] += 1
    route_txt = '  |  '.join(f'{k} {v}题' for k, v in sorted(routes.items()))

    fig, ax = plt.subplots(figsize=(14, 6.5))
    x = np.arange(len(QTYPE_ORDER))
    width = 0.35
    bars_a = ax.bar(x - width / 2, a_vals, width, label='纯RAG', color=COLOR_A, alpha=0.85)
    bars_b = ax.bar(x + width / 2, b_vals, width, label='Agent', color=COLOR_B, alpha=0.85)
    for bars, vals, c in [(bars_a, a_vals, COLOR_A), (bars_b, b_vals, '#c2410c')]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f'{val:.1f}', ha='center', fontsize=9, fontweight='bold', color=c)
    counts = [len(type_recs[t]) for t in QTYPE_ORDER]
    ax.set_xticks(x)
    ax.set_xticklabels([f'{t}\n(n={n})' for t, n in zip(QTYPE_ORDER, counts)], fontsize=10)
    ax.set_ylim(0, 112)
    ax.set_ylabel('综合分（0-100）', fontsize=11)
    ax.set_title(f'各题型综合分对比（Agent路由：{route_txt}）', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✅ 题型对比图：{out}')


# ──────────────────────────────────────────────
# 图09：逐题配对差值分布（B-A 综合分）
# ──────────────────────────────────────────────
def plot_paired_diff(records, out):
    diffs = np.array([r['b']['scores']['composite'] - r['a']['scores']['composite']
                      for r in records])
    n_better = int((diffs > 0).sum())
    n_worse = int((diffs < 0).sum())
    n_tie = int((diffs == 0).sum())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6),
                                   gridspec_kw={'width_ratios': [1.4, 1]})
    # 左：逐题发散条形
    order = np.argsort(diffs)
    colors = [COLOR_A if d < 0 else (COLOR_B if d > 0 else '#cbd5e1') for d in diffs[order]]
    ax1.barh(np.arange(len(diffs)), diffs[order], color=colors, height=0.75)
    ax1.axvline(0, color='#334155', linewidth=1)
    ax1.set_yticks([])
    ax1.set_xlabel('Agent − 纯RAG 综合分差值（pp）', fontsize=11)
    ax1.set_title('60题逐题配对差值（左=纯RAG占优，右=Agent占优）', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    # 右：汇总
    summary = [n_worse, n_tie, n_better]
    labels = [f'纯RAG占优\n{n_worse}题', f'持平\n{n_tie}题', f'Agent占优\n{n_better}题']
    bars = ax2.bar(labels, summary, color=[COLOR_A, '#cbd5e1', COLOR_B], width=0.55)
    for bar, val in zip(bars, summary):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6,
                 f'{val}题', ha='center', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, max(summary) * 1.25)
    ax2.set_ylabel('题数', fontsize=11)
    ax2.set_title(f'配对结果汇总（中位差值 {np.median(diffs):.1f}pp）',
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✅ 配对差值图：{out}')


# ──────────────────────────────────────────────
# 图10：端到端耗时对比（平均/p50/p95/最长）
# ──────────────────────────────────────────────
def plot_latency(records, out):
    def stats(pipe):
        vals = np.array([r[pipe]['e2e_seconds'] for r in records if r[pipe].get('success')])
        return [vals.mean(), np.percentile(vals, 50), np.percentile(vals, 95), vals.max()]
    a_stats, b_stats = stats('a'), stats('b')
    stat_names = ['平均', 'p50', 'p95', '最长']

    fig, ax = plt.subplots(figsize=(11, 6.5))
    x = np.arange(len(stat_names))
    width = 0.35
    bars_a = ax.bar(x - width / 2, a_stats, width, label='纯RAG', color=COLOR_A, alpha=0.85)
    bars_b = ax.bar(x + width / 2, b_stats, width, label='Agent', color=COLOR_B, alpha=0.85)
    for bars, vals, c in [(bars_a, a_stats, COLOR_A), (bars_b, b_stats, '#c2410c')]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f'{val:.1f}s', ha='center', fontsize=10, fontweight='bold', color=c)
    # p95 倍数标注
    ratio = b_stats[2] / a_stats[2] if a_stats[2] else 0
    ax.annotate(f'p95 相差 {ratio:.1f} 倍', xy=(2 + width / 2, b_stats[2]),
                xytext=(2.6, b_stats[2] * 0.82), fontsize=12, color='#dc2626',
                fontweight='bold', arrowprops=dict(arrowstyle='->', color='#dc2626'))
    ax.set_xticks(x)
    ax.set_xticklabels(stat_names, fontsize=11)
    ax.set_ylim(0, max(b_stats) * 1.18)
    ax.set_ylabel('端到端耗时（秒）', fontsize=11)
    ax.set_title('端到端耗时对比（纯RAG 7.1s vs Agent 15.1s，Agent尾部延迟高）',
                 fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✅ 耗时对比图：{out}')


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    raw, records = load_data()
    print(f'📊 加载 {len(records)} 题 × 2 管道')
    plot_overall_5dim(raw, records, os.path.join(OUTPUT_DIR, '06_overall_5dim.png'))
    plot_radar(records, os.path.join(OUTPUT_DIR, '07_capability_radar.png'))
    plot_qtype(records, os.path.join(OUTPUT_DIR, '08_qtype_comparison.png'))
    plot_paired_diff(records, os.path.join(OUTPUT_DIR, '09_paired_diff.png'))
    plot_latency(records, os.path.join(OUTPUT_DIR, '10_latency_comparison.png'))
    print(f'\n🎉 全部图表已生成到 {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
