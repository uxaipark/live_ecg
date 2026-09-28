#!/usr/bin/env python3
"""Render the introduction's key screens.

Each screen is an HTML page drawn in SVG from data the engine produced
(`ecg-eval intro-dump`) and captured by headless Chrome at 1600x900, scale 2.
Public corpora only: the patch corpus is private and none of its signal is
shown.

    tools/intro/render.py DATA_DIR OUT_DIR
"""
import json
import os
import subprocess
import sys

DATA, OUT = sys.argv[1], sys.argv[2]
IMG = os.path.join(OUT, "images")
SRC = os.path.join(DATA, "html")
os.makedirs(IMG, exist_ok=True)
os.makedirs(SRC, exist_ok=True)
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def load(name):
    with open(os.path.join(DATA, name + ".json")) as f:
        return json.load(f)


CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body { width: 1600px; height: 900px; overflow: hidden; }
body { background: #ffffff; color: #17212b;
  font-family: "Apple SD Gothic Neo", "Pretendard", "Noto Sans KR", -apple-system, sans-serif; }
.page { position: relative; width: 1600px; height: 900px; padding: 56px 64px 0 64px; }
.kicker { font-size: 18px; font-weight: 700; letter-spacing: .08em; color: #c0392b; text-transform: uppercase; }
h1 { font-size: 44px; font-weight: 800; letter-spacing: -.01em; margin-top: 6px; }
.sub { font-size: 22px; color: #4a5866; margin-top: 10px; line-height: 1.45; }
.foot { position: absolute; left: 64px; right: 64px; bottom: 26px; font-size: 15px; color: #7a8794;
  display: flex; justify-content: space-between; border-top: 1px solid #e6eaee; padding-top: 12px; }
.legend { display: flex; gap: 22px; font-size: 18px; color: #33414e; align-items: center; }
.legend i { display: inline-block; width: 14px; height: 14px; border-radius: 4px; margin-right: 7px; vertical-align: -1px; }
.stat { font-size: 20px; color: #33414e; }
.stat b { font-size: 30px; color: #17212b; }
svg text { font-family: "Apple SD Gothic Neo", -apple-system, sans-serif; }
.mono { font-family: "SF Mono", Menlo, monospace; }
"""

# Shared drawing code, run in the page.
JS = r"""
const NS = 'http://www.w3.org/2000/svg';
const COL = { N: '#5d7892', S: '#e39b00', V: '#d23c3c', F: '#8a55c9', Q: '#a9b3bd' };
const NAME = { N: '정상', S: '상심실', V: '심실', F: '융합', Q: '판정 보류' };
function el(tag, attrs, parent, text) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (text !== undefined) e.textContent = text;
  if (parent) parent.appendChild(e);
  return e;
}
function svg(id, w, h) {
  const s = el('svg', { width: w, height: h, viewBox: `0 0 ${w} ${h}` });
  document.getElementById(id).appendChild(s);
  return s;
}
// ECG paper and trace. Returns the mappings from sample and millivolts.
function strip(s, box, sig, fs, t0, opt = {}) {
  const { x, y, w, h } = box;
  let lo = opt.lo, hi = opt.hi;
  if (lo === undefined) { lo = Math.min(...sig); hi = Math.max(...sig); const pad = (hi - lo) * 0.12; lo -= pad; hi += pad; }
  const n = sig.length;
  const X = i => x + (i / (n - 1)) * w;
  const Y = v => y + h - ((v - lo) / (hi - lo)) * h;
  el('rect', { x, y, width: w, height: h, fill: '#fff7f7' }, s);
  const secs = n / fs;
  const gridT = opt.gridT || 0.04, majorT = opt.majorT || 0.2;
  if (!opt.coarse) {
    for (let t = 0; t <= secs + 1e-9; t += gridT) {
      const gx = x + (t / secs) * w;
      const major = Math.abs(t / majorT - Math.round(t / majorT)) < 1e-6;
      el('line', { x1: gx, x2: gx, y1: y, y2: y + h, stroke: major ? '#efb9b9' : '#fbe3e3', 'stroke-width': major ? 1 : 0.6 }, s);
    }
    for (let v = Math.ceil(lo / 0.1) * 0.1; v <= hi; v += 0.1) {
      const major = Math.abs(v / 0.5 - Math.round(v / 0.5)) < 1e-6;
      el('line', { x1: x, x2: x + w, y1: Y(v), y2: Y(v), stroke: major ? '#efb9b9' : '#fbe3e3', 'stroke-width': major ? 1 : 0.6 }, s);
    }
  } else {
    for (let t = 0; t <= secs + 1e-9; t += 1) {
      const gx = x + (t / secs) * w;
      el('line', { x1: gx, x2: gx, y1: y, y2: y + h, stroke: t % 5 === 0 ? '#efb9b9' : '#f8dada', 'stroke-width': 1 }, s);
    }
  }
  // Downsample by min/max per pixel column so peaks survive.
  let d = '';
  const cols = Math.min(n, Math.round(w * 2));
  for (let c = 0; c < cols; c++) {
    const a = Math.floor(c * n / cols), b = Math.max(a + 1, Math.floor((c + 1) * n / cols));
    let mn = Infinity, mx = -Infinity;
    for (let i = a; i < b; i++) { mn = Math.min(mn, sig[i]); mx = Math.max(mx, sig[i]); }
    const px = X((a + b - 1) / 2);
    d += (c ? 'L' : 'M') + px.toFixed(1) + ' ' + Y(mx).toFixed(1) + 'L' + px.toFixed(1) + ' ' + Y(mn).toFixed(1);
  }
  el('path', { d, fill: 'none', stroke: opt.color || '#1b2733', 'stroke-width': opt.width || 1.7, 'stroke-linejoin': 'round' }, s);
  el('rect', { x, y, width: w, height: h, fill: 'none', stroke: '#e3b3b3' }, s);
  const S = smp => X(smp - t0);
  return { X, Y, S, lo, hi };
}
function anchorAt(t, secs) { return t < 1e-9 ? 'start' : (t > secs - 1e-6 ? 'end' : 'middle'); }
function timeAxis(s, box, secs, t0s, step, fmt) {
  for (let t = 0; t <= secs + 1e-9; t += step) {
    const gx = box.x + (t / secs) * box.w;
    el('text', { x: gx, y: box.y + box.h + 24, 'text-anchor': anchorAt(t, secs), 'font-size': 15, fill: '#6b7885' }, s, fmt ? fmt(t0s + t) : (t0s + t).toFixed(0) + ' s');
  }
}
function pill(s, cx, cy, label, color, big) {
  const w = big ? 34 : 28, h = big ? 30 : 24;
  el('rect', { x: cx - w / 2, y: cy - h / 2, width: w, height: h, rx: 7, fill: color }, s);
  el('text', { x: cx, y: cy + (big ? 7 : 6), 'text-anchor': 'middle', 'font-size': big ? 19 : 16, 'font-weight': 800, fill: '#fff' }, s, label);
}
"""


def page(name, body, script, data):
    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><style>{CSS}</style></head>
<body><div class="page">{body}</div>
<script>const DATA = {json.dumps(data)};</script>
<script>{JS}</script>
<script>{script}</script></body></html>"""
    path = os.path.join(SRC, name + ".html")
    with open(path, "w") as f:
        f.write(html)
    png = os.path.join(IMG, name + ".png")
    subprocess.run(
        [CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
         "--window-size=1600,900", "--virtual-time-budget=3000", f"--screenshot={png}", "file://" + path],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("wrote", png)


def foot(left, right="live-ecg · 엔진 실제 출력"):
    return f'<div class="foot"><span>{left}</span><span>{right}</span></div>'


def legend(keys):
    names = {"N": "정상 N", "S": "상심실 S", "V": "심실 V", "F": "융합 F", "Q": "판정 보류"}
    cols = {"N": "#5d7892", "S": "#e39b00", "V": "#d23c3c", "F": "#8a55c9", "Q": "#a9b3bd"}
    return '<div class="legend">' + "".join(f'<span><i style="background:{cols[k]}"></i>{names[k]}</span>' for k in keys) + "</div>"


# ---------------------------------------------------------------- 01 overview
def overview():
    d = load("beats_208")
    tiles = [
        ("QRS 검출", "99.2% / 98.0%", "민감도 / 정밀도 · 318개 기록, 885시간"),
        ("심방세동", "90.5% / 98.8%", "민감도 / 정밀도 · 하루 오경보 2.2회"),
        ("심실세동 경보", "19 / 20 · 9초", "발생 탐지 / 지연 중앙값 · 봉인 데이터"),
        ("심정지 · 휴지", "100% · 99.2%", "MIT-BIH 봉인 데이터"),
        ("처리량", "≈19,000 채널", "CPU 코어 1개, 250 Hz 기준"),
        ("배포", "파일 하나", "표준 라이브러리만 사용 · 약 0.8 MB 바이너리"),
    ]
    cards = "".join(
        f'<div style="border:1px solid #e3e8ed;border-radius:18px;padding:22px 24px;background:#fbfcfd">'
        f'<div style="font-size:18px;color:#5b6875;font-weight:700">{t}</div>'
        f'<div style="font-size:36px;font-weight:800;margin-top:8px;letter-spacing:-.01em">{v}</div>'
        f'<div style="font-size:15px;color:#7a8794;margin-top:8px">{c}</div></div>' for t, v, c in tiles)
    body = f"""
<div class="kicker">live-ecg</div>
<h1>실시간 단일 유도 심전도 분석 엔진</h1>
<div class="sub">웨어러블 패치의 심전도를 박동 단위로 판독하고 부정맥을 실시간으로 알립니다.<br>
수백 채널을 처리하는 서버부터 Raspberry Pi·스마트폰까지, 같은 엔진 파일 하나로 동작합니다.</div>
<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:18px;margin-top:30px">{cards}</div>
<div id="fig" style="margin-top:22px"></div>
{foot("수치: 학습·조정에 쓰지 않은 봉인(TEST) 공개 데이터 · 파형: MIT-BIH 208")}"""
    script = """
const s = svg('fig', 1472, 150);
const r = strip(s, {x:0,y:18,w:1472,h:128}, DATA.signal.slice(0, Math.round(DATA.fs*6)), DATA.fs, DATA.from, {});
for (const b of DATA.beats) { if (b.s - DATA.from < DATA.fs*6) pill(s, r.S(b.s), 13, b.c, COL[b.c]); }
"""
    d = dict(d)
    page("01_overview", body, script, {"signal": d["signal"], "fs": d["fs"], "from": d["from"], "beats": d["beats"]})


# ---------------------------------------------------------- 02 beat classes
def beats():
    d = load("beats_208")
    n = len(d["beats"])
    body = f"""
<div class="kicker">박동 분류</div>
<h1>박동마다 정상 · 상심실 · 심실 · 융합을 판정합니다</h1>
<div class="sub">위: 엔진 판정과 심실 확률. 아래: 심장 전문의 주석. 10초 구간의 {n}개 박동이 모두 일치합니다.</div>
<div style="display:flex;justify-content:space-between;align-items:center;margin-top:18px">{legend("NVF")}
<div class="stat">엔진 판정 <b>{n}</b> 박동 · 주석 일치 <b>{n}/{n}</b></div></div>
<div id="fig" style="margin-top:8px"></div>
{foot("MIT-BIH Arrhythmia Database, 기록 208, 834–844초 · 기본 구성 (beats.clinical@4)")}"""
    script = """
const s = svg('fig', 1472, 560);
const box = {x:0, y:78, w:1472, h:380};
const r = strip(s, box, DATA.signal, DATA.fs, DATA.from, {});
el('text', {x:0, y:20, 'font-size':16, fill:'#5b6875', 'font-weight':700}, s, '엔진');
for (const b of DATA.beats) {
  const cx = r.S(b.s);
  pill(s, cx, 44, b.c, COL[b.c], true);
  el('text', {x:cx, y:72, 'text-anchor':'middle', 'font-size':13, fill:'#7a8794'}, s, 'V ' + (b.pv*100).toFixed(0) + '%');
  el('line', {x1:cx, x2:cx, y1:box.y, y2:box.y+box.h, stroke:COL[b.c], 'stroke-width':1.2, 'stroke-dasharray':'3 4', opacity:.55}, s);
}
const map = {N:'N',L:'N',R:'N',e:'N',j:'N',A:'S',a:'S',J:'S',S:'S',V:'V',E:'V',F:'F'};
el('text', {x:0, y:box.y+box.h+30, 'font-size':16, fill:'#5b6875', 'font-weight':700}, s, '전문가 주석');
for (const [smp, sym] of DATA.reference) {
  const cx = r.S(smp); const c = map[sym] || 'Q';
  el('circle', {cx, cy: box.y+box.h+52, r: 14, fill:'#fff', stroke: COL[c], 'stroke-width': 3}, s);
  el('text', {x:cx, y:box.y+box.h+58, 'text-anchor':'middle', 'font-size':16, 'font-weight':800, fill:COL[c]}, s, c);
}
timeAxis(s, {x:0,y:box.y,w:1472,h:box.h+70}, 10, DATA.from/DATA.fs, 1);
"""
    page("02_beat_classification", body, script, d)


# ----------------------------------------------------------------- 03 AF
def af():
    d = load("af_08455")
    fs = d["fs"]
    # Hour-scale timeline, binned.
    total = d["n_samples"] / fs
    bins = 480
    prob = [0.0] * bins
    cnt = [0] * bins
    state = [0] * bins
    for s0, s1, p, inaf in d["af"]:
        b = min(bins - 1, int(s1 / fs / total * bins))
        prob[b] += p
        cnt[b] += 1
        state[b] += 1 if inaf else 0
    tl = [{"p": prob[i] / cnt[i] if cnt[i] else None, "af": state[i] / cnt[i] if cnt[i] else 0} for i in range(bins)]
    ref = [[s / fs, e / fs, n] for s, e, n in d["rhythm_spans"]]
    beats = [b["s"] for b in d["beats"]]
    data = {"signal": d["signal"], "fs": fs, "from": d["from"], "beats": beats, "tl": tl, "total": total, "ref": ref}
    body = """
<div class="kicker">심방세동</div>
<h1>심방세동을 시작부터 끝까지 추적합니다</h1>
<div class="sub">박동 간격의 불규칙성과 심방 활동의 일관성으로 판정하고, 확인된 구간만 에피소드로 보고합니다.</div>
<div id="fig" style="margin-top:22px"></div>
""" + foot("MIT-BIH Atrial Fibrillation Database, 기록 08455 (10.2시간) · AFDB 봉인 결과: 민감도 90.5%, 정밀도 98.8%")
    script = """
const s = svg('fig', 1472, 610);
const T = {x:150, y:30, w:1322, h:150};
el('text', {x:0, y:T.y+18, 'font-size':17, 'font-weight':700, fill:'#33414e'}, s, 'AF 확률');
el('rect', {x:T.x, y:T.y, width:T.w, height:T.h, fill:'#fafbfc', stroke:'#e3e8ed'}, s);
const n = DATA.tl.length; let d = '';
DATA.tl.forEach((b, i) => { if (b.p === null) return; const x = T.x + i/(n-1)*T.w, y = T.y + T.h - b.p*T.h; d += (d ? 'L' : 'M') + x.toFixed(1) + ' ' + y.toFixed(1); });
el('path', {d, fill:'none', stroke:'#2f6db3', 'stroke-width':2}, s);
const bandY = T.y + T.h + 22;
el('text', {x:0, y:bandY+16, 'font-size':17, 'font-weight':700, fill:'#33414e'}, s, '엔진 판정');
el('text', {x:0, y:bandY+62, 'font-size':17, 'font-weight':700, fill:'#33414e'}, s, '전문가 주석');
DATA.tl.forEach((b, i) => { const x = T.x + i/n*T.w; el('rect', {x, y:bandY, width:T.w/n+0.6, height:26, fill: b.af > 0.5 ? '#2f6db3' : '#e8edf2'}, s); });
for (const [a, b, name] of DATA.ref) { const x0 = T.x + a/DATA.total*T.w, x1 = T.x + b/DATA.total*T.w; el('rect', {x:x0, y:bandY+46, width:Math.max(1, x1-x0), height:26, fill: name === 'AFIB' ? '#1f4f86' : '#e8edf2'}, s); }
for (let h = 0; h <= 10; h += 1) { const x = T.x + h*3600/DATA.total*T.w; el('text', {x, y:bandY+96, 'text-anchor':'middle', 'font-size':15, fill:'#6b7885'}, s, h + 'h'); }
el('text', {x:T.x + 3.12*3600/DATA.total*T.w + 14, y:T.y+T.h*0.62, 'font-size':17, fill:'#1f4f86', 'font-weight':800}, s, '← 3시간 7분: 심방세동 시작, 엔진도 같은 시점에 판정');
// The strip, with each RR interval written over it.
const B = {x:150, y:bandY+130, w:1322, h:200};
el('text', {x:0, y:B.y+18, 'font-size':17, 'font-weight':700, fill:'#33414e'}, s, '12초 파형');
el('text', {x:0, y:B.y+42, 'font-size':15, fill:'#6b7885'}, s, '4시간 지점');
const r = strip(s, B, DATA.signal, DATA.fs, DATA.from, {});
const bs = DATA.beats;
for (let i = 1; i < bs.length; i++) { const a = r.S(bs[i-1]), b = r.S(bs[i]); const ms = (bs[i]-bs[i-1])/DATA.fs*1000;
  el('text', {x:(a+b)/2, y:B.y-8, 'text-anchor':'middle', 'font-size':14, fill:'#2f6db3', 'font-weight':700}, s, ms.toFixed(0)); }
el('text', {x:B.x+B.w, y:B.y+B.h+26, 'text-anchor':'end', 'font-size':15, fill:'#6b7885'}, s, '숫자: 박동 간격(ms) — 불규칙하게 흔들립니다');
"""
    page("03_atrial_fibrillation", body, script, data)


# ----------------------------------------------------------------- 04 VF
def vf():
    d = load("vf_46")
    fs = d["fs"]
    onset = 13307.0
    eps = [e for e in d["vf_episodes"] if e[1] / fs >= onset - 30]
    alarm = eps[0][0] / fs + 4.0 if eps else None
    vfw = [[w[0] / fs, w[1]] for w in d["vf"] if d["from"] / fs <= w[0] / fs <= d["to"] / fs]
    data = {"signal": d["signal"], "fs": fs, "from": d["from"], "onset": onset, "alarm": alarm, "vf": vfw}
    lat = alarm - onset if alarm else 0
    body = f"""
<div class="kicker">심실세동 경보</div>
<h1>심실세동이 시작되고 {lat:.0f}초 만에 경보를 울립니다</h1>
<div class="sub">박동이 사라지는 리듬이라 박동 검출과 별개로 파형 자체의 모양과 스펙트럼을 매초 판정합니다.</div>
<div id="fig" style="margin-top:26px"></div>
{foot("Sudden Cardiac Death Holter Database, 기록 46 (봉인 데이터) · 봉인 20건 중 19건 탐지, 지연 중앙값 9초")}"""
    script = """
const s = svg('fig', 1472, 590);
const secs = DATA.signal.length / DATA.fs, t0 = DATA.from / DATA.fs;
const E = {x:0, y:20, w:1472, h:330};
const r = strip(s, E, DATA.signal, DATA.fs, DATA.from, {coarse:true, width:1.1});
const P = {x:0, y:400, w:1472, h:120};
el('rect', {x:P.x, y:P.y, width:P.w, height:P.h, fill:'#fafbfc', stroke:'#e3e8ed'}, s);
const PX = t => P.x + (t - t0)/secs*P.w, PY = p => P.y + P.h - p*P.h;
el('line', {x1:P.x, x2:P.x+P.w, y1:PY(0.8), y2:PY(0.8), stroke:'#d23c3c', 'stroke-dasharray':'6 5', 'stroke-width':1.5}, s);
el('text', {x:P.x+8, y:PY(0.8)+20, 'font-size':15, fill:'#d23c3c', 'font-weight':700}, s, '경보 기준 0.8');
let d = ''; DATA.vf.forEach(([t, p], i) => { d += (i ? 'L' : 'M') + PX(t).toFixed(1) + ' ' + PY(p).toFixed(1); });
el('path', {d, fill:'none', stroke:'#d23c3c', 'stroke-width':2.4}, s);
el('text', {x:P.x, y:P.y-10, 'font-size':17, 'font-weight':800, fill:'#33414e'}, s, 'VF 확률 (매초 갱신)');
const ox = PX(DATA.onset);
el('line', {x1:ox, x2:ox, y1:E.y, y2:P.y+P.h, stroke:'#17212b', 'stroke-width':2}, s);
el('text', {x:ox-8, y:E.y+22, 'text-anchor':'end', 'font-size':17, 'font-weight':800, fill:'#17212b'}, s, '심실세동 시작 (주석)');
if (DATA.alarm) { const ax = PX(DATA.alarm);
  el('line', {x1:ax, x2:ax, y1:E.y, y2:P.y+P.h, stroke:'#d23c3c', 'stroke-width':3}, s);
  el('rect', {x:ax+8, y:E.y+4, width:150, height:34, rx:8, fill:'#d23c3c'}, s);
  el('text', {x:ax+83, y:E.y+27, 'text-anchor':'middle', 'font-size':18, 'font-weight':800, fill:'#fff'}, s, '경보 +' + (DATA.alarm-DATA.onset).toFixed(0) + '초'); }
for (let t = 0; t <= secs + 1e-9; t += 10) el('text', {x:P.x + t/secs*P.w, y:P.y+P.h+24, 'text-anchor':anchorAt(t, secs), 'font-size':15, fill:'#6b7885'}, s, (t - (DATA.onset - t0) > 0 ? '+' : '') + (t - (DATA.onset - t0)).toFixed(0) + ' s');
el('text', {x:P.x+P.w, y:P.y+P.h+50, 'text-anchor':'end', 'font-size':15, fill:'#6b7885'}, s, '시간: 심실세동 시작(주석) 기준');
"""
    page("04_vf_alarm", body, script, data)


# ---------------------------------------------------------------- 05 pause
def pause():
    d = load("pause_232")
    fs = d["fs"]
    eps = [e for e in d["episodes"] if e[0] in ("pause", "asystole") and e[2] - e[1] > fs]
    data = {"signal": d["signal"], "fs": fs, "from": d["from"], "eps": eps, "beats": d["beats"]}
    longest = max((e[2] - e[1]) / fs for e in eps) if eps else 0
    body = f"""
<div class="kicker">휴지 · 심정지</div>
<h1>{longest:.1f}초 동안 멈춘 심장을 놓치지 않습니다</h1>
<div class="sub">2초 이상 박동이 없으면 휴지, 4초 이상이면 심정지로 보고합니다. 전극 탈락과 구분하기 위해 전극 상태를 함께 봅니다.</div>
<div id="fig" style="margin-top:28px"></div>
{foot("MIT-BIH Arrhythmia Database, 기록 232, 1700–1730초 · MIT-BIH 봉인 결과: 심정지 14/14, 휴지 91/92")}"""
    script = """
const s = svg('fig', 1472, 600);
const B = {x:0, y:70, w:1472, h:400};
const secs = DATA.signal.length / DATA.fs;
const r = strip(s, B, DATA.signal, DATA.fs, DATA.from, {gridT:0.2, majorT:1});
for (const [name, a, b] of DATA.eps) {
  if (name !== 'pause') continue;
  const x0 = r.S(a), x1 = r.S(b);
  el('rect', {x:x0, y:B.y, width:x1-x0, height:B.h, fill:'#d23c3c', opacity:.10}, s);
  el('line', {x1:x0, x2:x1, y1:B.y-22, y2:B.y-22, stroke:'#d23c3c', 'stroke-width':3}, s);
  el('text', {x:(x0+x1)/2, y:B.y-34, 'text-anchor':'middle', 'font-size':22, 'font-weight':800, fill:'#d23c3c'}, s, '박동 없음 ' + ((b-a)/DATA.fs).toFixed(1) + '초 → 휴지 · 심정지 경보');
}
for (const b of DATA.beats) el('circle', {cx:r.S(b.s), cy:B.y+B.h+22, r:6, fill:COL[b.c]}, s);
timeAxis(s, {x:0,y:B.y+26,w:1472,h:B.h}, secs, DATA.from/DATA.fs, 5);
const LY = B.y + B.h + 88; let lx = 0;
el('text', {x:lx, y:LY, 'font-size':15, fill:'#6b7885'}, s, '점: 엔진이 검출한 박동 —'); lx += 190;
for (const [k, t] of [['N','정상'],['S','상심실'],['Q','판정 보류']]) { el('circle', {cx:lx+6, cy:LY-5, r:6, fill:COL[k]}, s); el('text', {x:lx+18, y:LY, 'font-size':15, fill:'#6b7885'}, s, t); lx += 30 + t.length*15; }
"""
    page("05_pause_asystole", body, script, data)


# -------------------------------------------------------------- 06 quality
def quality():
    d = load("noise_118e00")
    data = {"signal": d["signal"], "fs": d["fs"], "from": d["from"], "q": d["quality"]}
    body = f"""
<div class="kicker">신호 품질</div>
<h1>믿을 수 있는 구간과 없는 구간을 샘플 단위로 가립니다</h1>
<div class="sub">잡음이 섞이면 품질 등급을 낮추고, 그 구간의 판정을 믿지 않도록 알립니다. 사람 판독자 4명과 비교한 판별력 AUC 0.996.</div>
<div style="margin-top:18px" class="legend"><span><i style="background:#2e9d62"></i>양호</span><span><i style="background:#e6a23c"></i>리듬만 신뢰</span><span><i style="background:#d23c3c"></i>사용 불가</span></div>
<div id="fig" style="margin-top:12px"></div>
{foot("MIT-BIH Noise Stress Test Database, 118e00 (전극 움직임 잡음 0 dB), 290–320초 · 품질 AUC는 BUT QDB 봉인 결과")}"""
    script = """
const s = svg('fig', 1472, 590);
const B = {x:0, y:20, w:1472, h:440};
const secs = DATA.signal.length / DATA.fs;
const r = strip(s, B, DATA.signal, DATA.fs, DATA.from, {gridT:0.2, majorT:1, width:1.3});
const C = ['#2e9d62', '#e6a23c', '#d23c3c'];
const q = DATA.q; const QY = B.y + B.h + 16;
for (let i = 0; i < q.length; i++) { const x0 = r.S(q[i][0]), x1 = i + 1 < q.length ? r.S(q[i+1][0]) : B.x + B.w; el('rect', {x:x0, y:QY, width:Math.max(0, x1-x0)+0.5, height:34, fill:C[q[i][1]]}, s); }
el('text', {x:r.S(DATA.from + 10*DATA.fs), y:B.y+30, 'text-anchor':'middle', 'font-size':20, 'font-weight':800, fill:'#d23c3c'}, s, '▼ 300초: 잡음 시작');
timeAxis(s, {x:0,y:QY+10,w:1472,h:34}, secs, DATA.from/DATA.fs, 5);
"""
    page("06_signal_quality", body, script, data)


# ---------------------------------------------------------- 07 review queue
def queue():
    d = load("beats_208")
    cl = [c for c in d["clusters"] if c["count"] >= 20][:6]
    data = {"clusters": cl, "fs": d["fs"]}
    body = """
<div class="kicker">심실 검토 큐</div>
<h1>수천 개의 박동을 모양별로 묶어 한 번에 판독합니다</h1>
<div class="sub">같은 모양의 박동을 하나의 형태로 묶고, 심실성일 가능성이 높은 형태부터 보여줍니다. 판독자는 대표 박동 하나를 보고 묶음 전체를 판정합니다.</div>
<div id="fig" style="margin-top:26px"></div>
""" + foot("MIT-BIH Arrhythmia Database, 기록 208 (30분) · 막대: 각 형태에 속한 박동의 전문가 주석 구성 · MIT-BIH 봉인 결과: 민감도 96.1%, 정밀도 89.2%")
    script = """
const s = svg('fig', 1472, 560);
const W = 350, H = 470, G = 24;
const refName = ['N','S','V','F','Q'];
DATA.clusters.forEach((c, i) => {
  const x = i * (W + G), y = 10;
  el('rect', {x, y, width:W, height:H, rx:16, fill:'#fbfcfd', stroke:'#e3e8ed'}, s);
  const ref = c.reference, tot = ref.reduce((a, b) => a + b, 0) || 1;
  let top = 0; for (let k = 1; k < 5; k++) if (ref[k] > ref[top]) top = k;
  const col = COL[refName[top]];
  el('text', {x:x+20, y:y+38, 'font-size':22, 'font-weight':800, fill:'#17212b'}, s, (i+1) + '순위 형태');
  el('text', {x:x+W-20, y:y+38, 'text-anchor':'end', 'font-size':18, fill:'#4a5866'}, s, c.count.toLocaleString() + ' 박동');
  const w = c.wave, lo = Math.min(...w), hi = Math.max(...w);
  const X = j => x + 24 + j/(w.length-1)*(W-48), Y = v => y + 70 + (1 - (v - lo)/(hi - lo || 1)) * 250;
  let d = ''; w.forEach((v, j) => d += (j ? 'L' : 'M') + X(j).toFixed(1) + ' ' + Y(v).toFixed(1));
  el('path', {d, fill:'none', stroke:col, 'stroke-width':3, 'stroke-linejoin':'round'}, s);
  let bx = x + 20; const bw = W - 40;
  el('text', {x:x+20, y:y+372, 'font-size':15, fill:'#6b7885'}, s, '이 형태에 묶인 박동의 전문가 주석');
  for (let k = 0; k < 5; k++) { if (!ref[k]) continue; const ww = ref[k]/tot*bw; el('rect', {x:bx, y:y+386, width:ww, height:20, fill:COL[refName[k]]}, s); bx += ww; }
  el('text', {x:x+20, y:y+440, 'font-size':20, 'font-weight':800, fill:col}, s, NAME[refName[top]] + ' ' + (ref[top]/tot*100).toFixed(0) + '%');
});
"""
    page("07_review_queue", body, script, data)


# --------------------------------------------------------- 08 performance
def performance():
    rows = [
        ("QRS 검출", "318개 기록 · 885시간", 99.2, 98.0),
        ("심정지", "MIT-BIH", 100.0, 100.0),
        ("휴지", "MIT-BIH", 99.2, 99.2),
        ("서맥", "MIT-BIH", 99.5, 95.7),
        ("빈맥", "MIT-BIH", 97.6, 100.0),
        ("심방세동", "AFDB, 끝단 평가", 90.5, 98.8),
        ("심실세동 경보", "sddb 발생 19/20", 95.0, None),
        ("심실 박동", "MIT-BIH", 95.1, 83.3),
        ("심실 박동", "INCART II 유도", 88.8, 91.6),
        ("이단맥", "MIT-BIH", 84.7, 98.7),
    ]
    weak = [
        ("상심실 박동", "MIT-BIH", 25.5, 25.9),
        ("심실빈맥 경보", "MIT-BIH", 50.0, 22.9),
    ]
    data = {"rows": rows, "weak": weak}
    body = """
<div class="kicker">성능</div>
<h1>학습에 한 번도 쓰지 않은 데이터로 잰 성능입니다</h1>
<div style="display:flex;justify-content:space-between;align-items:flex-end">
<div class="sub">모든 수치는 공개 데이터베이스의 봉인(TEST) 구역에서 측정했습니다. 잘 되는 것과 아직 부족한 것을 함께 공개합니다.</div>
<div class="legend" style="flex:none;margin-left:24px"><span><i style="background:#2f6db3"></i>민감도</span><span><i style="background:#8fb3dc"></i>정밀도</span></div></div>
<div id="fig" style="margin-top:20px"></div>
""" + foot("전체 표와 근거: reports/PERFORMANCE.md · 심실세동 경보의 정밀도 대신 오경보: 돌연사 환자 하루 5.0회, 그 외 616시간 4회")
    script = """
const s = svg('fig', 1472, 590);
const L = 330, BW = 1060, RH = 40;
function row(y, name, sub, se, pp, weak) {
  el('text', {x:0, y:y+17, 'font-size':20, 'font-weight':800, fill: weak ? '#a0522d' : '#17212b'}, s, name);
  el('text', {x:0, y:y+37, 'font-size':14, fill:'#7a8794'}, s, sub);
  const bar = (v, yy, c) => { if (v === null) return; el('rect', {x:L, y:yy, width:BW, height:14, rx:7, fill:'#eef2f5'}, s);
    el('rect', {x:L, y:yy, width:BW*v/100, height:14, rx:7, fill:c}, s);
    el('text', {x:L+BW+14, y:yy+13, 'font-size':16, 'font-weight':700, fill:'#33414e'}, s, v.toFixed(1) + '%'); };
  bar(se, y+2, weak ? '#c9793f' : '#2f6db3'); bar(pp, y+20, weak ? '#e8b48d' : '#8fb3dc');
}
DATA.rows.forEach((r, i) => row(i*(RH+4), r[0], r[1], r[2], r[3], false));
const y0 = DATA.rows.length*(RH+4) + 18;
el('text', {x:0, y:y0, 'font-size':16, 'font-weight':800, fill:'#a0522d'}, s, '아직 부족한 부분');
DATA.weak.forEach((r, i) => row(y0 + 12 + i*(RH+4), r[0], r[1], r[2], r[3], true));
"""
    page("08_performance", body, script, data)


# -------------------------------------------------------- 09 architecture
def architecture():
    body = """
<div class="kicker">구조</div>
<h1>엔진 전체도, 단계 하나도 바꿔 끼울 수 있습니다</h1>
<div class="sub">호스트는 고정된 표준 인터페이스(C ABI 1.1)만 압니다. 엔진 파일을 통째로 바꾸거나, 채널마다 단계 구현을 이름으로 골라 새 버전과 이전 버전을 나란히 비교합니다.</div>
<div id="fig" style="margin-top:26px"></div>
""" + foot("dist/ecg_engine.rs · dist/ecg.h · tools/ecg_conformance.c")
    script = """
const s = svg('fig', 1472, 560);
function box(x, y, w, h, title, sub, fill, stroke, swap) {
  el('rect', {x, y, width:w, height:h, rx:14, fill, stroke, 'stroke-width': swap ? 2.5 : 1.5, 'stroke-dasharray': swap ? '0' : '0'}, s);
  el('text', {x:x+w/2, y:y+h/2 - (sub ? 4 : -6), 'text-anchor':'middle', 'font-size':19, 'font-weight':800, fill:'#17212b'}, s, title);
  if (sub) el('text', {x:x+w/2, y:y+h/2+20, 'text-anchor':'middle', 'font-size':14, fill:'#4a5866'}, s, sub);
  if (swap) { el('rect', {x:x+w-52, y:y-12, width:62, height:24, rx:12, fill:'#d23c3c'}, s); el('text', {x:x+w-21, y:y+5, 'text-anchor':'middle', 'font-size':13, 'font-weight':800, fill:'#fff'}, s, '교체'); }
}
function arrow(x1, y1, x2, y2) { el('line', {x1, y1, x2, y2, stroke:'#8a97a4', 'stroke-width':2.2, 'marker-end':'url(#a)'}, s); }
const defs = el('defs', {}, s); const m = el('marker', {id:'a', markerWidth:10, markerHeight:10, refX:8, refY:5, orient:'auto'}, defs); el('path', {d:'M0 0L10 5L0 10z', fill:'#8a97a4'}, m);
// engine frame
el('rect', {x:210, y:10, width:1052, height:470, rx:22, fill:'#f6f8fb', stroke:'#b9c6d3', 'stroke-width':2, 'stroke-dasharray':'8 6'}, s);
el('text', {x:230, y:44, 'font-size':18, 'font-weight':800, fill:'#2f6db3'}, s, '엔진 파일 하나 — ecg_engine.rs → libecg (교체 가능)');
box(0, 200, 170, 90, '심전도 입력', 'mV, 채널별', '#fff', '#c9d3dc');
arrow(170, 245, 240, 245);
box(240, 200, 180, 90, '전처리', '필터 뱅크 · 품질 감시', '#fff', '#c9d3dc');
arrow(420, 245, 460, 245);
box(460, 200, 170, 90, 'QRS 검출', 'qrs.pt@1', '#fff', '#d23c3c', true);
arrow(630, 245, 670, 245);
box(670, 200, 190, 90, '박동 분류', 'beats.clinical@4 · @3', '#fff', '#d23c3c', true);
arrow(860, 245, 900, 245);
box(900, 90, 170, 80, '심방세동', 'af.logistic@1', '#fff', '#d23c3c', true);
box(900, 205, 170, 80, '상심실 런', 'svrun.rate@2 · @1', '#fff', '#d23c3c', true);
box(900, 320, 170, 80, '리듬 에피소드', '휴지 · 서맥 · 빈맥 · 런', '#fff', '#c9d3dc');
box(470, 345, 170, 80, '심실세동', 'vf.spectral@2 · linear@1', '#fff', '#d23c3c', true);
arrow(350, 290, 470, 385);
arrow(860, 245, 900, 130); arrow(860, 245, 900, 360);
arrow(1070, 245, 1110, 245); arrow(1070, 130, 1110, 215); arrow(1070, 360, 1110, 275);
el('path', {d:'M640 405 L 640 440 L 1180 440 L 1180 312', fill:'none', stroke:'#8a97a4', 'stroke-width':2.2, 'marker-end':'url(#a)'}, s);
box(1110, 190, 140, 110, '이벤트', 'kind · code', '#eaf2fb', '#2f6db3');
arrow(1262, 245, 1302, 245);
box(1302, 170, 170, 150, '호스트', '서버 · 앱 · Pi', '#fff', '#c9d3dc');
// interface bar
el('rect', {x:210, y:500, width:1262, height:56, rx:14, fill:'#17212b'}, s);
el('text', {x:230, y:535, 'font-size':17, 'font-weight':700, fill:'#fff'}, s, 'ecg.h  ·  ecg_channel_create → push → poll → status → destroy');
el('text', {x:1452, y:535, 'text-anchor':'end', 'font-size':16, fill:'#b7c7d6'}, s, 'stages = "vf=vf.linear@1;beats=beats.clinical@3"');
"""
    page("09_architecture", body, script, {})


# ------------------------------------------------------------ 10 terminal
def terminal():
    with open(os.path.join(DATA, "term_conformance.txt")) as f:
        conf = f.read().rstrip().split("\n")
    with open(os.path.join(DATA, "term_vfalarm.txt")) as f:
        alarm = f.read().rstrip()
    def fmt(line):
        line = line.replace("&", "&amp;").replace("<", "&lt;")
        if line.strip().startswith("ok"):
            return '<span style="color:#58d68d">  ok</span>' + line.split("ok", 1)[1]
        if "CONFORMANT" in line:
            return f'<span style="color:#58d68d;font-weight:700">{line}</span>'
        if line.strip().startswith("info"):
            return f'<span style="color:#9fb3c8">{line}</span>'
        return line
    n_ok = sum(1 for l in conf if l.strip().startswith("ok"))
    keep = conf[:2] + conf[2:10] + ["  …"] + conf[-7:]
    lines = "<br>".join(fmt(l) for l in keep)
    body = f"""
<div class="kicker">인터페이스 검증</div>
<h1>새 엔진 파일은 올리기 전에 자동으로 검증합니다</h1>
<div class="sub">적합성 검사기는 엔진을 파일 경로로만 불러 규약 {n_ok}개 항목을 확인하고, 엔진에 든 단계 구현을 하나씩 모두 돌려봅니다.</div>
<div style="margin-top:24px;border-radius:14px;overflow:hidden;box-shadow:0 12px 40px rgba(20,30,40,.18)">
<div style="background:#2b3440;height:40px;display:flex;align-items:center;gap:9px;padding-left:18px">
<i style="width:13px;height:13px;border-radius:50%;background:#ff5f57;display:inline-block"></i>
<i style="width:13px;height:13px;border-radius:50%;background:#febc2e;display:inline-block"></i>
<i style="width:13px;height:13px;border-radius:50%;background:#28c840;display:inline-block"></i>
<span style="color:#aab7c4;font-size:15px;margin-left:14px" class="mono">dist/ecg_conformance dist/libecg.dylib</span></div>
<div class="mono" style="background:#161c24;color:#dce6f0;font-size:17.5px;line-height:1.62;padding:20px 26px;height:560px;overflow:hidden">
<span style="color:#7fb2e5">$</span> dist/ecg_conformance dist/libecg.dylib<br>{lines}</div></div>
{foot("tools/ecg_conformance.c · 실제 실행 출력")}"""
    page("10_conformance", body, "", {})


# ----------------------------------------------------------- 11 deployment
def deployment():
    targets = [("Raspberry Pi 5", "aarch64 Linux (musl, 정적)", "0.82 MB"),
               ("32비트 ARM Linux", "armv7 (musl, 정적)", "0.81 MB"),
               ("저사양 PC · 서버", "x86-64 Linux (musl, 정적)", "0.84 MB"),
               ("iPhone · iPad", "aarch64 iOS", "0.6 MB"),
               ("Mac", "Apple Silicon · Intel", "0.6 / 0.7 MB")]
    trs = "".join(f'<tr><td style="padding:14px 0;font-weight:800;font-size:21px">{a}</td><td style="color:#5b6875;font-size:18px">{b}</td><td style="text-align:right;font-weight:800;font-size:21px">{c}</td></tr>' for a, b, c in targets)
    body = f"""
<div class="kicker">배포</div>
<h1>서버 한 대로 수만 채널, 작은 기기에서도 그대로</h1>
<div class="sub">순수 Rust, 외부 의존성 없음. 샘플마다 메모리를 할당하지 않는 스트리밍 구조라 채널이 늘어도 비용이 거의 그대로입니다.</div>
<div style="display:grid;grid-template-columns:1fr 1.15fr;gap:40px;margin-top:34px">
<div>
<div style="border:1px solid #e3e8ed;border-radius:18px;padding:28px;background:#fbfcfd">
<div style="font-size:18px;color:#5b6875;font-weight:700">CPU 코어 1개 처리량 (250 Hz)</div>
<div style="font-size:64px;font-weight:800;margin-top:6px">≈19,000<span style="font-size:28px"> 채널</span></div>
<div style="font-size:17px;color:#5b6875;margin-top:6px">샘플당 211 ns · 1,000채널에 코어의 5.3%</div></div>
<div style="border:1px solid #e3e8ed;border-radius:18px;padding:28px;background:#fbfcfd;margin-top:18px">
<div style="font-size:18px;color:#5b6875;font-weight:700">채널당 메모리</div>
<div style="font-size:48px;font-weight:800;margin-top:6px">≈146 KB</div>
<div style="font-size:17px;color:#5b6875;margin-top:6px">1,000채널 ≈ 146 MB (구조체 크기로 계산)</div></div></div>
<table style="width:100%;border-collapse:collapse">{trs.replace('<tr>', '<tr style="border-bottom:1px solid #e6eaee">')}</table></div>
<div style="margin-top:34px">
<div style="font-size:18px;color:#5b6875;font-weight:700;margin-bottom:12px">샘플 하나를 처리하는 비용 (단일 코어, 227 ns)</div>
<div style="display:flex;height:54px;border-radius:12px;overflow:hidden;font-size:17px;font-weight:700;color:#fff">
<div style="width:4.7%;background:#8fb3dc;display:flex;align-items:center;justify-content:center">11</div>
<div style="width:22.7%;background:#5d8fc6;display:flex;align-items:center;padding-left:14px">품질 감시 52 ns</div>
<div style="width:72.6%;background:#2f6db3;display:flex;align-items:center;padding-left:14px">QRS 검출 · 박동 분류 · 리듬 · 심방세동 · 심실세동 165 ns</div></div>
<div style="font-size:15px;color:#7a8794;margin-top:8px">맨 앞: 필터 뱅크 11 ns</div></div>
{foot("측정: Apple M1 Ultra, 4스레드 · 기기별 바이너리 크기는 교차 컴파일 결과 · Raspberry Pi 등 실기기 처리량은 아직 측정 전")}"""
    page("11_deployment", body, "", {})


for f in [overview, beats, af, vf, pause, quality, queue, performance, architecture, terminal, deployment]:
    f()
