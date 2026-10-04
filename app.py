from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf

from PIL import Image
from skimage.feature import hog


BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
MODELS_DIR = BASE_DIR / "models"


def render_html(content, sidebar=False, **kwargs):
    """Render HTML in Streamlit without Markdown treating indented lines as code blocks."""
    cleaned = "\n".join(
        line.strip() for line in content.splitlines() if line.strip()
    )
    target = st.sidebar if sidebar else st
    target.markdown(cleaned, unsafe_allow_html=True)



def render_html_sidebar(content, **kwargs):
    render_html(content, sidebar=True)


# ---------------------------------------------------------------------------
# Hero แบบ Infographic ผลไม้
# วาดด้วย SVG + CSS + JavaScript ภายใน iframe (ไม่ใช้ WebGL)
# - variant="full"    : ใช้หน้า Overview (การ์ดผลไม้ 8 ชนิด + ขั้นตอนการทำงาน + ตัวเลขสถิติ)
# - variant="compact" : ใช้หน้าอื่น (แถวผลไม้ขนาดเล็ก)
# แก้ข้อความ สี หรือรูปผลไม้ได้ในตัวแปร HERO_HTML ด้านล่าง
# ---------------------------------------------------------------------------

HERO_HTML = r"""<!DOCTYPE html>
<html lang="th">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Thai:wght@400;500;600;700;800&display=swap');

* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; background: transparent;
    font-family: 'Noto Sans Thai', 'Segoe UI', Tahoma, sans-serif; color: #1f3d2a; }
body { padding: 4px 4px 22px 4px; }

/* ---------------- hero shell ---------------- */
.hero { position: relative; overflow: hidden; display: flex; align-items: center;
    height: 420px; border-radius: 28px; border: 1px solid #d3e4cc;
    background: linear-gradient(120deg, #dff0da 0%, #eef6e6 52%, #fdf1de 100%);
    box-shadow: 0 14px 40px rgba(47, 107, 67, .12); }
body.compact .hero { height: 250px; }

.blob { position: absolute; border-radius: 50%; filter: blur(8px); pointer-events: none; }
.b1 { width: 420px; height: 420px; right: -90px; top: -150px;
    background: radial-gradient(circle, rgba(79,154,95,.30), transparent 68%);
    animation: drift 14s ease-in-out infinite; }
.b2 { width: 300px; height: 300px; left: 26%; bottom: -170px;
    background: radial-gradient(circle, rgba(242,176,74,.28), transparent 70%);
    animation: drift 17s ease-in-out infinite reverse; }
@keyframes drift { 0%,100% { transform: translate(0,0); } 50% { transform: translate(-22px, 16px); } }

.leaf { position: absolute; width: 26px; height: 26px; opacity: .55; pointer-events: none;
    animation: sway 7s ease-in-out infinite; }
.leaf svg { width: 100%; height: 100%; display: block; }
.l1 { left: 4%; top: 10%; }
.l2 { left: 38%; top: 6%; animation-delay: -2s; transform: scale(.8); }
.l3 { left: 30%; bottom: 8%; animation-delay: -4s; transform: scale(.7); }
@keyframes sway { 0%,100% { transform: translateY(0) rotate(-10deg); } 50% { transform: translateY(10px) rotate(14deg); } }

/* ---------------- copy ---------------- */
.copy { position: relative; z-index: 2; flex: 0 0 41%; padding: 0 0 0 2.6rem; }
body.compact .copy { flex-basis: 44%; }

.eyebrow { display: inline-flex; align-items: center; gap: .4rem; color: #2f6b43;
    background: rgba(255,255,255,.8); border: 1px solid #cfe3c8; border-radius: 999px;
    padding: .28rem .9rem .28rem .65rem; font-size: .85rem; font-weight: 700; margin-bottom: .8rem; }
.eyebrow svg { width: 16px; height: 16px; }

.title { background: linear-gradient(90deg, #1f3d2a, #3f8a55); -webkit-background-clip: text;
    background-clip: text; -webkit-text-fill-color: transparent; font-size: 3rem; font-weight: 800;
    line-height: 1.3; margin: 0; }
body.compact .title { font-size: 2.35rem; }

.text { color: #52604f; font-size: 1.02rem; line-height: 1.75; margin-top: .45rem; max-width: 470px; }

.chips { display: flex; gap: .55rem; flex-wrap: wrap; margin-top: 1rem; }
body.compact .chips, body.compact .seg { display: none; }
.chip { background: rgba(255,255,255,.82); border: 1px solid #dbe8d4; border-radius: 14px;
    padding: .35rem .7rem; min-width: 74px; text-align: center; box-shadow: 0 4px 10px rgba(47,107,67,.07); }
.chip b { display: block; font-size: 1.28rem; font-weight: 800; color: #2f6b43; line-height: 1.15; }
.chip span { font-size: .72rem; color: #667267; }

.seg { display: inline-flex; margin-top: .9rem; padding: 3px; border-radius: 999px;
    background: rgba(255,255,255,.75); border: 1px solid #d3e4cc; }
.seg button { font: inherit; font-size: .82rem; font-weight: 700; color: #4d6a55; cursor: pointer;
    border: 0; background: transparent; padding: .3rem .85rem; border-radius: 999px; transition: all .2s; }
.seg button:hover { background: rgba(79,154,95,.12); }
.seg button.on { background: linear-gradient(135deg, #4f9a5f, #2f6b43); color: #fff;
    box-shadow: 0 4px 10px rgba(47,107,67,.28); }
.seg button[data-mode="rotten"].on { background: linear-gradient(135deg, #d4695d, #a8382d);
    box-shadow: 0 4px 10px rgba(168,56,45,.28); }

/* ---------------- infographic ---------------- */
.info { position: relative; z-index: 2; flex: 1; padding: 0 2rem 0 .6rem; }
.gridwrap { position: relative; }
.grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
body.compact .grid { grid-template-columns: repeat(8, 1fr); gap: 6px; }

.card { position: relative; text-align: center; cursor: pointer; user-select: none;
    background: rgba(255,255,255,.86); border: 1.5px solid #dbe8d4; border-radius: 18px;
    padding: 8px 4px 8px; box-shadow: 0 6px 16px rgba(35,55,40,.08);
    transition: transform .25s, box-shadow .25s, border-color .6s, background .6s; }
.card:hover { transform: translateY(-5px); box-shadow: 0 14px 26px rgba(47,107,67,.16); }
.card.rotten { border-color: #efc4bc; background: rgba(255,247,245,.9); }

.fruit { height: 78px; display: flex; align-items: center; justify-content: center; }
body.compact .fruit { height: 44px; }
.fruit svg { width: 78px; height: 78px; overflow: visible;
    filter: saturate(1) sepia(0) brightness(1) contrast(1) drop-shadow(0 5px 5px rgba(0,0,0,.12));
    transition: filter 1s ease; animation: bob 4.2s ease-in-out infinite; animation-delay: calc(var(--i) * -.55s); }
body.compact .fruit svg { width: 46px; height: 46px; }
.card.rotten .fruit svg { filter: saturate(.55) sepia(.5) brightness(.72) contrast(1.06) drop-shadow(0 5px 5px rgba(0,0,0,.18)); }
@keyframes bob { 0%,100% { transform: translateY(0) rotate(-1.5deg); } 50% { transform: translateY(-4px) rotate(1.5deg); } }
.card:hover .fruit svg { animation: wiggle .55s ease; }
@keyframes wiggle { 0%,100% { transform: rotate(0); } 25% { transform: rotate(-9deg) scale(1.06); } 75% { transform: rotate(9deg) scale(1.06); } }

.name { margin-top: 2px; line-height: 1.2; }
.name b { display: block; font-size: .82rem; color: #1f3d2a; }
.name small { font-size: .7rem; color: #7a867a; }
body.compact .name { display: none; }

.tag { display: inline-block; margin-top: 5px; min-width: 46px; padding: 1px 9px; border-radius: 999px;
    font-size: .72rem; font-weight: 800; background: #e3f1de; color: #2f6b43; transition: all .4s; }
.card.rotten .tag { background: #fbe4e0; color: #b13e32; }
.tag.pend { background: #eef0ee; color: #98a198; }
body.compact .tag { margin-top: 3px; font-size: .62rem; min-width: 0; padding: 0 6px; }

/* mould, bruises, smell */
.rot circle, .rot ellipse { opacity: 0; transform-box: fill-box; transform-origin: center; transform: scale(.2);
    transition: opacity .8s ease, transform .8s ease; transition-delay: var(--d, 0s); }
.card.rotten .rot circle, .card.rotten .rot ellipse { opacity: .74; transform: scale(1); }
.mold circle { opacity: 0; transition: opacity .9s ease; transition-delay: calc(var(--d, 0s) + .4s); }
.card.rotten .mold circle { opacity: .85; }
.stink { opacity: 0; transition: opacity .6s; }
.card.rotten .stink { opacity: 1; }
.stink path { stroke-dasharray: 5 4; animation: rise 1.6s linear infinite; }
@keyframes rise { to { stroke-dashoffset: -18; } }

/* scanner */
.scanner { position: absolute; top: -8px; bottom: -8px; left: 0; width: 56px; margin-left: -28px;
    pointer-events: none; opacity: 0; z-index: 5; border-radius: 14px;
    background: linear-gradient(90deg, transparent, rgba(110,210,140,.20) 35%, rgba(110,210,140,.55) 50%, rgba(110,210,140,.20) 65%, transparent); }
.scanner::after { content: ""; position: absolute; top: 0; bottom: 0; left: 50%; width: 2px; margin-left: -1px;
    background: #38b36a; box-shadow: 0 0 12px 3px rgba(56,179,106,.55); }
.scanner::before { content: "กำลังสแกน"; position: absolute; top: -4px; left: 50%; transform: translateX(-50%);
    font-size: .66rem; font-weight: 700; background: #2f6b43; color: #fff; border-radius: 999px;
    padding: 1px 9px; white-space: nowrap; z-index: 2; }
body.compact .scanner::before { display: none; }

/* flow strip */
.flow { display: flex; align-items: center; justify-content: center; gap: 8px; margin-top: 12px;
    background: rgba(255,255,255,.72); border: 1px solid #dbe8d4; border-radius: 16px; padding: 8px 10px; }
body.compact .flow { display: none; }
.step { display: flex; align-items: center; gap: 7px; font-size: .82rem; font-weight: 700; color: #2f6b43; white-space: nowrap; }
.step .ico { width: 30px; height: 30px; border-radius: 10px; background: linear-gradient(135deg, #4f9a5f, #2f6b43);
    display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 10px rgba(47,107,67,.25); }
.step .ico svg { width: 18px; height: 18px; }
.step.mid .ico { background: linear-gradient(135deg, #f2b04a, #d9861a); box-shadow: 0 4px 10px rgba(217,134,26,.28); animation: pulse 2s ease-in-out infinite; }
.step.res .ico { background: linear-gradient(135deg, #4f9a5f 50%, #c4574c 50%); }
@keyframes pulse { 0%,100% { transform: scale(1); } 50% { transform: scale(1.1); } }
.arrow { position: relative; flex: 1; min-width: 22px; max-width: 70px; height: 2px; background: repeating-linear-gradient(90deg, #a9ceA3 0 5px, transparent 5px 9px); }
.arrow i { position: absolute; top: -3px; left: 0; width: 8px; height: 8px; border-radius: 50%; background: #4f9a5f; animation: travel 1.8s linear infinite; }
.arrow:nth-of-type(2) i { animation-delay: -.9s; }
@keyframes travel { from { left: 0; opacity: 0; } 15% { opacity: 1; } 85% { opacity: 1; } to { left: calc(100% - 8px); opacity: 0; } }

.note { position: absolute; right: 16px; bottom: 7px; z-index: 3; font-size: .62rem; color: #8a968a; }
body.compact .note { display: none; }

/* ---------------- small screens ---------------- */
@media (max-width: 760px) {
    .hero { flex-direction: column; align-items: stretch; justify-content: center; height: 440px; padding: 1.1rem 0 0; }
    body.compact .hero { height: 330px; }
    .copy { flex: 0 0 auto; padding: 0 1.2rem; }
    .title { font-size: 2.2rem; }
    body.compact .title { font-size: 1.9rem; }
    .text { font-size: .92rem; }
    .chips, .seg, .flow, .note, .name small { display: none; }
    .info { padding: .7rem 1rem 0; }
    .grid { gap: 6px; }
    .fruit { height: 54px; } .fruit svg { width: 52px; height: 52px; }
    body.compact .grid { grid-template-columns: repeat(4, 1fr); }
    body.compact .fruit { height: 40px; } body.compact .fruit svg { width: 40px; height: 40px; }
    body.compact .hero { height: 258px; padding-top: .8rem; }
    body.compact .text, body.compact .tag { display: none; }
    .name b { font-size: .7rem; } .leaf { display: none; }
}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation: none !important; transition-duration: .01s !important; }
}
</style>
</head>
<body class="__VARIANT__">

<div class="hero">
    <div class="blob b1"></div><div class="blob b2"></div>
    <div class="leaf l1"><svg viewBox="0 0 24 24"><path d="M3 21C3 10 10 3 21 3c0 11-7 18-18 18z" fill="#6fb27a"/><path d="M5 19C9 13 13 9 18 6" stroke="#2f6b43" stroke-width="1.3" fill="none"/></svg></div>
    <div class="leaf l2"><svg viewBox="0 0 24 24"><path d="M3 21C3 10 10 3 21 3c0 11-7 18-18 18z" fill="#8cc58f"/></svg></div>
    <div class="leaf l3"><svg viewBox="0 0 24 24"><path d="M3 21C3 10 10 3 21 3c0 11-7 18-18 18z" fill="#f2b04a" opacity=".8"/></svg></div>

    <div class="copy">
        <div class="eyebrow"><span id="eyeIcon"></span> __EYEBROW__</div>
        <div class="title">__TITLE__</div>
        <div class="text">__TEXT__</div>

        <div class="chips">
            <div class="chip"><b data-to="8">0</b><span>ชนิดผลไม้</span></div>
            <div class="chip"><b data-to="16">0</b><span>กลุ่มข้อมูลย่อย</span></div>
            <div class="chip"><b data-to="3193">0</b><span>ภาพ</span></div>
            <div class="chip"><b data-to="3">0</b><span>โมเดล</span></div>
        </div>

        <div class="seg" id="seg">
            <button data-mode="auto" class="on">สลับอัตโนมัติ</button>
            <button data-mode="fresh">ผลไม้สด</button>
            <button data-mode="rotten">ผลไม้เน่า</button>
        </div>
    </div>

    <div class="info">
        <div class="gridwrap" id="gridwrap">
            <div class="grid" id="grid"></div>
            <div class="scanner" id="scanner"></div>
        </div>
        <div class="flow">
            <div class="step"><span class="ico"><svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 16V5"/><path d="M7 9l5-5 5 5"/><path d="M5 19h14"/></svg></span>อัปโหลดภาพ</div>
            <div class="arrow"><i></i></div>
            <div class="step mid"><span class="ico"><svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/><circle cx="12" cy="12" r="1.6" fill="#fff"/></svg></span>AI วิเคราะห์</div>
            <div class="arrow"><i></i></div>
            <div class="step res"><span class="ico"><svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M4 12l4 4 6-8"/></svg></span>สด / เน่า</div>
        </div>
    </div>
    <div class="note">ภาพประกอบเพื่ออธิบายแนวคิดของระบบ</div>
</div>

<script>
(function () {
    var VARIANT = '__VARIANT__';
    var reduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    /* ---- eyebrow icon ---- */
    var ICONS = {
        eco: '<svg viewBox="0 0 24 24" fill="#2f6b43"><path d="M3 21C3 10 10 3 21 3c0 11-7 18-18 18z"/></svg>',
        search: '<svg viewBox="0 0 24 24" fill="none" stroke="#2f6b43" stroke-width="2.6" stroke-linecap="round"><circle cx="10.5" cy="10.5" r="6"/><path d="M15 15l6 6"/></svg>',
        analytics: '<svg viewBox="0 0 24 24" fill="#2f6b43"><rect x="3" y="12" width="4.5" height="9" rx="1.2"/><rect x="9.8" y="6" width="4.5" height="15" rx="1.2"/><rect x="16.5" y="9" width="4.5" height="12" rx="1.2"/></svg>'
    };
    document.getElementById('eyeIcon').innerHTML = ICONS['__ICON__'] || ICONS.eco;

    /* ---- fruit drawings (viewBox 0 0 100 100) ---- */
    var LEAF = '<linearGradient id="lf" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#7cc576"/><stop offset="1" stop-color="#2f7d3a"/></linearGradient>';
    function rg(id, c0, c1, c2, cx, cy) {
        return '<radialGradient id="' + id + '" cx="' + (cx || .36) + '" cy="' + (cy || .3) + '" r=".85"><stop offset="0" stop-color="' + c0 +
               '"/><stop offset=".55" stop-color="' + c1 + '"/><stop offset="1" stop-color="' + c2 + '"/></radialGradient>';
    }
    var DRAW = {
        apple: {
            defs: rg('g-apple', '#ff8a7a', '#e02f2f', '#9b1219'),
            clip: '<path d="M50 30C40 21 15 25 16 52C17 75 35 92 50 85C65 92 83 75 84 52C85 25 60 21 50 30Z"/>',
            body: '<path d="M50 30C40 21 15 25 16 52C17 75 35 92 50 85C65 92 83 75 84 52C85 25 60 21 50 30Z" fill="url(#g-apple)"/>' +
                  '<path d="M50 30C50 22 52 16 57 11" stroke="#6b3f1d" stroke-width="3.5" fill="none" stroke-linecap="round"/>' +
                  '<path d="M55 20C62 8 77 9 80 15C72 24 61 25 55 20Z" fill="url(#lf)"/>' +
                  '<ellipse cx="32" cy="46" rx="5.5" ry="11" fill="#fff" opacity=".35" transform="rotate(18 32 46)"/>',
            spots: [[62, 58, 9], [34, 68, 7], [52, 42, 5], [72, 72, 5]]
        },
        banana: {
            defs: '<linearGradient id="g-banana" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#ffe873"/><stop offset=".55" stop-color="#f5bd10"/><stop offset="1" stop-color="#c98a00"/></linearGradient>',
            clip: '<path d="M20 32C24 68 54 90 86 72C90 69 87 64 82 66C58 78 40 62 36 32C35 24 19 24 20 32Z"/>',
            body: '<path d="M20 32C24 68 54 90 86 72C90 69 87 64 82 66C58 78 40 62 36 32C35 24 19 24 20 32Z" fill="url(#g-banana)"/>' +
                  '<rect x="19" y="18" width="15" height="11" rx="3" fill="#7a5a22" transform="rotate(-6 26 24)"/>' +
                  '<circle cx="85" cy="69" r="3.6" fill="#4a3216"/>' +
                  '<path d="M27 40C33 66 52 80 78 70" stroke="#fff7b8" stroke-width="3" fill="none" opacity=".55" stroke-linecap="round"/>',
            spots: [[33, 58, 6], [52, 76, 6], [70, 73, 4], [28, 44, 4]]
        },
        grape: {
            defs: rg('g-grape', '#c58ae6', '#7b2fb0', '#3f0f69'),
            clip: [[32, 38], [50, 36], [68, 38], [41, 55], [59, 55], [50, 72]].map(function (p) { return '<circle cx="' + p[0] + '" cy="' + p[1] + '" r="12"/>'; }).join(''),
            body: '<path d="M50 28C50 20 52 15 56 11" stroke="#6b3f1d" stroke-width="3.2" fill="none" stroke-linecap="round"/>' +
                  '<path d="M54 20C60 8 76 10 80 18C72 26 60 26 54 20Z" fill="url(#lf)"/>' +
                  [[32, 38], [50, 36], [68, 38], [41, 55], [59, 55], [50, 72]].map(function (p) {
                      return '<circle cx="' + p[0] + '" cy="' + p[1] + '" r="12" fill="url(#g-grape)"/>' +
                             '<ellipse cx="' + (p[0] - 4) + '" cy="' + (p[1] - 5) + '" rx="2.6" ry="4" fill="#fff" opacity=".38" transform="rotate(25 ' + (p[0] - 4) + ' ' + (p[1] - 5) + ')"/>';
                  }).join(''),
            spots: [[50, 38, 6], [36, 56, 5], [62, 58, 5], [68, 40, 4]]
        },
        guava: {
            defs: rg('g-guava', '#eef7b0', '#b4d868', '#6aa23a') + '<radialGradient id="bl-guava"><stop offset="0" stop-color="#ff9aa2" stop-opacity=".6"/><stop offset="1" stop-color="#ff9aa2" stop-opacity="0"/></radialGradient>',
            clip: '<ellipse cx="50" cy="58" rx="31" ry="30"/>',
            body: '<ellipse cx="50" cy="58" rx="31" ry="30" fill="url(#g-guava)"/>' +
                  '<ellipse cx="66" cy="68" rx="17" ry="15" fill="url(#bl-guava)"/>' +
                  [[38, 50], [58, 46], [46, 66], [64, 58], [34, 66], [52, 78], [70, 70]].map(function (p) { return '<circle cx="' + p[0] + '" cy="' + p[1] + '" r="1.2" fill="#5c8a2c" opacity=".5"/>'; }).join('') +
                  '<path d="M50 29C50 24 50 20 52 16" stroke="#6b3f1d" stroke-width="3" fill="none" stroke-linecap="round"/>' +
                  '<path d="M50 28C38 17 26 21 24 27C34 33 46 32 50 28Z" fill="url(#lf)"/>' +
                  '<path d="M50 28C62 17 74 21 76 27C66 33 54 32 50 28Z" fill="url(#lf)"/>' +
                  '<ellipse cx="34" cy="48" rx="5" ry="9" fill="#fff" opacity=".35" transform="rotate(20 34 48)"/>',
            spots: [[60, 60, 8], [38, 70, 6], [48, 44, 5], [70, 72, 5]]
        },
        jujube: {
            defs: rg('g-jujube', '#ee8250', '#bb3420', '#6a180e'),
            clip: '<ellipse cx="38" cy="60" rx="15" ry="26" transform="rotate(-18 38 60)"/><ellipse cx="62" cy="56" rx="14" ry="24" transform="rotate(16 62 56)"/>',
            body: '<ellipse cx="38" cy="60" rx="15" ry="26" fill="url(#g-jujube)" transform="rotate(-18 38 60)"/>' +
                  '<ellipse cx="62" cy="56" rx="14" ry="24" fill="url(#g-jujube)" transform="rotate(16 62 56)"/>' +
                  '<path d="M33 52C36 60 36 68 34 76M43 50C46 60 45 70 42 78" stroke="#5a1608" stroke-width="1" fill="none" opacity=".3"/>' +
                  '<path d="M60 46C63 54 63 62 62 70M70 46C72 54 71 62 68 68" stroke="#5a1608" stroke-width="1" fill="none" opacity=".3"/>' +
                  '<path d="M30 36L28 28M68.5 34L70 26" stroke="#6b3f1d" stroke-width="3" stroke-linecap="round"/>' +
                  '<path d="M70 27C76 19 87 21 88 27C81 32 73 32 70 27Z" fill="url(#lf)"/>' +
                  '<ellipse cx="31" cy="54" rx="3" ry="8" fill="#fff" opacity=".35" transform="rotate(-16 31 54)"/>' +
                  '<ellipse cx="57" cy="48" rx="3" ry="7" fill="#fff" opacity=".35" transform="rotate(16 57 48)"/>',
            spots: [[38, 62, 6], [64, 60, 6], [34, 46, 4], [60, 44, 4]]
        },
        orange: {
            defs: rg('g-orange', '#ffc873', '#ff8c1a', '#d2580a'),
            clip: '<circle cx="50" cy="57" r="32"/>',
            body: '<circle cx="50" cy="57" r="32" fill="url(#g-orange)"/>' +
                  (function () { var s = ''; for (var y = 36; y < 86; y += 9) for (var x = 30 + ((y / 9) % 2) * 4; x < 74; x += 9) { if ((x - 50) * (x - 50) + (y - 57) * (y - 57) < 780) s += '<circle cx="' + x + '" cy="' + y + '" r=".9" fill="#b9480a" opacity=".35"/>'; } return s; })() +
                  '<circle cx="50" cy="25" r="4" fill="#5b7f2b"/>' +
                  '<path d="M52 22C60 10 76 12 79 19C71 27 58 27 52 22Z" fill="url(#lf)"/>' +
                  '<ellipse cx="34" cy="45" rx="6" ry="11" fill="#fff" opacity=".33" transform="rotate(25 34 45)"/>',
            spots: [[62, 60, 9], [36, 70, 7], [50, 42, 5], [70, 74, 5]]
        },
        pomegranate: {
            defs: rg('g-pom', '#ff7468', '#c8102e', '#6e0919'),
            clip: '<ellipse cx="50" cy="59" rx="32" ry="29"/><path d="M38 34L40 20L46 27L50 17L54 27L60 20L62 34Z"/>',
            body: '<path d="M38 34L40 20L46 27L50 17L54 27L60 20L62 34Z" fill="#a50e2a"/>' +
                  '<ellipse cx="50" cy="59" rx="32" ry="29" fill="url(#g-pom)"/>' +
                  '<ellipse cx="50" cy="34" rx="8" ry="3.6" fill="#7a0a1c"/>' +
                  '<ellipse cx="33" cy="48" rx="5.5" ry="10" fill="#fff" opacity=".33" transform="rotate(22 33 48)"/>' +
                  '<path d="M68 52C74 60 74 70 66 78" stroke="#5a0616" stroke-width="1.2" fill="none" opacity=".35"/>',
            spots: [[62, 60, 8], [36, 72, 6], [52, 46, 5], [72, 70, 5]]
        },
        strawberry: {
            defs: rg('g-straw', '#ff7f86', '#e3263a', '#9c0f21', .4, .3),
            clip: '<path d="M50 90C22 72 16 46 22 36C30 28 43 32 50 35C57 32 70 28 78 36C84 46 78 72 50 90Z"/>',
            body: '<path d="M50 90C22 72 16 46 22 36C30 28 43 32 50 35C57 32 70 28 78 36C84 46 78 72 50 90Z" fill="url(#g-straw)"/>' +
                  [[34, 46], [46, 48], [58, 46], [68, 48], [28, 56], [40, 58], [52, 58], [64, 58], [74, 58], [34, 68], [46, 68], [58, 68], [68, 68], [42, 78], [54, 78]].map(function (p) {
                      return '<ellipse cx="' + p[0] + '" cy="' + p[1] + '" rx="1.5" ry="2.2" fill="#ffe27a"/>'; }).join('') +
                  '<path d="M50 38C44 30 34 30 30 34C36 41 44 41 50 38Z" fill="url(#lf)"/>' +
                  '<path d="M50 38C56 30 66 30 70 34C64 41 56 41 50 38Z" fill="url(#lf)"/>' +
                  '<path d="M50 38C46 30 48 24 50 21C52 24 54 30 50 38Z" fill="url(#lf)"/>' +
                  '<path d="M50 38C40 42 33 48 35 50C42 48 48 44 50 38Z" fill="url(#lf)"/>' +
                  '<path d="M50 38C60 42 67 48 65 50C58 48 52 44 50 38Z" fill="url(#lf)"/>' +
                  '<path d="M50 24L52 14" stroke="#3f8a3f" stroke-width="3" stroke-linecap="round"/>' +
                  '<ellipse cx="31" cy="50" rx="3.5" ry="8" fill="#fff" opacity=".3" transform="rotate(14 31 50)"/>',
            spots: [[58, 62, 8], [36, 60, 6], [50, 74, 5], [66, 50, 4]]
        }
    };
    var FRUITS = [
        ['apple', 'Apple', 'แอปเปิล'], ['banana', 'Banana', 'กล้วย'], ['grape', 'Grape', 'องุ่น'], ['guava', 'Guava', 'ฝรั่ง'],
        ['jujube', 'Jujube', 'พุทรา'], ['orange', 'Orange', 'ส้ม'], ['pomegranate', 'Pomegranate', 'ทับทิม'], ['strawberry', 'Strawberry', 'สตรอว์เบอร์รี']
    ];

    function fruitSVG(k) {
        var f = DRAW[k];
        var rot = f.spots.map(function (s, j) {
            return '<circle cx="' + s[0] + '" cy="' + s[1] + '" r="' + s[2] + '" fill="#4a2a12" style="--d:' + (j * .15) + 's"/>';
        }).join('');
        var mold = f.spots.slice(0, 3).map(function (s, j) {
            return '<circle cx="' + (s[0] + 2) + '" cy="' + (s[1] - 2) + '" r="1.8" fill="#e9efe1" style="--d:' + (j * .15) + 's"/>' +
                   '<circle cx="' + (s[0] - 3) + '" cy="' + (s[1] + 2) + '" r="1.3" fill="#d9e3d0" style="--d:' + (j * .15) + 's"/>';
        }).join('');
        return '<svg viewBox="0 0 100 100" aria-hidden="true"><defs>' + LEAF + f.defs + '<clipPath id="c-' + k + '">' + f.clip + '</clipPath></defs>' +
               '<g>' + f.body + '</g>' +
               '<g clip-path="url(#c-' + k + ')"><g class="rot">' + rot + '</g><g class="mold">' + mold + '</g></g>' +
               '<g class="stink" fill="none" stroke="#8a9a3c" stroke-width="2" stroke-linecap="round">' +
               '<path d="M13 34C9 28 17 24 13 17"/><path d="M87 34C83 28 91 24 87 17"/></g></svg>';
    }

    /* ---- build cards ---- */
    var grid = document.getElementById('grid');
    var cards = FRUITS.map(function (f, i) {
        var el = document.createElement('div');
        el.className = 'card';
        el.style.setProperty('--i', i);
        el.innerHTML = '<div class="fruit">' + fruitSVG(f[0]) + '</div>' +
                       '<div class="name"><b>' + f[1] + '</b><small>' + f[2] + '</small></div>' +
                       '<span class="tag">สด</span>';
        el.title = f[1] + ' / ' + f[2];
        grid.appendChild(el);
        return el;
    });
    var cols = VARIANT === 'compact' && window.innerWidth > 760 ? 8 : 4;
    var tags = cards.map(function (c) { return c.querySelector('.tag'); });

    function setState(i, rotten) {
        cards[i].classList.toggle('rotten', rotten);
        tags[i].classList.remove('pend');
        tags[i].textContent = rotten ? 'เน่า' : 'สด';
    }

    /* ---- scan sweep ---- */
    var scanner = document.getElementById('scanner');
    var wrap = document.getElementById('gridwrap');
    var timers = [];
    function sweep(states) {
        timers.forEach(clearTimeout); timers = [];
        if (reduced) { states.forEach(function (s, i) { setState(i, s); }); return; }
        var dur = 1700;
        var w = wrap.clientWidth;
        tags.forEach(function (t) { t.classList.add('pend'); t.textContent = '…'; });
        scanner.animate([
            { transform: 'translateX(0)', opacity: 0 },
            { opacity: 1, offset: .08 },
            { opacity: 1, offset: .92 },
            { transform: 'translateX(' + w + 'px)', opacity: 0 }
        ], { duration: dur, easing: 'linear', fill: 'none' });
        cards.forEach(function (c, i) {
            var colIdx = i % cols;
            var frac = (colIdx + .5) / cols;
            timers.push(setTimeout(function () { setState(i, states[i]); }, dur * frac));
        });
    }

    function randomMix() {
        var s;
        do { s = cards.map(function () { return Math.random() < .45; }); }
        while (s.filter(Boolean).length < 2 || s.filter(Boolean).length > 6);
        return s;
    }

    /* ---- modes ---- */
    var mode = 'auto', autoTimer = null, visible = true;
    function startAuto() {
        stopAuto();
        autoTimer = setInterval(function () { if (visible && !document.hidden && mode === 'auto') sweep(randomMix()); }, 5600);
    }
    function stopAuto() { if (autoTimer) { clearInterval(autoTimer); autoTimer = null; } }

    var seg = document.getElementById('seg');
    seg.addEventListener('click', function (e) {
        var b = e.target.closest('button'); if (!b) return;
        mode = b.dataset.mode;
        Array.prototype.forEach.call(seg.children, function (x) { x.classList.toggle('on', x === b); });
        if (mode === 'auto') { sweep(randomMix()); startAuto(); }
        else { stopAuto(); sweep(cards.map(function () { return mode === 'rotten'; })); }
    });
    cards.forEach(function (c, i) {
        c.addEventListener('click', function () {
            mode = 'manual'; stopAuto();
            Array.prototype.forEach.call(seg.children, function (x) { x.classList.remove('on'); });
            setState(i, !c.classList.contains('rotten'));
        });
    });

    if (window.IntersectionObserver) new IntersectionObserver(function (en) { visible = en[0].isIntersecting; }).observe(document.querySelector('.hero'));
    if (!reduced) { setTimeout(function () { if (mode === 'auto') { sweep(randomMix()); startAuto(); } }, 1500); }

    /* ---- count-up chips ---- */
    var nums = document.querySelectorAll('.chip b');
    Array.prototype.forEach.call(nums, function (n) {
        var to = +n.dataset.to;
        if (reduced) { n.textContent = to.toLocaleString('en-US'); return; }
        var t0 = performance.now(), dur = 1300;
        (function step(now) {
            var p = Math.min(1, (now - t0) / dur);
            var e = 1 - Math.pow(1 - p, 3);
            n.textContent = Math.round(to * e).toLocaleString('en-US');
            if (p < 1) requestAnimationFrame(step);
        })(t0);
    });
})();
</script>
</body>
</html>
"""


def render_hero(icon, eyebrow, title, text, variant="compact"):
    html = (
        HERO_HTML
        .replace("__VARIANT__", variant)
        .replace("__ICON__", icon)
        .replace("__EYEBROW__", eyebrow)
        .replace("__TITLE__", title)
        .replace("__TEXT__", text)
    )
    components.html(html, height=450 if variant == "full" else 280)


FRUIT_LIST = [
    ("Apple", "แอปเปิล", "#e02f2f"),
    ("Banana", "กล้วย", "#f5bd10"),
    ("Grape", "องุ่น", "#7b2fb0"),
    ("Guava", "ฝรั่ง", "#a9d35e"),
    ("Jujube", "พุทรา", "#bb3420"),
    ("Orange", "ส้ม", "#ff8c1a"),
    ("Pomegranate", "ทับทิม", "#c8102e"),
    ("Strawberry", "สตรอว์เบอร์รี", "#e3263a"),
]


def fruit_marquee_html():
    items = "".join(
        f'<span class="marquee-item"><i style="--c:{c}"></i>{en} <small>{th}</small></span>'
        for en, th, c in FRUIT_LIST
    )
    return f'<div class="marquee"><div class="marquee-track">{items}{items}</div></div>'


def gauge_html(confidence, has_probability):
    if not has_probability:
        return (
            '<div class="result-confidence" style="margin-top:0.8rem;">'
            'HOG + SVM ให้ผลเป็นคลาสเท่านั้น ไม่มีค่าความน่าจะเป็น'
            '</div>'
        )
    return (
        '<div class="result-confidence">ความแม่นยำ</div>'
        f'<div class="ring" style="--p:{confidence * 100:.2f};"><span>{confidence * 100:.1f}%</span></div>'
    )


def fruit_pills_html():
    return "".join(
        f'<span class="pill"><i style="--c:{c}"></i>{en}</span>'
        for en, th, c in FRUIT_LIST
    )


st.set_page_config(
    page_title="FreshFruit AI | ตรวจสอบผลไม้สด/เน่า",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)


render_html(
    """
    <style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Thai:wght@400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,500,1,0&display=swap');

.icon {
    font-family: 'Material Symbols Rounded';
    font-weight: normal;
    font-style: normal;
    font-size: 1.25em;
    line-height: 1;
    letter-spacing: normal;
    text-transform: none;
    white-space: nowrap;
    display: inline-block;
    vertical-align: -0.2em;
    -webkit-font-feature-settings: 'liga';
    font-feature-settings: 'liga';
    -webkit-font-smoothing: antialiased;
}

:root {
    --green-900: #1f3d2a;
    --green-700: #2f6b43;
    --green-500: #4f9a5f;
    --green-100: #e6f2e4;
    --cream: #f7f8f2;
    --text-soft: #64705f;
    --red-600: #c4574c;
    --red-100: #fbecea;
}

html, body, [class*="css"], .stApp, button, input, textarea, select {
    font-family: 'Noto Sans Thai', 'Segoe UI', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 12% 8%, rgba(160, 205, 150, 0.22), transparent 38%),
        radial-gradient(circle at 92% 4%, rgba(255, 214, 170, 0.20), transparent 34%),
        var(--cream);
}

.block-container {
    padding-top: 2.2rem;
    padding-bottom: 3rem;
    max-width: 1180px;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes floaty {
    0%, 100% { transform: translateY(0) rotate(-6deg); }
    50%      { transform: translateY(-10px) rotate(4deg); }
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #eaf3e4 0%, #f4f7ee 55%, #fbf3e6 100%);
    border-right: 1px solid #dbe6d3;
}

[data-testid="stSidebar"] p {
    color: var(--text-soft);
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    background: rgba(255, 255, 255, 0.65);
    border: 1px solid #dfe8d8;
    border-radius: 14px;
    padding: 0.65rem 0.9rem;
    margin-bottom: 0.45rem;
    width: 100%;
    transition: all 0.2s ease;
    cursor: pointer;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: white;
    border-color: #b9d3b3;
    transform: translateX(4px);
    box-shadow: 0 4px 14px rgba(47, 107, 67, 0.10);
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background: linear-gradient(135deg, #3f8a55, #2f6b43);
    border-color: #2f6b43;
    box-shadow: 0 6px 18px rgba(47, 107, 67, 0.28);
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p {
    color: white;
    font-weight: 700;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label p {
    font-size: 1.1rem;
    font-weight: 600;
    color: #35503b;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child {
    display: none;
}

.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.6rem 0 1.2rem 0;
}

.sidebar-logo-icon {
    width: 52px;
    height: 52px;
    border-radius: 16px;
    background: linear-gradient(135deg, #ff8a7a, #e0524a);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.7rem;
    box-shadow: 0 8px 18px rgba(224, 82, 74, 0.30);
}

.sidebar-logo-title {
    font-size: 1.7rem;
    font-weight: 800;
    color: var(--green-900);
    line-height: 1.1;
}

.sidebar-logo-sub {
    font-size: 0.95rem;
    color: var(--text-soft);
    margin-top: 0.15rem;
}

.sidebar-menu-label {
    font-size: 0.85rem;
    font-weight: 800;
    color: #7f8f7f;
    margin: 0.4rem 0 0.7rem 0.2rem;
}

.sidebar-footer {
    color: #7b857d;
    font-size: 0.9rem;
    line-height: 1.7;
}

/* ---------- Hero ---------- */
.hero {
    position: relative;
    overflow: hidden;
    padding: 2.8rem 3rem;
    border-radius: 28px;
    background: linear-gradient(120deg, #dff0da 0%, #eef6e6 50%, #fdf1de 100%);
    border: 1px solid #d3e4cc;
    margin-bottom: 1.8rem;
    box-shadow: 0 14px 40px rgba(47, 107, 67, 0.10);
    animation: fadeUp 0.6s ease both;
}

.hero::before {
    content: "";
    position: absolute;
    width: 320px;
    height: 320px;
    right: -80px;
    top: -120px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(79, 154, 95, 0.28), transparent 70%);
}

.hero::after {
    content: "eco";
    font-family: 'Material Symbols Rounded';
    font-feature-settings: 'liga';
    position: absolute;
    right: 3rem;
    bottom: 1rem;
    font-size: 6rem;
    line-height: 1;
    letter-spacing: normal;
    color: rgba(63, 138, 85, 0.35);
    animation: floaty 5s ease-in-out infinite;
    filter: drop-shadow(0 10px 14px rgba(0, 0, 0, 0.12));
}

.hero-eyebrow {
    display: inline-block;
    color: var(--green-700);
    background: rgba(255, 255, 255, 0.75);
    border: 1px solid #cfe3c8;
    border-radius: 999px;
    padding: 0.3rem 0.95rem;
    font-size: 0.85rem;
    font-weight: 700;
    margin-bottom: 0.9rem;
}

.hero-title {
    background: linear-gradient(90deg, #1f3d2a, #3f8a55);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3.2rem;
    font-weight: 800;
    line-height: 1.3;
    margin: 0;
}

.hero-text {
    color: #52604f;
    font-size: 1.1rem;
    line-height: 1.8;
    max-width: 640px;
    margin-top: 0.8rem;
}

/* ---------- Section headings ---------- */
.section-label {
    display: inline-block;
    color: var(--green-700);
    background: var(--green-100);
    border-radius: 999px;
    padding: 0.2rem 0.8rem;
    font-size: 0.8rem;
    font-weight: 800;
    margin-bottom: 0.45rem;
}

.section-title {
    color: var(--green-900);
    font-size: 1.85rem;
    font-weight: 750;
    margin-bottom: 1.1rem;
    line-height: 1.4;
}

.section-title::after {
    content: "";
    display: block;
    width: 54px;
    height: 4px;
    border-radius: 4px;
    margin-top: 0.35rem;
    background: linear-gradient(90deg, #4f9a5f, #f2b04a);
}

/* ---------- Cards ---------- */
.info-card {
    position: relative;
    background: white;
    border: 1px solid #e3e9de;
    border-radius: 20px;
    padding: 1.5rem;
    height: 100%;
    box-shadow: 0 6px 20px rgba(35, 55, 40, 0.06);
    transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
    animation: fadeUp 0.6s ease both;
}

.info-card:hover {
    transform: translateY(-5px);
    border-color: #b9d6b2;
    box-shadow: 0 16px 34px rgba(47, 107, 67, 0.15);
}

.info-card-title {
    color: var(--green-700);
    font-size: 1.15rem;
    font-weight: 750;
    margin-bottom: 0.55rem;
}

.info-card-text {
    color: #5f6b60;
    font-size: 0.98rem;
    line-height: 1.75;
}

.workflow-number {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 42px;
    height: 42px;
    padding: 0 0.9rem;
    border-radius: 14px;
    background: linear-gradient(135deg, #4f9a5f, #2f6b43);
    color: white;
    font-size: 0.95rem;
    font-weight: 800;
    box-shadow: 0 6px 14px rgba(47, 107, 67, 0.25);
}

.workflow-title {
    color: var(--green-900);
    font-size: 1.15rem;
    font-weight: 750;
    margin-top: 0.8rem;
}

.workflow-text {
    color: #667267;
    font-size: 0.95rem;
    line-height: 1.65;
    margin-top: 0.2rem;
}

/* ---------- Result ---------- */
.result-fresh, .result-rotten {
    position: relative;
    overflow: hidden;
    border-radius: 24px;
    padding: 2rem 1.8rem;
    text-align: center;
    animation: fadeUp 0.5s ease both;
}

.result-fresh {
    background: linear-gradient(145deg, #e9f7ea, #d4efd8);
    border: 1px solid #b9dfc0;
    box-shadow: 0 14px 34px rgba(79, 154, 95, 0.25);
}

.result-rotten {
    background: linear-gradient(145deg, #fdf0ee, #f7d9d4);
    border: 1px solid #efc0b9;
    box-shadow: 0 14px 34px rgba(196, 87, 76, 0.22);
}

.result-label {
    font-size: 2.3rem;
    font-weight: 850;
}

.result-fresh .result-label { color: var(--green-700); }
.result-rotten .result-label { color: var(--red-600); }

.result-confidence {
    color: #687269;
    font-size: 1rem;
    margin-top: 0.4rem;
}

.result-percent {
    font-size: 3.2rem;
    font-weight: 850;
    margin-top: 0.2rem;
}

.result-fresh .result-percent { color: var(--green-700); }
.result-rotten .result-percent { color: var(--red-600); }

.model-badge {
    display: inline-block;
    background: linear-gradient(135deg, #e6f2e4, #f6efd9);
    color: #3f6a47;
    border: 1px solid #cfe1c9;
    border-radius: 999px;
    padding: 0.4rem 1rem;
    font-size: 0.9rem;
    font-weight: 700;
    margin-bottom: 0.8rem;
}

.upload-title {
    color: var(--green-900);
    font-size: 1.2rem;
    font-weight: 750;
    margin-bottom: 0.3rem;
}

.small-note {
    color: #747d76;
    font-size: 0.9rem;
    line-height: 1.6;
    margin-top: 1rem;
}

/* ---------- Streamlit widgets ---------- */
div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e3e9de;
    border-top: 4px solid #4f9a5f;
    padding: 1.1rem 1.2rem;
    border-radius: 18px;
    box-shadow: 0 6px 18px rgba(35, 55, 40, 0.06);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
    box-shadow: 0 14px 28px rgba(47, 107, 67, 0.14);
}

div[data-testid="stMetricLabel"] {
    color: #6e776f;
}

div[data-testid="stMetricValue"] {
    color: var(--green-700);
    font-weight: 800;
}

.stButton > button {
    border-radius: 14px;
    border: none;
    background: linear-gradient(135deg, #4f9a5f, #2f6b43);
    color: white;
    font-weight: 700;
    font-size: 1.05rem;
    min-height: 3rem;
    box-shadow: 0 8px 20px rgba(47, 107, 67, 0.28);
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #5aa86b, #357a4d);
    color: white;
    transform: translateY(-2px);
    box-shadow: 0 12px 26px rgba(47, 107, 67, 0.36);
}

.stButton > button:active {
    transform: translateY(0);
}

[data-testid="stFileUploader"] section {
    background: rgba(255, 255, 255, 0.8);
    border: 2px dashed #a9ceA3;
    border-radius: 18px;
    transition: all 0.2s ease;
}

[data-testid="stFileUploader"] section:hover {
    background: white;
    border-color: #4f9a5f;
}

div[data-baseweb="select"] > div {
    border-radius: 14px;
    border-color: #d5e3cf;
    background: white;
}

[data-testid="stImage"] img {
    border-radius: 20px;
    box-shadow: 0 10px 28px rgba(35, 55, 40, 0.14);
}

[data-testid="stDataFrame"] {
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 6px 18px rgba(35, 55, 40, 0.07);
    border: 1px solid #e3e9de;
}

div[data-testid="stProgress"] > div > div > div > div {
    background: linear-gradient(90deg, #4f9a5f, #f2b04a);
}

div[data-testid="stAlert"] {
    border-radius: 14px;
}

hr {
    border-color: #dbe6d3 !important;
}

footer {
    visibility: hidden;
}

    
/* ================= extra flair ================= */
html { scroll-behavior: smooth; }
::selection { background: #bfe3bf; color: #1f3d2a; }
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: #b9d3b3; border-radius: 10px; border: 2px solid transparent; background-clip: padding-box; }
::-webkit-scrollbar-thumb:hover { background: #8fbf8f; background-clip: padding-box; }

.block-container { animation: fadeUp 0.55s ease both; }

/* scrolling fruit ribbon */
.marquee {
    overflow: hidden;
    border-radius: 16px;
    background: linear-gradient(90deg, #e6f2e4, #fdf1de);
    border: 1px solid #d3e4cc;
    padding: 0.6rem 0;
    margin: 0.2rem 0 1.3rem 0;
    -webkit-mask-image: linear-gradient(90deg, transparent, #000 5%, #000 95%, transparent);
    mask-image: linear-gradient(90deg, transparent, #000 5%, #000 95%, transparent);
}
.marquee-track { display: flex; gap: 2.4rem; width: max-content; animation: marquee 34s linear infinite; }
.marquee:hover .marquee-track { animation-play-state: paused; }
.marquee-item { display: inline-flex; align-items: center; gap: 0.5rem; font-weight: 700; color: #2f6b43; white-space: nowrap; font-size: 1.05rem; }
.marquee-item small { color: #7a867a; font-weight: 500; font-size: 0.9rem; }
.marquee-item i { width: 11px; height: 11px; border-radius: 50%; background: var(--c); display: inline-block; box-shadow: 0 0 0 3px rgba(255,255,255,0.7); }
@keyframes marquee { to { transform: translateX(-50%); } }

/* sidebar fruit pills */
.pill-wrap { margin-bottom: 0.4rem; }
.pill { display: inline-flex; align-items: center; gap: 0.4rem; background: rgba(255,255,255,0.72); border: 1px solid #dfe8d8;
        border-radius: 999px; padding: 0.12rem 0.65rem; margin: 0 0.3rem 0.38rem 0; font-size: 0.92rem; color: #35503b; transition: all 0.2s; }
.pill:hover { background: white; transform: translateY(-2px); box-shadow: 0 4px 10px rgba(47,107,67,0.12); }
.pill i { width: 9px; height: 9px; border-radius: 50%; background: var(--c); display: inline-block; }

/* result card: pop-in + confidence ring */
@property --p { syntax: '<number>'; inherits: false; initial-value: 0; }
@keyframes popIn { 0% { opacity: 0; transform: scale(0.85) translateY(14px); } 60% { transform: scale(1.03); } 100% { opacity: 1; transform: scale(1); } }
@keyframes fillRing { from { --p: 0; } }
.result-fresh, .result-rotten { animation: popIn 0.6s cubic-bezier(0.2, 1, 0.3, 1) both; }
.ring { --p: 0; --ringc: #2f6b43; position: relative; width: 150px; height: 150px; border-radius: 50%; margin: 0.7rem auto 0.2rem auto;
        display: grid; place-items: center; background: conic-gradient(var(--ringc) calc(var(--p) * 1%), rgba(0,0,0,0.09) 0);
        animation: fillRing 1.4s cubic-bezier(0.2, 0.8, 0.2, 1) both; }
.ring::before { content: ""; position: absolute; inset: 13px; border-radius: 50%; background: rgba(255,255,255,0.92); }
.ring span { position: relative; font-size: 2.1rem; font-weight: 850; color: var(--ringc); }
.result-rotten .ring { --ringc: #c4574c; }

/* button shimmer */
.stButton > button { position: relative; overflow: hidden; }
.stButton > button::after { content: ""; position: absolute; top: 0; bottom: 0; width: 40%; left: -60%;
        background: linear-gradient(100deg, transparent, rgba(255,255,255,0.35), transparent); transform: skewX(-20deg); }
.stButton > button:hover::after { left: 130%; transition: left 0.7s ease; }

    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_data
def load_comparison():
    return pd.read_csv(
        ASSETS_DIR / "model_comparison.csv"
    )


@st.cache_resource
def load_models():

    cnn = tf.keras.models.load_model(
        MODELS_DIR / "custom_cnn_best.keras"
    )

    mobilenet = tf.keras.models.load_model(
        MODELS_DIR / "mobilenetv2_best.keras"
    )

    svm = joblib.load(
        MODELS_DIR / "hog_svm_model.pkl"
    )

    return cnn, mobilenet, svm


df = load_comparison()
cnn, mobilenet, svm = load_models()


def predict_cnn(image):

    image = image.resize(
        (160, 160)
    )

    image = np.array(
        image,
        dtype=np.float32
    )

    image = np.expand_dims(
        image,
        axis=0
    )

    probability = cnn.predict(
        image,
        verbose=0
    )[0][0]

    return float(probability)


def predict_mobilenet(image):

    image = image.resize(
        (224, 224)
    )

    image = np.array(
        image,
        dtype=np.float32
    )

    image = np.expand_dims(
        image,
        axis=0
    )

    probability = mobilenet.predict(
        image,
        verbose=0
    )[0][0]

    return float(probability)


def predict_hog_svm(image):

    image = image.convert(
        "RGB"
    )

    image = image.resize(
        (160, 160)
    )

    image = np.array(
        image
    )

    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        channel_axis=-1
    )

    prediction = svm.predict(
        [features]
    )[0]

    return int(prediction)


render_html_sidebar(
    """
    <div class="sidebar-logo">
        <div class="sidebar-logo-icon"><span class="icon" style="font-size:1.9rem; color:white; vertical-align:middle;">nutrition</span></div>
        <div>
            <div class="sidebar-logo-title">FreshFruit</div>
            <div class="sidebar-logo-sub">การจำแนกผลไม้ด้วย AI</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


render_html_sidebar(
    """
    <div class="sidebar-menu-label">เมนูหลัก</div>
    """,
    unsafe_allow_html=True
)


page = st.sidebar.radio(
    "เมนูนำทาง",
    [
        "Overview",
        "Predict",
        "Model Comparison"
    ],
    label_visibility="collapsed"
)


render_html_sidebar(
    f"""
    <div class="sidebar-menu-label" style="margin-top:1.2rem;">ผลไม้ในชุดข้อมูล</div>
    <div class="pill-wrap">{fruit_pills_html()}</div>
    """,
    unsafe_allow_html=True
)

celebrate = st.sidebar.checkbox(
    "เอฟเฟกต์ฉลองเมื่อผลไม้สด",
    value=True
)

st.sidebar.divider()

render_html_sidebar(
    """
    <div class="sidebar-footer">
        <b>FreshFruit AI</b><br>
        Project FreshFruit AI<br>
        การจำแนกผลไม้สด/เน่า
    </div>
    """,
    unsafe_allow_html=True
)


if page == "Overview":

    render_hero(
        icon='eco',
        eyebrow='Machine Learning • คุณภาพผลไม้',
        title='FreshFruit AI',
        text='ระบบจำแนกภาพที่ใช้ Machine Learning เพื่อตรวจสอบว่าผลไม้ในภาพ <b>สด</b> หรือ <b>เน่า</b>',
        variant='full'
    )

    render_html(
        fruit_marquee_html(),
        unsafe_allow_html=True
    )


    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "จำนวนภาพในชุดข้อมูล",
            "3,193"
        )

    with col2:
        st.metric(
            "จำนวนคลาส",
            "2"
        )

    with col3:
        st.metric(
            "จำนวนโมเดล",
            "3"
        )

    with col4:
        st.metric(
            "ความแม่นยำสูงสุด (ชุดทดสอบ)",
            f"{df['Test Accuracy'].max():.2f}%"
        )


    render_html(
        "<br>",
        unsafe_allow_html=True
    )


    render_html(
        '<div class="section-label">เกี่ยวกับโปรเจกต์</div>',
        unsafe_allow_html=True
    )

    render_html(
        '<div class="section-title">FreshFruit AI คืออะไร?</div>',
        unsafe_allow_html=True
    )


    col1, col2 = st.columns(
        [1.25, 1]
    )


    with col1:

        render_html(
            """
            <div class="info-card">

                <div class="info-card-title">
                    <span class="icon">biotech</span> ตรวจสอบผลไม้ด้วย AI
                </div>

                <div class="info-card-text">
                    FreshFruit AI เปรียบเทียบแนวทางการจำแนกภาพ
                    3 แบบ เพื่อแยกผลไม้สดออกจากผลไม้เน่า
                    ผู้ใช้สามารถอัปโหลดภาพ
                    แล้วให้โมเดลที่ฝึกไว้วิเคราะห์ผลได้
                </div>

                <br>

                <div class="info-card-text">
                    โปรเจกต์นี้ประเมินประสิทธิภาพของโมเดล
                    ด้วยค่า Accuracy, Precision, Recall
                    และ F1-score บนชุดข้อมูลทดสอบที่แยกอิสระ
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        render_html(
            """
            <div class="info-card">

                <div class="info-card-title">
                    <span class="icon">psychology</span> โมเดลที่ใช้
                </div>

                <div class="info-card-text">

                    <span class="icon">grid_view</span> <b>HOG + SVM</b><br>
                    แนวทางคอมพิวเตอร์วิทัศน์แบบดั้งเดิม
                    โดยใช้ฟีเจอร์ของภาพที่ออกแบบขึ้นเอง

                    <br><br>

                    <span class="icon">hub</span> <b>Custom CNN</b><br>
                    โครงข่ายประสาทเทียมแบบคอนโวลูชัน
                    ที่ฝึกขึ้นเฉพาะสำหรับชุดข้อมูลนี้

                    <br><br>

                    <span class="icon">smartphone</span> <b>MobileNetV2</b><br>
                    โมเดลจำแนกภาพขนาดเบา
                    ที่ใช้เทคนิค Transfer Learning

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    render_html(
        "<br>",
        unsafe_allow_html=True
    )


    render_html(
        '<div class="section-label">ขั้นตอนการทำงาน</div>',
        unsafe_allow_html=True
    )

    render_html(
        '<div class="section-title">ระบบทำงานอย่างไร</div>',
        unsafe_allow_html=True
    )


    workflow_cols = st.columns(4)

    workflow_data = [
        (
            '<span class="icon">upload</span> 1',
            "อัปโหลด",
            "อัปโหลดภาพผลไม้ที่หน้าทำนายผล"
        ),
        (
            '<span class="icon">image</span> 2',
            "เตรียมภาพ",
            "ปรับขนาดและเตรียมภาพให้เหมาะกับโมเดลที่เลือก"
        ),
        (
            '<span class="icon">psychology</span> 3',
            "วิเคราะห์",
            "โมเดล Machine Learning ที่ฝึกไว้วิเคราะห์ภาพ"
        ),
        (
            '<span class="icon">check_circle</span> 4',
            "แสดงผล",
            "ระบบแสดงผลการจำแนกว่าสดหรือเน่า"
        )
    ]


    for col, item in zip(
        workflow_cols,
        workflow_data
    ):

        with col:

            render_html(
                f"""
                <div class="info-card">

                    <div class="workflow-number">
                        {item[0]}
                    </div>

                    <div class="workflow-title">
                        {item[1]}
                    </div>

                    <div class="workflow-text">
                        {item[2]}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


    render_html(
        "<br>",
        unsafe_allow_html=True
    )


    render_html(
        '<div class="section-label">ประสิทธิภาพ</div>',
        unsafe_allow_html=True
    )

    render_html(
        '<div class="section-title">ความแม่นยำของโมเดลบนชุดทดสอบ</div>',
        unsafe_allow_html=True
    )


    chart_data = df.set_index(
        "Model"
    )["Test Accuracy"]


    st.bar_chart(
        chart_data,
        height=330
    )


    st.caption(
        "ประสิทธิภาพวัดจากชุดข้อมูลทดสอบที่แยกอิสระ"
    )


elif page == "Predict":

    render_hero(
        icon='search',
        eyebrow='ตรวจสอบด้วย AI',
        title='ทำนายคุณภาพผลไม้',
        text='อัปโหลดภาพผลไม้ แล้วให้โมเดลที่เลือก จำแนกว่าเป็นผลไม้ <b>สด</b> หรือ <b>เน่า</b>'
    )


    col1, col2 = st.columns(
        [1, 1]
    )


    with col1:

        render_html(
            """
            <div class="upload-title">
                เลือกโมเดล
            </div>
            """,
            unsafe_allow_html=True
        )


        model_name = st.selectbox(
            "โมเดล",
            [
                "Custom CNN",
                "MobileNetV2",
                "HOG + SVM"
            ],
            label_visibility="collapsed"
        )


        model_accuracy = {
            "Custom CNN": "93.75%",
            "MobileNetV2": "90.42%",
            "HOG + SVM": "87.08%"
        }


        render_html(
            f"""
            <div class="model-badge">
                ความแม่นยำบนชุดทดสอบ: {model_accuracy[model_name]}
            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        uploaded_file = st.file_uploader(
            "อัปโหลดภาพผลไม้",
            type=[
                "jpg",
                "jpeg",
                "png"
            ],
            label_visibility="visible"
        )

        with st.expander("เคล็ดลับการถ่ายภาพ"):
            st.markdown(
                "ผลการทำนายอาจได้รับผลกระทบจากแสง มุมถ่าย พื้นหลัง และคุณภาพของภาพ "
                "ข้อแนะนำทั่วไป:\n"
                "- ถ่ายในที่ที่มีแสงสว่างเพียงพอ\n"
                "- ให้เห็นผลไม้ทั้งผลและอยู่กลางภาพ\n"
                "- ใช้พื้นหลังเรียบ ๆ และหลีกเลี่ยงแสงสะท้อนจ้า\n"
                "- หลีกเลี่ยงภาพเบลอหรือภาพที่ถูกบีบอัดมาก"
            )


    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")


        render_html(
            "<br>",
            unsafe_allow_html=True
        )


        col1, col2 = st.columns(
            [1, 1]
        )


        with col1:

            render_html(
                '<div class="section-label">ภาพที่นำเข้า</div>',
                unsafe_allow_html=True
            )

            st.image(
                image,
                use_container_width=True
            )


        with col2:

            render_html(
                '<div class="section-label">การวิเคราะห์</div>',
                unsafe_allow_html=True
            )


            render_html(
                f"""
                <div class="info-card">

                    <div class="info-card-title">
                        พร้อมวิเคราะห์
                    </div>

                    <div class="info-card-text">
                        โมเดล:
                        <b>{model_name}</b>
                        <br><br>
                        ความแม่นยำบนชุดทดสอบ:
                        <b>{model_accuracy[model_name]}</b>
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


            render_html(
                "<br>",
                unsafe_allow_html=True
            )


            predict_button = st.button(
                "วิเคราะห์ภาพ",
                use_container_width=True
            )


            if predict_button:

                with st.spinner("AI กำลังวิเคราะห์ภาพ..."):

                    if model_name == "Custom CNN":

                        probability = predict_cnn(
                            image
                        )

                        label = (
                            "Rotten"
                            if probability >= 0.5
                            else "Fresh"
                        )

                        fresh_probability = (
                            1 - probability
                        )

                        rotten_probability = probability


                    elif model_name == "MobileNetV2":

                        probability = predict_mobilenet(
                            image
                        )

                        label = (
                            "Rotten"
                            if probability >= 0.5
                            else "Fresh"
                        )

                        fresh_probability = (
                            1 - probability
                        )

                        rotten_probability = probability


                    else:

                        prediction = predict_hog_svm(
                            image
                        )

                        label = (
                            "Rotten"
                            if prediction == 1
                            else "Fresh"
                        )

                        fresh_probability = None
                        rotten_probability = None


                render_html(
                    "<br>",
                    unsafe_allow_html=True
                )


                if label == "Fresh":

                    confidence = (
                        fresh_probability
                        if fresh_probability is not None
                        else 1.0
                    )

                    render_html(
                        f"""
                        <div class="result-fresh">
                            <div class="result-label">
                                <span class="icon">check_circle</span> สด
                            </div>
                            {gauge_html(confidence, fresh_probability is not None)}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                else:

                    confidence = (
                        rotten_probability
                        if rotten_probability is not None
                        else 1.0
                    )

                    render_html(
                        f"""
                        <div class="result-rotten">
                            <div class="result-label">
                                <span class="icon">cancel</span> เน่า
                            </div>
                            {gauge_html(confidence, fresh_probability is not None)}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                st.toast("วิเคราะห์เสร็จแล้ว")

                if label == "Fresh" and celebrate:
                    st.balloons()

                if fresh_probability is not None:

                    render_html(
                        "<br>",
                        unsafe_allow_html=True
                    )


                    prob1, prob2 = st.columns(2)


                    with prob1:

                        st.metric(
                            "สด",
                            f"{fresh_probability * 100:.2f}%"
                        )

                        st.progress(
                            float(
                                fresh_probability
                            )
                        )


                    with prob2:

                        st.metric(
                            "เน่า",
                            f"{rotten_probability * 100:.2f}%"
                        )

                        st.progress(
                            float(
                                rotten_probability
                            )
                        )


                else:

                    st.info(
                        "HOG + SVM ให้ผลทำนายเป็นคลาสเท่านั้น "
                        "โดยไม่มีค่าประมาณความน่าจะเป็น"
                    )


                render_html(
                    """
                    <div class="small-note">
                        ผลการทำนายอาจได้รับผลกระทบจากแสง
                        คุณภาพของภาพ มุมถ่าย พื้นหลัง
                        และลักษณะของผลไม้
                    </div>
                    """,
                    unsafe_allow_html=True
                )


elif page == "Model Comparison":

    render_hero(
        icon='analytics',
        eyebrow='การประเมินโมเดล',
        title='Model Comparison',
        text='เปรียบเทียบ 3 แนวทางในการจำแนก ผลไม้สดและผลไม้เน่า โดยใช้ ชุดข้อมูลทดสอบที่แยกอิสระชุดเดียวกัน'
    )


    best_model = df.loc[
        df["Test Accuracy"].idxmax()
    ]


    col1, col2, col3 = st.columns(3)


    with col1:
        st.metric(
            "ความแม่นยำสูงสุด (ชุดทดสอบ)",
            best_model["Test Accuracy"]
        )


    with col2:
        st.metric(
            "จำนวนโมเดลที่ประเมิน",
            "3"
        )


    with col3:
        st.metric(
            "จำนวนภาพทดสอบ",
            "480"
        )


    render_html(
        "<br>",
        unsafe_allow_html=True
    )


    render_html(
        '<div class="section-label">ผลลัพธ์</div>',
        unsafe_allow_html=True
    )

    render_html(
        '<div class="section-title">สรุปประสิทธิภาพ</div>',
        unsafe_allow_html=True
    )


    display_df = df.copy().rename(
        columns={
            "Model": "โมเดล",
            "Test Accuracy": "ความแม่นยำ (ชุดทดสอบ)"
        }
    )


    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


    render_html(
        "<br>",
        unsafe_allow_html=True
    )


    render_html(
        '<div class="section-label">ความแม่นยำ</div>',
        unsafe_allow_html=True
    )

    render_html(
        '<div class="section-title">เปรียบเทียบความแม่นยำบนชุดทดสอบ</div>',
        unsafe_allow_html=True
    )


    accuracy_image = (
        ASSETS_DIR /
        "accuracy_comparison.png"
    )


    if accuracy_image.exists():

        st.image(
            str(accuracy_image),
            use_container_width=True
        )


    render_html(
        "<br>",
        unsafe_allow_html=True
    )


    render_html(
        '<div class="section-label">ตัวชี้วัด</div>',
        unsafe_allow_html=True
    )

    render_html(
        '<div class="section-title">ตัวชี้วัดการจำแนก</div>',
        unsafe_allow_html=True
    )


    metrics_image = (
        ASSETS_DIR /
        "metrics_comparison.png"
    )


    if metrics_image.exists():

        st.image(
            str(metrics_image),
            use_container_width=True
        )


    render_html(
        "<br>",
        unsafe_allow_html=True
    )


    render_html(
        '<div class="section-label">ผลการทำนาย</div>',
        unsafe_allow_html=True
    )

    render_html(
        '<div class="section-title">การทำนายที่ถูกต้องและไม่ถูกต้อง</div>',
        unsafe_allow_html=True
    )


    correct_image = (
        ASSETS_DIR /
        "correct_incorrect.png"
    )


    if correct_image.exists():

        st.image(
            str(correct_image),
            use_container_width=True
        )


    st.caption(
        "ตัวชี้วัดทั้งหมดคำนวณจากชุดข้อมูลทดสอบที่แยกอิสระ"
    )