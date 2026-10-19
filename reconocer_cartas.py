#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reconocedor experimental de texturas de Yu-Gi-Oh! Duel Transer usando referencias locales de YGOPro."""
from __future__ import annotations
import argparse, json, os, re, shutil, sys, time
from pathlib import Path
import cv2
import numpy as np

EXTS={'.png','.jpg','.jpeg','.bmp','.webp','.tif','.tiff'}


def images_under(root: Path):
    if not root.exists(): return []
    return sorted((p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in EXTS), key=lambda p: str(p).lower())


def read_img(path: Path):
    # imdecode tolerates non-ASCII Windows paths better than cv2.imread
    try:
        raw=np.fromfile(str(path),dtype=np.uint8)
        if not len(raw): return None
        return cv2.imdecode(raw,cv2.IMREAD_COLOR)
    except Exception: return None


def perceptual_hash(img, size=32, low=8):
    # DCT-based pHash; normalize card dimensions while preserving full card silhouette.
    if img is None: return None
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray=cv2.resize(gray,(size,size),interpolation=cv2.INTER_AREA)
    dct=cv2.dct(np.float32(gray))[:low,:low]
    vals=dct.flatten()[1:]
    med=np.median(vals)
    return (vals>med).astype(np.uint8)


def color_signature(img, bins=(8,8,8)):
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h=cv2.calcHist([hsv],[0,1,2],None,list(bins),[0,180,0,256,0,256])
    cv2.normalize(h,h,alpha=1.0,norm_type=cv2.NORM_L1)
    return h.flatten()


def load_features(paths, label):
    hashes=[]; colors=[]; valid=[]
    total=len(paths)
    for i,p in enumerate(paths,1):
        img=read_img(p)
        if img is None: continue
        ph=perceptual_hash(img)
        if ph is None: continue
        hashes.append(ph); colors.append(color_signature(img)); valid.append(p)
        if i%500==0 or i==total: print(f'  {label}: {i}/{total}',flush=True)
    return valid,np.asarray(hashes,dtype=np.uint8),np.asarray(colors,dtype=np.float32)


def card_name(path: Path):
    # YGOPro filename example: "Blue-Eyes White Dragon [89631139].jpg"
    name=path.stem
    name=re.sub(r'\s*\[\d+\]\s*$','',name).strip()
    return name or path.stem


def score_matches(src_hash, src_color, ref_hashes, ref_colors, limit=3):
    # Binary pHash Hamming similarity + histogram similarity. This is a ranking heuristic, not a calibrated probability.
    ham=np.count_nonzero(ref_hashes != src_hash,axis=1)
    phash_score=1.0-ham/max(1,len(src_hash))
    # normalized histograms; Bhattacharyya coefficient in [0,1]
    coeff=np.sum(np.sqrt(np.clip(ref_colors,0,None)*np.clip(src_color,0,None)),axis=1)
    coeff=np.clip(coeff,0,1)
    scores=0.78*phash_score+0.22*coeff
    inds=np.argpartition(scores,-min(limit,len(scores)))[-min(limit,len(scores)):]
    inds=inds[np.argsort(scores[inds])[::-1]]
    return [(int(j),float(scores[j]),float(phash_score[j]),float(coeff[j])) for j in inds]


def relpath(p: Path, base: Path):
    try: return os.path.relpath(p,base).replace('\\','/')
    except ValueError: return str(p).replace('\\','/')


def main():
    ap=argparse.ArgumentParser(description='Compara texturas de Duel Transer con cartas JPG de YGOPro y genera un HTML revisable.')
    ap.add_argument('--texturas',default='Grandes',help='Carpeta con PNG de Duel Transer (por defecto: ./Grandes)')
    ap.add_argument('--referencias',default='cartas_completas',help='Carpeta con imágenes JPG de YGOPro (por defecto: ./cartas_completas)')
    ap.add_argument('--salida',default='revision_duel_transer.html',help='HTML generado')
    ap.add_argument('--top',type=int,default=3,help='Cantidad de candidatas por textura')
    ap.add_argument('--max-texturas',type=int,default=0,help='Solo probar N texturas; 0 = todas')
    ap.add_argument('--solo-pendientes',default='',help='JSON exportado desde el HTML: volver a comparar solo las texturas marcadas No coincide o pendientes')
    args=ap.parse_args()
    texture_root=Path(args.texturas).resolve(); ref_root=Path(args.referencias).resolve(); out=Path(args.salida).resolve()
    if not texture_root.is_dir(): sys.exit(f'No existe la carpeta de texturas: {texture_root}')
    if not ref_root.is_dir(): sys.exit(f'No existe la carpeta de referencias: {ref_root}')
    textures=images_under(texture_root); refs=images_under(ref_root)
    if args.solo_pendientes:
        pending_file=Path(args.solo_pendientes).resolve()
        try:
            saved=json.loads(pending_file.read_text(encoding='utf-8'))
            pending_names=set()
            for row in saved:
                decision=row.get('decision') or {}
                if decision.get('type') in ('none','pending') or row.get('status') in ('unmatched','pending'):
                    pending_names.add(row.get('texture'))
            textures=[p for p in textures if p.name in pending_names]
            print(f'Filtro de pendientes: {len(textures)} texturas encontradas en {pending_file.name}')
        except Exception as exc:
            sys.exit(f'No se pudo leer el JSON de pendientes {pending_file}: {exc}')
    if args.max_texturas: textures=textures[:args.max_texturas]
    if not textures: sys.exit('No encontré imágenes de texturas en la carpeta indicada.')
    if not refs: sys.exit('No encontré imágenes de referencia en la carpeta indicada.')
    print(f'Texturas: {len(textures)} | Referencias: {len(refs)}')
    print('Calculando características de las referencias...')
    refs,ref_hashes,ref_colors=load_features(refs,'Referencias')
    if not refs: sys.exit('No se pudo leer ninguna imagen de referencia.')
    print('Comparando texturas...')
    results=[]
    for i,p in enumerate(textures,1):
        img=read_img(p)
        if img is None: continue
        ph=perceptual_hash(img); color=color_signature(img)
        ranked=score_matches(ph,color,ref_hashes,ref_colors,max(1,args.top))
        candidates=[]
        for idx,score,ps,cs in ranked:
            candidates.append({'name':card_name(refs[idx]),'path':relpath(refs[idx],out.parent),'score':round(score*100,1),'phash':round(ps*100,1),'color':round(cs*100,1),'filename':refs[idx].name})
        results.append({'texture':p.name,'texture_path':relpath(p,out.parent),'expected_name':None,'candidates':candidates})
        if i%100==0 or i==len(textures): print(f'  Texturas: {i}/{len(textures)}',flush=True)
    payload=json.dumps(results,ensure_ascii=False).replace('</','<\\/')
    html=HTML.replace('__DATA__',payload).replace('__TEXTURE_COUNT__',str(len(results))).replace('__REF_COUNT__',str(len(refs)))
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(html,encoding='utf-8')
    print(f'HTML generado: {out}\nAbrilo en el navegador para revisar las candidatas.')

HTML=r'''<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Revisión de cartas · Duel Transer</title>
<style>
:root{color-scheme:dark;--bg:#0c111b;--panel:#151e2c;--line:#2b394d;--text:#eef3fa;--muted:#a0afc3;--blue:#74b9ff;--green:#58d6a0;--amber:#ffd27a}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,Segoe UI,sans-serif}header{padding:24px clamp(16px,3vw,36px);background:#111927;border-bottom:1px solid var(--line)}h1{font-size:clamp(22px,3vw,30px);margin:0 0 5px}p{color:var(--muted);margin:0}main{max-width:1600px;margin:auto;padding:22px clamp(12px,2vw,28px)}.toolbar{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px}button,input,select{font:inherit;color:var(--text);background:#202c3e;border:1px solid #3a4a62;border-radius:8px;padding:9px 11px}button{cursor:pointer}button:hover{filter:brightness(1.15)}button.active{border-color:var(--blue);background:#23476c}input{flex:1;min-width:210px}.stats{display:flex;gap:12px;align-items:center;flex-wrap:wrap;color:var(--muted);margin-bottom:16px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:13px}.item{background:var(--panel);border:1px solid var(--line);border-radius:13px;padding:14px;min-width:0}.item.doubt{border-color:#826632}.top{display:flex;justify-content:space-between;gap:10px;margin-bottom:10px}.filename{font-size:11px;color:var(--muted);overflow-wrap:anywhere}.pill{white-space:nowrap;border-radius:20px;padding:4px 8px;font-size:11px;background:#173e32;color:#91e8c5;height:max-content}.pill.mid{background:#4b3b1f;color:#ffdc91}.compare{display:grid;grid-template-columns:minmax(0,1fr) 24px minmax(0,1fr);gap:8px;align-items:center}.pane{background:#0e1520;border:1px solid var(--line);border-radius:9px;padding:8px;min-width:0}.pane label{display:block;font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.07em;margin-bottom:7px}.thumb{height:180px;display:flex;align-items:center;justify-content:center;background:#080d14;border-radius:6px;overflow:hidden}.thumb img{max-width:100%;max-height:100%;object-fit:contain}.noimg{font-size:11px;color:var(--muted);padding:10px;text-align:center}.name{font-weight:650;margin-top:7px;overflow-wrap:anywhere}.sub{font-size:11px;color:var(--muted)}.arrow{text-align:center;color:#8297b2;font-size:20px}.candidateActions{display:flex;gap:6px;flex-wrap:wrap;margin-top:10px}.candidateActions button{font-size:11px;padding:6px 8px}.candidateActions button.selected{outline:2px solid var(--green)}.correct{color:var(--green);font-size:12px;margin-top:8px}.bottom{margin-top:20px;border-top:1px solid var(--line);padding-top:16px;display:flex;gap:8px;justify-content:space-between;flex-wrap:wrap}.primary{background:#2469a5;border-color:#4b98d7;font-weight:700}@media(max-width:850px){.grid{grid-template-columns:1fr}}@media(max-width:500px){.compare{grid-template-columns:minmax(0,1fr) 16px minmax(0,1fr);gap:4px}.thumb{height:130px}.pane{padding:5px}}
</style></head><body><header><h1>Revisión de coincidencias · Duel Transer</h1><p>__TEXTURE_COUNT__ texturas analizadas · __REF_COUNT__ imágenes de referencia · ranking experimental, no veredicto automático</p></header><main><div class="toolbar"><button class="active" data-filter="all">Todas</button><button data-filter="doubt">Dudosas (&lt; 75%)</button><button data-filter="unreviewed">Sin revisar</button><button data-filter="unmatched">Sin coincidencia</button><input id="search" placeholder="Buscar nombre o archivo…"><select id="sort"><option value="source">Orden de textura</option><option value="low">Menor puntuación primero</option><option value="name">Nombre de textura A–Z</option></select></div><div class="stats"><span id="count"></span><span id="progress"></span><button id="export" class="primary">Exportar correcciones JSON</button></div><section id="grid" class="grid"></section><div class="bottom"><p>Las correcciones se guardan al exportar el JSON. La puntuación combina hash perceptual y color; necesita validación humana.</p><button id="reset">Borrar decisiones de esta sesión</button></div></main>
<script>
const data=__DATA__;let decisions={};let filter='all';const $=s=>document.querySelector(s);const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function render(){let rows=data.map((x,i)=>({...x,i,decision:decisions[i]}));const q=$('#search').value.toLowerCase().trim();if(q)rows=rows.filter(x=>(x.texture+' '+x.candidates.map(c=>c.name).join(' ')).toLowerCase().includes(q));if(filter==='doubt')rows=rows.filter(x=>!x.candidates.length||x.candidates[0].score<75);if(filter==='unreviewed')rows=rows.filter(x=>!x.decision);if(filter==='unmatched')rows=rows.filter(x=>x.decision?.type==='none'||!x.candidates.length);if($('#sort').value==='low')rows.sort((a,b)=>(a.candidates[0]?.score||0)-(b.candidates[0]?.score||0));if($('#sort').value==='name')rows.sort((a,b)=>a.texture.localeCompare(b.texture));$('#count').textContent=`Mostrando ${rows.length} de ${data.length}`;const unresolved=data.filter((x,i)=>decisions[i]?.type==='none'||!x.candidates.length).length;$('#progress').textContent=`${Object.keys(decisions).length} decisiones · ${unresolved} sin coincidencia`;$('#grid').innerHTML=rows.map(x=>{const best=x.candidates[0];return `<article class="item ${!best||best.score<75?'doubt':''}"><div class="top"><div><div class="name">${esc(x.texture)}</div><div class="filename">${esc(x.texture_path)}</div></div><span class="pill ${!best||best.score<75?'mid':''}">${best?best.score+'% similitud':'Sin candidata'}</span></div><div class="compare"><div class="pane"><label>Textura Duel Transer</label><div class="thumb"><img src="${esc(x.texture_path)}" loading="lazy" onerror="this.style.display='none';this.nextElementSibling.hidden=false"><div class="noimg" hidden>No se pudo cargar la imagen.<br>Comprobá la ruta local.</div></div><div class="sub">Original</div></div><div class="arrow">↔</div><div class="pane"><label>Mejor candidata YGOPro</label><div class="thumb">${best?`<img src="${esc(best.path)}" loading="lazy" onerror="this.style.display='none';this.nextElementSibling.hidden=false"><div class="noimg" hidden>No se pudo cargar la referencia.</div>`:'<div class="noimg">Sin imagen candidata</div>'}</div><div class="name">${best?esc(best.name):'No identificada'}</div><div class="sub">pHash ${best?best.phash+'%':'—'} · Color ${best?best.color+'%':'—'}</div></div></div><div class="candidateActions">${x.candidates.map((c,j)=>`<button class="${(x.decision?.candidate??0)===j && x.decision?.type!=='none'?'selected':' '}" onclick="pick(${x.i},${j})">${j===0?'1.º':j===1?'2.º':'3.º'} ${esc(c.name)} · ${c.score}%</button>`).join('')}<button onclick="mark(${x.i},'correct')">✓ Confirmar elegida</button><button onclick="mark(${x.i},'none')">No coincide / dejar pendiente</button></div>${x.decision?`<div class="correct">Decisión: ${x.decision.type==='none'?'SIN COINCIDENCIA — se revisará después':x.decision.type==='picked'?'candidata '+(x.decision.candidate+1)+' seleccionada':'candidata '+(x.decision.candidate+1)+' confirmada'} </div>`:''}</article>`}).join('')}
function pick(i,j){decisions[i]={type:'picked',candidate:j};render()}function mark(i,type){let prev=decisions[i]||{candidate:0};decisions[i]={type,candidate:prev.candidate??0};render()}document.querySelectorAll('[data-filter]').forEach(b=>b.onclick=()=>{filter=b.dataset.filter;document.querySelectorAll('[data-filter]').forEach(k=>k.classList.toggle('active',k===b));render()});$('#search').oninput=render;$('#sort').onchange=render;$('#reset').onclick=()=>{if(confirm('¿Borrar las decisiones de esta sesión?')){decisions={};render()}};$('#export').onclick=()=>{const out=data.map((x,i)=>{const d=decisions[i]||{type:x.candidates.length?'auto':'none',candidate:0};return {texture:x.texture,texture_path:x.texture_path,candidates:x.candidates,selected_candidate:d.candidate??0,selected_name:x.candidates[d.candidate??0]?.name??null,decision:d,status:d.type==='none'?'unmatched':d.type==='pending'?'pending':'selected'};} );const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,2)],{type:'application/json'}));a.download='correcciones_duel_transer.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};render();
</script></body></html>'''
if __name__=='__main__': main()
