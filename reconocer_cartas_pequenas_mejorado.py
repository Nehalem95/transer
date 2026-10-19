#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reconocimiento mejorado para texturas pequeñas de Yu-Gi-Oh! Duel Transer.
Combina hashes perceptuales, color y estructura de bordes, sin deformar la proporción.
Conserva la revisión HTML y la exportación compatible con el flujo anterior.
"""
from __future__ import annotations
import argparse, json, os, re, sys
from pathlib import Path
import cv2
import numpy as np

EXTS={'.png','.jpg','.jpeg','.bmp','.webp','.tif','.tiff'}

def images_under(root: Path):
    if not root.exists(): return []
    return sorted((p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in EXTS), key=lambda p: str(p).lower())

def read_img(path: Path):
    try:
        raw=np.fromfile(str(path),dtype=np.uint8)
        if not len(raw): return None
        return cv2.imdecode(raw,cv2.IMREAD_COLOR)
    except Exception: return None

def resize_letterbox(img, width=64, height=96, interpolation=cv2.INTER_AREA):
    """Redimensiona sin deformar la proporción y centra en un lienzo fijo."""
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    scale = min(width / w, height / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=interpolation)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x, y = (width - nw) // 2, (height - nh) // 2
    canvas[y:y+nh, x:x+nw] = resized
    return canvas

def perceptual_hash(img, size=(64, 96), low=8):
    """pHash con proporción conservada (la carta no se aplasta a un cuadrado)."""
    if img is None: return None
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    if isinstance(size, int):
        gray=cv2.resize(gray,(size,size),interpolation=cv2.INTER_AREA)
    else:
        # Mantener proporción dentro de un lienzo vertical fijo.
        h,w=gray.shape[:2]
        scale=min(size[0]/w,size[1]/h)
        nw,nh=max(1,round(w*scale)),max(1,round(h*scale))
        resized=cv2.resize(gray,(nw,nh),interpolation=cv2.INTER_AREA)
        canvas=np.zeros((size[1],size[0]),dtype=np.uint8)
        x,y=(size[0]-nw)//2,(size[1]-nh)//2
        canvas[y:y+nh,x:x+nw]=resized
        gray=canvas
    dct=cv2.dct(np.float32(gray))[:low,:low]
    vals=dct.flatten()[1:]
    return (vals>np.median(vals)).astype(np.uint8)

def edge_signature(img, width=32, height=48):
    """Firma binaria de contornos, menos sensible a variaciones globales de color."""
    if img is None: return None
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    h,w=gray.shape[:2]
    scale=min(width/max(1,w),height/max(1,h))
    nw,nh=max(1,round(w*scale)),max(1,round(h*scale))
    small=cv2.resize(gray,(nw,nh),interpolation=cv2.INTER_AREA)
    canvas=np.zeros((height,width),dtype=np.uint8)
    x,y=(width-nw)//2,(height-nh)//2
    canvas[y:y+nh,x:x+nw]=small
    edges=cv2.Canny(canvas,45,120)
    return (edges.flatten()>0).astype(np.uint8)

def color_signature(img, bins=(8,8,8)):
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h=cv2.calcHist([hsv],[0,1,2],None,list(bins),[0,180,0,256,0,256])
    cv2.normalize(h,h,alpha=1.0,norm_type=cv2.NORM_L1)
    return h.flatten()

def art_crop(img):
    """Recorte aproximado de la zona de ilustración de una carta Yu-Gi-Oh!."""
    h,w=img.shape[:2]
    # Evita recortes degenerados en imágenes pequeñas o inusuales.
    x1,x2=int(w*.08),max(int(w*.92),1)
    y1,y2=int(h*.18),max(int(h*.65),1)
    if x2<=x1 or y2<=y1: return img
    return img[y1:y2,x1:x2]

def load_features(paths, label):
    valid=[]; full_h=[]; full_c=[]; art_h=[]; art_c=[]; full_e=[]; art_e=[]
    total=len(paths)
    for i,p in enumerate(paths,1):
        img=read_img(p)
        if img is not None:
            crop=art_crop(img)
            valid.append(p)
            full_h.append(perceptual_hash(img)); full_c.append(color_signature(img))
            art_h.append(perceptual_hash(crop)); art_c.append(color_signature(crop))
            full_e.append(edge_signature(img)); art_e.append(edge_signature(crop))
        if i%250==0 or i==total: print(f'  {label}: {i}/{total}',flush=True)
    return (valid,np.asarray(full_h,dtype=np.uint8),np.asarray(full_c,dtype=np.float32),
            np.asarray(art_h,dtype=np.uint8),np.asarray(art_c,dtype=np.float32),
            np.asarray(full_e,dtype=np.uint8),np.asarray(art_e,dtype=np.uint8))

def card_name(path: Path):
    name=re.sub(r'\s*\[\d+\]\s*$','',path.stem).strip()
    return name or path.stem

def score_matches(src_h,src_c,src_ah,src_ac,src_e,src_ae,ref_h,ref_c,ref_ah,ref_ac,ref_e,ref_ae,limit=10):
    def hsim(a,b): return 1.0-np.count_nonzero(b!=a,axis=1)/max(1,len(a))
    def csim(a,b): return np.clip(np.sum(np.sqrt(np.clip(b,0,None)*np.clip(a,0,None)),axis=1),0,1)
    ph=hsim(src_h,ref_h); col=csim(src_c,ref_c)
    aph=hsim(src_ah,ref_ah); acol=csim(src_ac,ref_ac)
    edge=hsim(src_e,ref_e); aedge=hsim(src_ae,ref_ae)
    # La ilustración y la estructura pesan más que el color global de la carta.
    # Los pesos suman 1.0; se guardan métricas individuales para depurar resultados.
    scores=.25*ph+.08*col+.30*aph+.07*acol+.18*edge+.12*aedge
    n=min(max(1,limit),len(scores)); inds=np.argpartition(scores,-n)[-n:]; inds=inds[np.argsort(scores[inds])[::-1]]
    return [(int(j),float(scores[j]),float(ph[j]),float(col[j]),float(aph[j]),float(acol[j]),float(edge[j]),float(aedge[j])) for j in inds]

def relpath(p: Path, base: Path):
    try: return os.path.relpath(p,base).replace('\\','/')
    except ValueError: return str(p).replace('\\','/')

HTML=r'''<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Reconocimiento mejorado · Pequeñas · Duel Transer</title>
<style>
:root{color-scheme:dark;--bg:#0c111b;--panel:#151e2c;--line:#2b394d;--text:#eef3fa;--muted:#a0afc3;--blue:#74b9ff;--green:#58d6a0;--amber:#ffd27a}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,Segoe UI,sans-serif}header{padding:22px clamp(14px,3vw,34px);background:#111927;border-bottom:1px solid var(--line)}h1{font-size:clamp(22px,3vw,30px);margin:0 0 5px}p{color:var(--muted);margin:0}main{max-width:1700px;margin:auto;padding:18px clamp(10px,2vw,26px)}.toolbar{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px}button,input,select{font:inherit;color:var(--text);background:#202c3e;border:1px solid #3a4a62;border-radius:8px;padding:9px 11px}button{cursor:pointer}button:hover{filter:brightness(1.15)}button.active{border-color:var(--blue);background:#23476c}input{min-width:190px}.search{flex:1}.stats{display:flex;gap:12px;align-items:center;flex-wrap:wrap;color:var(--muted);margin-bottom:14px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:13px}.item{background:var(--panel);border:1px solid var(--line);border-radius:13px;padding:13px;min-width:0}.item.doubt{border-color:#826632}.top{display:flex;justify-content:space-between;gap:10px;margin-bottom:9px}.filename{font-size:11px;color:var(--muted);overflow-wrap:anywhere}.pill{white-space:nowrap;border-radius:20px;padding:4px 8px;font-size:11px;background:#173e32;color:#91e8c5;height:max-content}.pill.mid{background:#4b3b1f;color:#ffdc91}.compare{display:grid;grid-template-columns:minmax(0,1fr) 20px minmax(0,1.18fr);gap:7px;align-items:stretch}.pane{background:#0e1520;border:1px solid var(--line);border-radius:9px;padding:7px;min-width:0}.pane label{display:block;font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px}.thumb{height:230px;display:flex;align-items:center;justify-content:center;background:#080d14;border-radius:6px;overflow:hidden}.thumb img{max-width:100%;max-height:100%;object-fit:contain}.pixelated{image-rendering:pixelated;image-rendering:crisp-edges}.main-candidate .thumb{height:280px}.main-candidate{border-color:#426b66}.keyboard-hint{font-size:11px;color:var(--muted);margin-top:7px}.actions button{font-size:11px;padding:6px 8px}.actions .selected{background:#174638;border-color:var(--green)}.noimg{font-size:11px;color:var(--muted);padding:10px;text-align:center}.name{font-weight:650;margin-top:6px;overflow-wrap:anywhere}.sub{font-size:11px;color:var(--muted);overflow-wrap:anywhere}.arrow{text-align:center;color:#8297b2;font-size:20px}.actions{display:flex;gap:6px;flex-wrap:wrap;margin-top:9px}.actions button{font-size:11px;padding:6px 8px}.selected{outline:2px solid var(--green)}.decision{color:var(--green);font-size:12px;margin-top:8px}.manual{margin-top:10px;border-top:1px solid var(--line);padding-top:10px}.manualhead{display:flex;gap:7px;flex-wrap:wrap}.manualhead input{flex:1;min-width:180px}.results{display:grid;grid-template-columns:repeat(auto-fill,minmax(112px,1fr));gap:8px;margin-top:9px}.ref{background:#0e1520;border:1px solid var(--line);border-radius:8px;padding:7px;cursor:pointer;min-width:0}.ref:hover{border-color:var(--blue)}.ref img{width:100%;height:115px;object-fit:contain;background:#080d14;border-radius:5px}.ref .name{font-size:11px}.ref .sub{font-size:10px}.pagination{display:flex;gap:8px;align-items:center;justify-content:center;flex-wrap:wrap;margin:16px 0}.pagination span{color:var(--muted)}.bottom{margin-top:18px;border-top:1px solid var(--line);padding-top:14px;display:flex;gap:8px;justify-content:space-between;flex-wrap:wrap}.primary{background:#2469a5;border-color:#4b98d7;font-weight:700}.hint{font-size:12px;margin-top:6px}@media(max-width:850px){.grid{grid-template-columns:1fr}}@media(max-width:500px){.compare{grid-template-columns:minmax(0,1fr) 14px minmax(0,1fr);gap:3px}.thumb{height:170px}.main-candidate .thumb{height:210px}.pane{padding:5px}}
</style></head><body><header><h1>Reconocimiento mejorado · Pequeñas</h1><p>__TEXTURE_COUNT__ texturas · __REF_COUNT__ referencias · seis señales: pHash, color y contornos, para carta completa y zona de ilustración.</p><p class="hint">Original ampliada conservando los píxeles. La primera candidata se muestra grande; las otras cuatro quedan como opciones debajo. Las decisiones se guardan automáticamente en este navegador.</p></header><main>
<div class="toolbar"><button class="active" data-filter="all">Todas las pendientes</button><button data-filter="unreviewed">Sin decisión</button><button data-filter="unmatched">Sin coincidencia</button><input class="search" id="search" placeholder="Buscar textura o nombre de candidata…"><select id="sort"><option value="source">Orden de textura</option><option value="low">Menor puntuación</option><option value="name">Textura A–Z</option></select><button id="export" class="primary">Exportar JSON completo</button></div><div class="stats"><span id="count"></span><span id="progress"></span></div><section id="grid" class="grid"></section><div class="pagination"><button id="prev">← Anterior</button><span id="pageinfo"></span><button id="next">Siguiente →</button></div><div class="bottom"><div><p>Las decisiones se guardan automáticamente en este navegador y se incluyen en el JSON completo al exportar. También se conservan las entradas anteriores que no forman parte de esta segunda pasada.</p><p class="hint" id="autosave">Recuperando guardado local…</p></div><button id="reset">Borrar decisiones de esta sesión</button></div></main>
<script>
const data=__DATA__, refs=__REFS__, prior=__PRIOR__; let decisions={};let filter='all',page=0;const PAGE_SIZE=40;const STORAGE_KEY='duel-transer-pequenas-review-v2:'+location.pathname;const $=s=>document.querySelector(s);const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function loadSaved(){try{const saved=JSON.parse(localStorage.getItem(STORAGE_KEY)||'{}');data.forEach((x,i)=>{if(saved[x.texture])decisions[i]=saved[x.texture]});$('#autosave').textContent=Object.keys(saved).length?`Se recuperaron ${Object.keys(saved).length} decisiones guardadas`:'Guardado automático activo'}catch(e){console.warn('No se pudo recuperar el guardado local',e);$('#autosave').textContent='El navegador no permite el guardado local; exportá el JSON con frecuencia.'}}
function saveDecisions(){try{const saved={};data.forEach((x,i)=>{if(decisions[i])saved[x.texture]=decisions[i]});localStorage.setItem(STORAGE_KEY,JSON.stringify(saved));$('#autosave').textContent='Guardado automático · '+new Date().toLocaleTimeString()}catch(e){$('#autosave').textContent='No se pudo guardar automáticamente; exportá el JSON como respaldo.'}}
function getDecision(x,i){return decisions[i]||x.saved_decision||{type:'auto',candidate:0}}
function getRows(){let rows=data.map((x,i)=>({...x,i,decision:getDecision(x,i)}));const q=$('#search').value.toLowerCase().trim();if(q)rows=rows.filter(x=>(x.texture+' '+x.candidates.map(c=>c.name).join(' ')).toLowerCase().includes(q));if(filter==='unreviewed')rows=rows.filter(x=>!decisions[x.i]&&!x.saved_decision);if(filter==='unmatched')rows=rows.filter(x=>{let d=getDecision(x,x.i);return d.type==='none'||d.type==='pending'});if($('#sort').value==='low')rows.sort((a,b)=>(a.candidates[0]?.score||0)-(b.candidates[0]?.score||0));if($('#sort').value==='name')rows.sort((a,b)=>a.texture.localeCompare(b.texture));return rows}
function cardHTML(x){const d=x.decision, best=x.candidates[d.candidate??0]||x.candidates[0];return `<article class="item ${!best||best.score<75?'doubt':''}" id="item-${x.i}"><div class="top"><div><div class="name">${esc(x.texture)}</div><div class="filename">${esc(x.texture_path)}</div></div><span class="pill ${!best||best.score<75?'mid':''}">${best?best.score+'% similitud':'Sin candidata'}</span></div><div class="compare"><div class="pane"><label>Textura original ampliada</label><div class="thumb"><img class="pixelated" src="${esc(x.texture_path)}" loading="lazy" onerror="this.style.display='none';this.nextElementSibling.hidden=false"><div class="noimg" hidden>No se pudo cargar.</div></div><div class="sub">Original: 40 × 58 px · ampliación nítida, sin suavizado</div></div><div class="arrow">↔</div><div class="pane main-candidate"><label>Mejor candidata seleccionada</label><div class="thumb">${best?`<img src="${esc(best.path)}" loading="lazy" onerror="this.style.display='none';this.nextElementSibling.hidden=false"><div class="noimg" hidden>No se pudo cargar la referencia.</div>`:'<div class="noimg">Sin imagen elegida</div>'}</div><div class="name">${best?esc(best.name):'No identificada'}</div><div class="sub">${best?`Similitud ${best.score??'manual'}% · pHash ${best.phash??'—'} · arte ${best.art_phash??'—'} · bordes ${best.edge??'—'}`:'Elegí una candidata o buscá manualmente.'}</div></div></div><div class="actions">${x.candidates.map((c,j)=>`<button class="${(d.candidate??0)===j&&d.type!=='none'&&d.type!=='pending'?'selected':''}" onclick="pick(${x.i},${j})">${j+1}. ${esc(c.name)} · ${c.score??'manual'}%</button>`).join('')}<button class="primary" onclick="mark(${x.i},'correct')">Confirmar selección · Enter</button><button onclick="mark(${x.i},'none')">Sin coincidencia · N</button></div><div class="keyboard-hint">Teclas: 1–5 elegir candidata · Enter confirmar · N marcar sin coincidencia · ← / → cambiar página</div>${d?`<div class="decision">Estado: ${d.type==='none'?'SIN COINCIDENCIA':d.type==='pending'?'PENDIENTE':d.type==='manual'?'SELECCIÓN MANUAL':d.type==='correct'?'CONFIRMADA':'candidata '+((d.candidate??0)+1)+' seleccionada'}</div>`:''}<div class="manual"><div class="manualhead"><input id="manual-${x.i}" placeholder="Buscar en todas las cartas (nombre de archivo)…" oninput="searchRefs(${x.i})"><button onclick="searchRefs(${x.i})">Buscar referencias</button></div><div class="sub">La búsqueda revisa nombres de archivo; elegí una miniatura para asignarla directamente.</div><div class="results" id="results-${x.i}"></div></div></article>`}
function refreshStats(){let decided=data.filter((x,i)=>decisions[i]||x.saved_decision).length;let unresolved=data.filter((x,i)=>{let d=getDecision(x,i);return d.type==='none'||d.type==='pending'}).length;$('#progress').textContent=`${decided} con decisión · ${unresolved} marcadas pendientes`}
function render(){let rows=getRows();const pages=Math.max(1,Math.ceil(rows.length/PAGE_SIZE));page=Math.min(page,pages-1);const visible=rows.slice(page*PAGE_SIZE,(page+1)*PAGE_SIZE);$('#count').textContent=`Mostrando ${rows.length} de ${data.length}`;$('#pageinfo').textContent=`Página ${page+1} de ${pages} · ${visible.length} tarjetas cargadas`;$('#prev').disabled=page===0;$('#next').disabled=page>=pages-1;$('#grid').innerHTML=visible.map(cardHTML).join('');refreshStats()}
function updateCard(i){const node=$('#item-'+i);if(!node)return;const x={...data[i],i,decision:getDecision(data[i],i)};node.outerHTML=cardHTML(x);refreshStats()}
function pick(i,j){decisions[i]={type:'picked',candidate:j};saveDecisions();updateCard(i)}function mark(i,type){let d=getDecision(data[i],i);decisions[i]={type:type==='correct'?'correct':'none',candidate:d.candidate??0};saveDecisions();updateCard(i)}
function searchRefs(i){const q=$('#manual-'+i).value.toLowerCase().trim(),box=$('#results-'+i);if(!q){box.innerHTML='';return}const found=refs.filter(r=>(r.name+' '+r.filename+' '+r.path).toLowerCase().includes(q)).slice(0,120);box.innerHTML=found.map(r=>`<div class="ref" onclick="chooseManual(${i},${refs.indexOf(r)})"><img loading="lazy" src="${esc(r.path)}" onerror="this.style.visibility='hidden'"><div class="name">${esc(r.name)}</div><div class="sub">${esc(r.filename)}</div></div>`).join('')||'<div class="sub">No se encontraron archivos con ese texto.</div>'}
function chooseManual(i,refIndex){const r=refs[refIndex];if(!r)return;const path=r.path,name=r.name,x=data[i];let c=x.candidates.findIndex(v=>v.path===path);if(c<0){x.candidates.push({name,path,filename:r.filename,score:null,phash:null,color:null,art_phash:null,manual:true});c=x.candidates.length-1}decisions[i]={type:'manual',candidate:c,manual_path:path,manual_name:name};saveDecisions();updateCard(i)}
document.querySelectorAll('[data-filter]').forEach(b=>b.onclick=()=>{filter=b.dataset.filter;page=0;document.querySelectorAll('[data-filter]').forEach(k=>k.classList.toggle('active',k===b));render()});let searchTimer;$('#search').oninput=()=>{clearTimeout(searchTimer);searchTimer=setTimeout(()=>{page=0;render()},180)};$('#sort').onchange=()=>{page=0;render()};$('#prev').onclick=()=>{if(page>0){page--;render();window.scrollTo({top:0,behavior:'instant'})}};$('#next').onclick=()=>{page++;render();window.scrollTo({top:0,behavior:'instant'})};$('#reset').onclick=()=>{if(confirm('¿Borrar todas las decisiones guardadas en este navegador para este HTML?')){decisions={};try{localStorage.removeItem(STORAGE_KEY)}catch(e){}render()}};
$('#export').onclick=()=>{saveDecisions();const merged=new Map((prior||[]).map(r=>[r.texture,r]));data.forEach((x,i)=>{const old=merged.get(x.texture)||{},d=getDecision(x,i),idx=d.candidate??0,c=x.candidates[idx]||null;let cand=[...x.candidates];const row={...old,...x,candidates:cand,selected_candidate:idx,selected_name:c?.name??null,selected_path:c?.path??null,decision:d,status:(d.type==='none'?'unmatched':d.type==='pending'?'pending':'selected')};if(d.type==='manual'&&c){row.selected_name=d.manual_name||c.name;row.selected_path=d.manual_path||c.path;row.manual_selection=true}merged.set(x.texture,row)});const out=[...merged.values()];const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,2)],{type:'application/json'}));a.download='correcciones_duel_transer.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
loadSaved();document.addEventListener('keydown',e=>{if(e.ctrlKey||e.altKey||e.metaKey)return;const tag=(e.target&&e.target.tagName||'').toLowerCase();if(['input','textarea','select'].includes(tag)||e.target.isContentEditable)return;const item=e.target.closest?.('.item')||document.querySelector('.item');if(!item)return;const i=Number(item.id.replace('item-',''));const moveNext=()=>{const next=item.nextElementSibling;if(next&&next.classList.contains('item')){next.querySelector('.actions button')?.focus({preventScroll:true});next.scrollIntoView({behavior:'smooth',block:'nearest'})}};if(e.key>='1'&&e.key<='5'){e.preventDefault();if(data[i]?.candidates[Number(e.key)-1])pick(i,Number(e.key)-1)}else if(e.key==='Enter'){e.preventDefault();mark(i,'correct');moveNext()}else if(e.key.toLowerCase()==='n'){e.preventDefault();mark(i,'none');moveNext()}else if(e.key==='ArrowRight'&&!$('#next').disabled){e.preventDefault();page++;render();window.scrollTo({top:0,behavior:'instant'})}else if(e.key==='ArrowLeft'&&!$('#prev').disabled){e.preventDefault();page--;render();window.scrollTo({top:0,behavior:'instant'})}});render();
</script></body></html>'''

def main():
    ap=argparse.ArgumentParser(description='Reconocimiento mejorado para texturas pequeñas: pHash, color y contornos a proporción conservada.')
    ap.add_argument('--texturas',default='Pequeñas',help='Carpeta de texturas (por defecto Pequeñas)')
    ap.add_argument('--referencias',default='cartas_completas',help='Carpeta de referencias completas')
    ap.add_argument('--salida',default='revision_duel_transer_pequenas_mejorado.html',help='HTML generado')
    ap.add_argument('--top',type=int,default=5,help='Candidatas por textura (predeterminado: 5)')
    ap.add_argument('--max-texturas',type=int,default=0,help='Solo probar N texturas; 0 = todas')
    ap.add_argument('--solo-pendientes',default='',help='JSON de revisión anterior; procesa solo pendientes/no coincidentes')
    args=ap.parse_args()
    texture_root=Path(args.texturas).resolve(); ref_root=Path(args.referencias).resolve(); out=Path(args.salida).resolve()
    if not texture_root.is_dir(): sys.exit(f'No existe la carpeta de texturas: {texture_root}')
    if not ref_root.is_dir(): sys.exit(f'No existe la carpeta de referencias: {ref_root}')
    textures=images_under(texture_root); ref_paths=images_under(ref_root); prior=[]; prior_by_name={}
    if args.solo_pendientes:
        pending_file=Path(args.solo_pendientes).resolve()
        try:
            prior=json.loads(pending_file.read_text(encoding='utf-8'))
            if not isinstance(prior,list): raise ValueError('el JSON debe contener una lista')
            prior_by_name={r.get('texture'):r for r in prior if isinstance(r,dict) and r.get('texture')}
            pending=set()
            for row in prior:
                d=row.get('decision') or {}; status=row.get('status')
                if d.get('type') in ('none','pending') or status in ('unmatched','pending'):
                    pending.add(row.get('texture'))
            textures=[p for p in textures if p.name in pending]
            print(f'Filtro de pendientes: {len(textures)} texturas encontradas en {pending_file.name}')
        except Exception as exc: sys.exit(f'No se pudo leer el JSON de pendientes {pending_file}: {exc}')
    if args.max_texturas: textures=textures[:args.max_texturas]
    if not textures: sys.exit('No encontré texturas pendientes. Revisá que el JSON tenga decisiones none/pending o status unmatched/pending.')
    if not ref_paths: sys.exit('No encontré imágenes de referencia.')
    print(f'Texturas a revisar: {len(textures)} | Referencias: {len(ref_paths)}')
    print('Calculando características mejoradas (proporción conservada, pHash, color y contornos)...')
    refs,rh,rc,rah,rac,re,rae=load_features(ref_paths,'Referencias')
    if not refs: sys.exit('No se pudo leer ninguna referencia.')
    ref_catalog=[{'name':card_name(p),'filename':p.name,'path':relpath(p,out.parent)} for p in refs]
    print('Comparando texturas con seis señales visuales...'); results=[]
    for i,p in enumerate(textures,1):
        img=read_img(p)
        if img is None: continue
        crop=art_crop(img)
        ranked=score_matches(perceptual_hash(img),color_signature(img),perceptual_hash(crop),color_signature(crop),edge_signature(img),edge_signature(crop),rh,rc,rah,rac,re,rae,min(5,max(1,args.top)))
        candidates=[]
        for idx,score,ps,cs,aps,acs,es,aes in ranked:
            candidates.append({'name':card_name(refs[idx]),'path':relpath(refs[idx],out.parent),'score':round(score*100,1),'phash':round(ps*100,1),'color':round(cs*100,1),'art_phash':round(aps*100,1),'art_color':round(acs*100,1),'edge':round(es*100,1),'art_edge':round(aes*100,1),'filename':refs[idx].name})
        old=prior_by_name.get(p.name,{})
        saved=old.get('decision') or {'type':'auto','candidate':0}
        if saved.get('type')=='pending': saved={'type':'auto','candidate':0}
        results.append({'texture':p.name,'texture_path':relpath(p,out.parent),'expected_name':old.get('expected_name'),'candidates':candidates,'saved_decision':saved})
        if i%50==0 or i==len(textures): print(f'  Texturas: {i}/{len(textures)}',flush=True)
    payload=json.dumps(results,ensure_ascii=False).replace('</','<\\/')
    catalog=json.dumps(ref_catalog,ensure_ascii=False).replace('</','<\\/')
    prior_json=json.dumps(prior,ensure_ascii=False).replace('</','<\\/')
    html=HTML.replace('__DATA__',payload).replace('__REFS__',catalog).replace('__PRIOR__',prior_json).replace('__TEXTURE_COUNT__',str(len(results))).replace('__REF_COUNT__',str(len(refs)))
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(html,encoding='utf-8')
    print(f'HTML generado: {out}\nAbrilo en el navegador para revisar las pendientes y exportar el JSON completo.')

if __name__=='__main__': main()
