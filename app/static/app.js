/* 一键出图：上传 → 点一下 → 出图。
 *
 * ## 这一版砍掉了什么
 *
 * 原来界面上有玩法、依托物、光向、参照物、动作、候选数、引擎七个参数。
 * 但 `run_guided` 内部**本来就能自己判断**位置、动作、表情、多大——
 * 那些参数大多只对已经废弃的 solo 路径有用。
 *
 * 所以全砍了。默认一个参数都不传，后端 `route=auto` 走引导生成，
 * 规划器看照片决定一切。想微调的人展开「高级」。
 *
 * 拖框也去掉了：引导生成不靠用户指定的框定位，它自己规划。
 * 留着反而是误导——用户会以为"框画得准，结果就准"。
 */

const state = {
  img: null,
  route: 'auto',                  // auto / sprite
  mood: 'random',                 // random（默认）/ auto（按场景）/ 具体 id
  engine: 'doubao-seedream-5-0-flash-260915',
  action: '',
  job: null, timer: null, ticker: null,
  t0: 0,
  actions: [], moods: [],
};

const $ = (id) => document.getElementById(id);
const canvas = $('canvas');
const ctx = canvas.getContext('2d');

const ENGINES = {
  'doubao-seedream-5-0-flash-260915': { name: 'Flash', cost: '0.12 元 / 张' },
  'doubao-seedream-5-0-pro-260628': { name: 'Pro', cost: '0.30 元 / 张' },
};

/** 去掉提示词里的 markdown 标记，供展示层使用。 */
function plain(s) {
  return (s || '').replace(/\*\*/g, '').replace(/`/g, '');
}

/* ================= 加载清单 ================= */

async function loadActions() {
  try {
    const r = await fetch('/api/actions');
    const j = await r.json();
    state.actions = j.enabled || [];
    const sel = $('action');
    sel.innerHTML = '<option value="">让模型挑</option>';
    state.actions.forEach((a) => {
      const o = document.createElement('option');
      o.value = a.id;
      o.textContent = a.name;
      o.title = a.note || '';
      sel.appendChild(o);
    });
  } catch (e) {
    note('动作列表加载失败：' + e.message, 'err');
  }
}

async function loadMoods() {
  const box = $('moodList');
  try {
    const r = await fetch('/api/moods');
    const j = await r.json();
    state.moods = j.moods || [];
    box.innerHTML = '';

    /* 第一个格是「随机」，默认选中。9 个表情 + 1 个随机 = 10 个，
       正好排满 5×2，不会出现最后一行缺一个的锯齿。
       「按场景挑」是另一档行为，放在标题行右边当链接——
       它和"随机"是互斥的两种模式，混在同一排格子容易分不清。 */
    const rnd = document.createElement('button');
    rnd.dataset.v = 'random';
    rnd.className = 'on';
    rnd.title = '随机挑一个（会避开和当前动作不搭的表情）';
    rnd.innerHTML = '<span class="m-name">随机</span><span class="m-id">random</span>';
    box.appendChild(rnd);

    state.moods.forEach((m) => {
      const b = document.createElement('button');
      b.dataset.v = m.id;
      /* `when` 是写给规划器的提示词，带 markdown 加粗。展示前剥掉星号，
         并且用 title 属性（不是 innerHTML）承载，避免当成 HTML 解析。 */
      b.title = plain(m.when);
      b.innerHTML = `<span class="m-name">${m.name}</span>`
        + `<span class="m-id">${m.id}</span>`;
      box.appendChild(b);
    });

    box.addEventListener('click', (e) => {
      const btn = e.target.closest('button');
      if (!btn) return;
      [...box.querySelectorAll('button')].forEach((x) => x.classList.remove('on'));
      btn.classList.add('on');
      state.mood = btn.dataset.v || 'random';
      updateMoodNote();
      syncAdvNote();
    });

    if (state.mood !== 'auto') setSeg('moodList', state.mood);
    updateMoodNote();
  } catch (e) {
    box.innerHTML = '';
    const p = document.createElement('p');
    p.className = 'note err';
    p.textContent = '表情列表加载失败：' + e.message;
    box.appendChild(p);
  }
}

/* 「按场景挑」：清掉网格里的选中（它不是网格里的某一格），
   把 mood 设成 'auto'，后端会用规划器看完照片的判断。 */
$('moodScene').addEventListener('click', () => {
  [...$('moodList').querySelectorAll('button')].forEach((b) => b.classList.remove('on'));
  state.mood = 'auto';
  updateMoodNote();
  syncAdvNote();
});

function updateMoodNote() {
  const el = $('moodNote');
  if (state.mood === 'auto') {
    el.textContent = '按场景挑：模型会看光线、空间和有没有人来决定。';
    el.className = 'note ok';
    return;
  }
  if (state.mood === 'random' || !state.mood) {
    el.textContent = '默认随机——同一个动作多出几张，脸都不一样。';
    el.className = 'note';
    return;
  }
  const hit = state.moods.find((m) => m.id === state.mood);
  el.textContent = hit
    ? `已指定「${hit.name}」：${plain(hit.when)}`
    : '默认随机。';
  el.className = 'note ok';
}

/* ================= 上传 ================= */

$('drop').addEventListener('click', () => $('file').click());
$('file').addEventListener('change', (e) => {
  if (e.target.files[0]) loadFile(e.target.files[0]);
  e.target.value = '';           // 允许重复选同一张
});
['dragover', 'dragenter'].forEach((ev) =>
  $('drop').addEventListener(ev, (e) => { e.preventDefault(); $('drop').classList.add('over'); }));
['dragleave', 'drop'].forEach((ev) =>
  $('drop').addEventListener(ev, (e) => { e.preventDefault(); $('drop').classList.remove('over'); }));
$('drop').addEventListener('drop', (e) => {
  const f = e.dataTransfer.files[0]; if (f) loadFile(f);
});

$('reset').addEventListener('click', (e) => {
  e.stopPropagation();
  state.img = null;
  $('canvasWrap').hidden = true; $('drop').hidden = false;
  $('results').innerHTML = '';
  refreshGo();
});

function loadFile(file) {
  if (!file.type.startsWith('image/')) return;
  const url = URL.createObjectURL(file);
  const im = new Image();
  im.onload = () => {
    state.img = im;
    $('drop').hidden = true; $('canvasWrap').hidden = false;
    fitCanvas(); paint(); updateBadge(); refreshGo();
  };
  im.src = url;
}

function fitCanvas() {
  /* 位图尺寸 = 原图尺寸（限制最大宽），显示尺寸交给 CSS 的 max-width/max-height。
     只用来预览，不再承载"画框"的功能。 */
  const maxW = 1400;
  const r = Math.min(1, maxW / state.img.width);
  canvas.width = Math.round(state.img.width * r);
  canvas.height = Math.round(state.img.height * r);
}

function paint() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (state.img) ctx.drawImage(state.img, 0, 0, canvas.width, canvas.height);
}

function updateBadge() {
  const badge = $('canvasBadge');
  if (!state.img) { badge.hidden = true; return; }
  badge.hidden = false;
  badge.textContent = `${state.img.width} × ${state.img.height}`;
}

/* ================= 控件 ================= */

function bindSeg(id, key, after) {
  const el = $(id);
  if (!el) return;
  el.addEventListener('click', (e) => {
    const btn = e.target.closest('button');
    if (!btn) return;
    [...el.querySelectorAll('button')].forEach((b) => b.classList.remove('on'));
    btn.classList.add('on');
    state[key] = btn.dataset.v;
    if (after) after();
  });
}

function setSeg(id, value) {
  const el = $(id);
  if (!el) return;
  const btn = el.querySelector(`button[data-v="${value}"]`);
  if (!btn) return;
  [...el.querySelectorAll('button')].forEach((b) => b.classList.remove('on'));
  btn.classList.add('on');
}

bindSeg('routeSeg', 'route', () => { syncRoute(); refreshGo(); });
bindSeg('engineSeg', 'engine', updateEnginePill);

const ROUTE_NOTE = {
  auto: '位置、动作、表情、多大都由模型看照片决定。她可以被人物挡住、会避开桌上的杯子。'
      + '代价：尺寸会偏大 1.1~1.8 倍。',
  sprite: '立绘 + 程序化合成。尺寸精确（要 60cm 就是 60cm），0 元、3~5 秒。'
        + '代价：永远贴在最前面，没有遮挡；表情只有立绘上那一种。',
};

/** 路线切换时更新界面：sprite 需要动作且没有表情；auto 相反。 */
function syncRoute() {
  const precise = state.route === 'sprite';
  $('moodField').hidden = precise;
  $('actionField').hidden = !precise;
  $('engineField').hidden = precise;
  $('routeNote').textContent = ROUTE_NOTE[state.route] || '';
  $('routeNote').className = 'note';

  const sel = $('action');
  if (sel.options.length) {
    sel.options[0].textContent = precise ? '请选择一个动作' : '让模型挑';
  }
  if (precise && !state.action && state.actions.length) {
    state.action = state.actions[0].id;
    sel.value = state.action;
  }

  $('goLabel').textContent = precise ? '生成（精确尺寸）' : '生成';
  $('goHint').textContent = precise
    ? '她会被贴在最前面，尺寸精确，不调用生成模型'
    : '她该站哪、做什么表情，都交给模型判断';
  syncAdvNote();
}

/** 高级区右上角那行小字：只有在真的改过默认值时才提示"已自定义"。
 *  默认的 mood 是 'random'（真值），直接用它判空会把每条都标成"已自定义"。 */
function syncAdvNote() {
  const custom = state.route === 'sprite'
    || (state.mood && state.mood !== 'random');
  $('advNote').textContent = custom ? '已自定义' : '默认全自动，不用动';
}

function updateEnginePill() {
  const e = ENGINES[state.engine] || { name: '未知', cost: '—' };
  $('engName').textContent = e.name;
  $('engCost').textContent = e.cost;
}

$('action').addEventListener('change', (e) => {
  state.action = e.target.value;
  refreshGo();
});

$('autoMatch').addEventListener('click', async () => {
  if (!state.img) { note('先选一张照片。', 'err'); return; }
  $('autoMatch').disabled = true;
  note('正在看照片…', '');
  try {
    const image = canvas.toDataURL('image/jpeg', 0.85);
    const r = await fetch('/api/match', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image }),
    });
    const j = await r.json();
    if (j.error) { note('匹配失败：' + j.error, 'err'); return; }
    const hit = state.actions.find((a) => a.id === j.action_id);
    if (hit) { $('action').value = hit.id; state.action = hit.id; }
    const FR = { closeup: '特写', half: '半身', full: '全身' };
    const fr = FR[j.framing] || '';
    note(`推荐「${hit ? hit.name : j.action_id}」`
      + (j.confidence ? `（${Math.round(j.confidence * 100)}%）` : '')
      + `：${j.reason || ''}` + (fr ? `　景别：${fr}` : ''), 'ok');
  } catch (e) {
    note('匹配失败：' + e.message, 'err');
  } finally {
    $('autoMatch').disabled = false;
  }
});

function note(text, cls) {
  const el = $('matchNote');
  el.textContent = text;
  el.className = 'note' + (cls ? ' ' + cls : '');
}

function refreshGo() {
  /* 一键：只要选了照片就能点。其他都有默认值。 */
  $('go').disabled = !state.img;
}

/* ================= 生成与进度 ================= */

const STEPS = [
  { name: '解析照片', at: 10 },
  { name: '定位置动作', at: 20 },
  { name: '生成', at: 44 },
  { name: '导出', at: 90 },
];

function renderSteps(pct, failed) {
  const wrap = $('steps');
  if (!wrap.children.length) {
    STEPS.forEach((s) => {
      const d = document.createElement('div');
      d.className = 'st'; d.textContent = s.name;
      wrap.appendChild(d);
    });
  }
  let cur = 0;
  STEPS.forEach((s, i) => { if (pct >= s.at) cur = i; });
  [...wrap.children].forEach((el, i) => {
    el.className = 'st' + (failed ? '' : (i < cur ? ' done' : i === cur ? ' now' : ''));
  });
}

function setProgress(pct, text, failed) {
  $('barFill').style.width = Math.max(0, Math.min(100, pct)) + '%';
  $('pmText').textContent = text;
  renderSteps(pct, failed);
}

function startTick() {
  state.t0 = Date.now();
  clearInterval(state.ticker);
  state.ticker = setInterval(() => {
    $('pmTick').textContent = ((Date.now() - state.t0) / 1000).toFixed(1) + 's';
  }, 100);
}

function stopTick() {
  clearInterval(state.ticker);
  $('pmTick').textContent = ((Date.now() - state.t0) / 1000).toFixed(1) + 's';
}

$('go').addEventListener('click', async () => {
  if (!state.img) return;
  $('go').disabled = true;
  $('progress').hidden = false;
  $('results').innerHTML = '';
  setProgress(4, '上传照片');
  startTick();

  const image = canvas.toDataURL('image/jpeg', 0.92);
  try {
    const r = await fetch('/api/generate', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image,
        /* 一键出图：只传图和用户在高级里改过的项。
           box / surface / light / anchor / mode 全部不传——
           后端 route=auto 会自己规划，传了也用不上。 */
        route: state.route,
        mood: state.mood,
        action: state.action,
        engine: state.engine,
      }),
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || '请求失败');
    state.job = j.job_id;
    poll();
  } catch (err) {
    setProgress(0, '失败：' + err.message, true);
    stopTick(); refreshGo();
    renderError(err.message);
  }
});

function poll() {
  clearInterval(state.timer);
  state.timer = setInterval(async () => {
    try {
      const r = await fetch('/api/job/' + state.job);
      const j = await r.json();
      setProgress(j.pct || 0, j.stage || '处理中');
      if (j.status === 'done') {
        clearInterval(state.timer); stopTick(); refreshGo();
        render(j.result);
      } else if (j.status === 'error') {
        clearInterval(state.timer); stopTick(); refreshGo();
        setProgress(0, '失败：' + (j.error || '未知错误'), true);
        renderError(j.error, j.trace);
      }
    } catch (e) { /* 单次轮询失败不算死，等下一轮 */ }
  }, 1000);
}

/* ================= 结果渲染 ================= */

const MOOD_CN = { happy: '雀跃', shy: '害羞', curious: '好奇', smug: '得意',
  sleepy: '困倦', surprised: '惊讶', tender: '温柔', determined: '鼓劲', pout: '嘟嘴' };

function render(result) {
  const box = $('results');
  box.innerHTML = '';
  if (!result) return;
  if (result.error) { renderError(result.error); return; }

  const run = result.run_id;
  const cands = result.candidates || [];
  const first = cands[0];
  if (!first) { renderError('没有产物'); return; }

  const isGuided = (result.mode || '') === 'guided';

  const head = document.createElement('div');
  head.className = 'res-head';
  head.innerHTML = `<h2>${isGuided ? '生成结果' : '精确尺寸结果'}</h2>
    <span class="meta">${result.size || ''}${result.height_cm
      ? `　她 ${result.height_cm} cm` : ''}</span>`;
  box.appendChild(head);

  const body = document.createElement('div');
  body.className = 'res-body';

  const shot = document.createElement('div');
  shot.className = 'shot';
  shot.innerHTML = `
    <img id="mainShot" src="/files/${run}/${first.file}" alt="成品">
    <div class="shot-bar">
      <span class="idx" id="shotIdx"></span>
      <a href="/files/${run}/${first.file}" download>
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor"
             stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
          <path d="M12 3v12M7 11l5 5 5-5M4 21h16"/></svg>
        下载
      </a>
    </div>`;
  body.appendChild(shot);
  body.appendChild(buildDiag(result, isGuided));
  box.appendChild(body);

  if (cands.length > 1) {
    const thumbs = document.createElement('div');
    thumbs.className = 'thumbs';
    cands.forEach((c, i) => {
      const b = document.createElement('button');
      if (i === 0) b.classList.add('on');
      b.innerHTML = `<img src="/files/${run}/${c.file}" alt="">`;
      b.addEventListener('click', () => {
        $('mainShot').src = `/files/${run}/${c.file}`;
        $('shotIdx').textContent = `第 ${i + 1} 张 / 共 ${cands.length} 张`;
        [...thumbs.children].forEach((x) => x.classList.remove('on'));
        b.classList.add('on');
      });
      thumbs.appendChild(b);
    });
    box.appendChild(thumbs);
  }

  setProgress(100, '完成');
}

function buildDiag(result, isGuided) {
  const el = document.createElement('div');
  el.className = 'diag';
  const pl = result.placement || {};
  const mo = result.mood || {};
  const rows = [];

  if (isGuided) {
    const srcMap = { plan: '模型选的', anchor_map: '兜底', scene_check: '按场景调整',
                     forced: '手动指定', default: '默认' };
    rows.push(['动作', `${pl.sprite_action || '—'}`
      + (pl.action_source ? `（${srcMap[pl.action_source] || pl.action_source}）` : '')]);

    const mid = mo.id || '';
    const SRC = { forced: '手动', random: '随机', scene: '按场景', plan: '按场景' };
    rows.push(['表情', mid
      ? `${MOOD_CN[mid] || mid}（${SRC[mo.source] || '自动'}）`
      : '—']);

    const plan = pl.h_pct, got = pl.measured_pct, err = pl.size_err;
    if (plan != null && got != null) {
      const good = err && Math.abs(err - 1) <= 0.25;
      rows.push(['尺寸', `计划 ${plan}% → 实测 ${got}%（${err}×）`, good ? 'good' : 'bad']);
    } else if (plan != null) {
      rows.push(['尺寸', `计划 ${plan}%`]);
    }

    const col = pl.collision || {};
    if (col.conflicts && col.conflicts.length) {
      rows.push(['避让', `挪不开：${col.conflicts.map((c) => c.name).join('、')}`, 'bad']);
    } else if (col.moved_px > 8) {
      const sits = col.sits_on ? `，坐于${col.sits_on.name}` : '';
      rows.push(['避让', `避开障碍，横向挪了 ${Math.round(col.moved_px)}px${sits}`,
        col.forced ? 'bad' : 'good']);
    } else {
      rows.push(['避让', '无冲突', 'good']);
    }

    const objs = (result.scene2d && result.scene2d.objects) || [];
    if (objs.length) {
      rows.push(['场景', `${objs.length} 件：`
        + objs.slice(0, 3).map((o) => o.name).join('、')
        + (objs.length > 3 ? '…' : '')]);
    }
  } else {
    const c = (result.candidates || [])[0] || {};
    if (c.defringe_replaced_ratio != null) {
      rows.push(['白边修正', `${(c.defringe_replaced_ratio * 100).toFixed(1)}%`]);
    }
    if (c.qc) {
      if (c.qc.changed_px_ratio != null) {
        rows.push(['改动像素', `${(c.qc.changed_px_ratio * 100).toFixed(1)}%`]);
      }
      if (c.qc.contact_band_delta != null) {
        rows.push(['接触暗部', c.qc.contact_band_delta.toFixed(1)]);
      }
    }
  }

  const h = document.createElement('h3');
  h.textContent = '这张图的情况';
  el.appendChild(h);
  rows.forEach(([k, v, cls]) => {
    const d = document.createElement('div');
    d.className = 'kv';
    d.innerHTML = `<span class="k">${k}</span><span class="v ${cls || ''}">${v}</span>`;
    el.appendChild(d);
  });

  const plan2 = result.plan || {};
  if (plan2.where || plan2.moment) {
    const sep = document.createElement('div');
    sep.className = 'sep';
    el.appendChild(sep);
    if (plan2.where) {
      const d = document.createElement('div');
      d.className = 'kv';
      d.innerHTML = `<span class="k">位置</span><span class="v">${plan2.where}</span>`;
      el.appendChild(d);
    }
    if (plan2.moment) {
      const d = document.createElement('div');
      d.className = 'kv';
      d.innerHTML = `<span class="k">此刻</span><span class="v">${plan2.moment}</span>`;
      el.appendChild(d);
    }
  }

  const warns = result.warnings || [];
  if (warns.length) {
    const ul = document.createElement('ul');
    ul.className = 'warns';
    warns.forEach((w) => {
      const li = document.createElement('li');
      li.textContent = w;
      ul.appendChild(li);
    });
    el.appendChild(ul);
  }
  return el;
}

function renderError(msg, trace) {
  const box = $('results');
  box.innerHTML = '';
  const d = document.createElement('div');
  d.className = 'err-box';
  d.textContent = msg || '未知错误';
  if (trace) {
    const pre = document.createElement('pre');
    pre.style.cssText = 'margin:10px 0 0;font-size:11px;opacity:.7;white-space:pre-wrap';
    pre.textContent = trace;
    d.appendChild(pre);
  }
  box.appendChild(d);
}

/* ================= 启动 ================= */

function applyPreset() {
  const q = new URLSearchParams(location.search);
  const route = q.get('route');
  if (route === 'auto' || route === 'sprite' || route === 'guided') {
    state.route = route === 'guided' ? 'auto' : route;
    setSeg('routeSeg', state.route);
  }
  const mood = q.get('mood');
  /* 支持三档：random / auto / 具体 id。'auto' 不在网格里，
     所以不能走 setSeg（它按 data-v 找按钮，找不到就静默返回）。 */
  if (mood === 'auto') { state.mood = 'auto'; }
  else if (mood) { state.mood = mood; }
  const engine = q.get('engine');
  if (engine && ENGINES[engine]) { state.engine = engine; setSeg('engineSeg', engine); }
  if (q.get('adv') === '1') $('adv').open = true;
}

applyPreset();
loadActions();
loadMoods();
updateEnginePill();
syncRoute();
refreshGo();
