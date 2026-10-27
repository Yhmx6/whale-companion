/* UI 交互的端到端验证：走真实用户路径，截图存证。
 *
 * ## 为什么不只截首页
 *
 * 首页是空的（没照片），真正容易出问题的是**上传之后**：照片能不能撑住
 * 容器、生成按钮会不会变亮、高级区展开后挤不挤、手机端会不会横向溢出。
 * 这些只有真的走一遍才看得见。
 *
 * ## 用 puppeteer-core + 已有的 chrome-headless-shell
 *
 * 本机没有 Chrome，但 `~/.cache/puppeteer/chrome-headless-shell` 里有一个。
 * puppeteer-core 不绑浏览器版本，指定 executablePath 就能用。
 *
 * 用法：
 *   node tools/ui_check.mjs                # 需要服务已在 8848 跑着
 *   PORT=8855 node tools/ui_check.mjs
 */

import puppeteer from 'puppeteer-core';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const PORT = process.env.PORT || '8848';
const BASE = `http://127.0.0.1:${PORT}`;
const OUT = path.resolve('output/ui');
const CHROME = path.join(
  os.homedir(),
  '.cache/puppeteer/chrome-headless-shell/155.0.8059.39',
  'chrome-headless-shell-win64/chrome-headless-shell.exe');

/* 拿一张现成的场景图当"用户上传的照片"，避免为了测试再造图。
 * 默认用**空场景**（没人脸），这样截图可以直接放进公开文档；
 * 想换成带人物的图：SAMPLE=assets/scenes/xxx.jpg node tools/ui_check.mjs */
const SAMPLE = path.resolve(
  process.env.SAMPLE || 'assets/scenes/batch3/c05_bench_empty.png');

const shots = [];
async function shot(page, name) {
  fs.mkdirSync(OUT, { recursive: true });
  const p = path.join(OUT, name);
  await page.screenshot({ path: p, fullPage: false });
  shots.push(p);
  console.log('  ->', path.relative(process.cwd(), p));
}

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: true,
  args: ['--no-sandbox', '--disable-gpu', '--hide-scrollbars'],
});

try {
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(m.text());
  });

  await page.setViewport({ width: 1100, height: 900 });
  await page.goto(BASE, { waitUntil: 'networkidle2' });
  await new Promise((r) => setTimeout(r, 600));
  console.log('[1] 空状态');
  await shot(page, '10_empty.png');

  /* 上传：input[type=file] 是 hidden 的，直接给它塞文件路径 */
  const input = await page.$('#file');
  await input.uploadFile(SAMPLE);
  await new Promise((r) => setTimeout(r, 900));
  console.log('[2] 上传后');
  await shot(page, '11_uploaded.png');

  /* 生成按钮应该已经可用 */
  const disabled = await page.$eval('#go', (el) => el.disabled);
  console.log('    生成按钮 disabled =', disabled, disabled ? '（应为 false）' : '（正确）');

  /* 展开高级 */
  await page.$eval('#adv', (el) => { el.open = true; });
  await new Promise((r) => setTimeout(r, 400));
  /* 高级区在照片下方，窗口不够高会被截掉一半。滚过去再截，
     否则截图里只能看到"渲染方式"的第一行，看不出下面的表情网格排得对不对。 */
  await page.$eval('#adv', (el) => el.scrollIntoView({ block: 'start' }));
  await new Promise((r) => setTimeout(r, 300));
  console.log('[3] 高级展开');
  await shot(page, '12_advanced.png');

  /* 切到"精确尺寸"，表情该隐藏、动作该出现 */
  await page.click('#routeSeg button[data-v="sprite"]');
  await new Promise((r) => setTimeout(r, 400));
  const ui = await page.evaluate(() => ({
    moodHidden: document.getElementById('moodField').hidden,
    actionShown: !document.getElementById('actionField').hidden,
    engineHidden: document.getElementById('engineField').hidden,
    label: document.getElementById('goLabel').textContent,
  }));
  console.log('[4] 切到精确尺寸');
  console.log('    表情隐藏 =', ui.moodHidden, '| 动作出现 =', ui.actionShown,
    '| 引擎隐藏 =', ui.engineHidden, '| 按钮 =', ui.label);
  await shot(page, '13_sprite.png');

  /* 回到自动，手机尺寸 */
  await page.click('#routeSeg button[data-v="auto"]');
  await page.setViewport({ width: 390, height: 844, deviceScaleFactor: 2 });
  await new Promise((r) => setTimeout(r, 500));
  console.log('[5] 移动端');
  const overflow = await page.evaluate(() =>
    document.documentElement.scrollWidth > window.innerWidth + 1);
  console.log('    横向溢出 =', overflow, overflow ? '（应修复）' : '（正确）');
  await shot(page, '14_mobile.png');

  if (errors.length) {
    console.log('\n页面报错：');
    errors.slice(0, 5).forEach((e) => console.log('  ', e));
  } else {
    console.log('\n页面无 JS 报错');
  }
  const bad = errors.length > 0 || disabled || overflow || !ui.moodHidden
    || !ui.actionShown || !ui.engineHidden;
  console.log(bad ? '\n有项目未通过' : '\n全部通过');
  process.exitCode = bad ? 1 : 0;
} finally {
  await browser.close();
}
