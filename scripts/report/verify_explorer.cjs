// Verify the report's real exported data and interactive behavior in a DOM engine.
// This does not replace a visual browser preview.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const {JSDOM} = require('jsdom');
const docs = path.resolve(__dirname, '../../docs');
const dom = new JSDOM(fs.readFileSync(path.join(docs, 'index.html'), 'utf8'), {runScripts: 'outside-only'});
const {window} = dom;
window.eval(fs.readFileSync(path.join(docs, 'report.js'), 'utf8'));
const $ = id => window.document.getElementById(id);
const change = (id, value, event = 'change') => {$(id).value = value; $(id).dispatchEvent(new window.Event(event));};
assert.match($('gene-count').textContent, /^58 genes/);
assert.equal($('gene-rows').querySelectorAll('tr').length, 12);
assert.equal($('detail-gene').textContent, 'CLEC5A');
assert.match($('detail-summary').textContent, /Model Δ \+0\.454; observed Δ \+0\.489; adjusted observed Δ \+0\.572/);
assert.match($('detail-chart').querySelector('svg').getAttribute('aria-label'), /Observed: 0\.246, 1\.075, 0\.714, 0\.735/);
for (const [value, count] of [['cohort',107], ['stable',614], ['all',2000]]) {
  change('gene-filter', value);
  assert.match($('gene-count').textContent, new RegExp(`^${count.toLocaleString('en-US')} genes`));
}
change('gene-search','CD163','input');
assert.match($('gene-count').textContent, /^2 genes/);
[...$('gene-rows').querySelectorAll('button')].find(el=>el.textContent==='CD163').click();
assert.equal($('detail-gene').textContent, 'CD163');
assert.match($('detail-caveat').textContent, /reverses after covariate adjustment/);
assert.match($('detail-summary').textContent, /adjusted observed Δ −0\.099/);
assert.equal($('gene-rows').querySelector('button').getAttribute('aria-pressed'), 'true');
change('gene-filter', 'robust');
assert.equal([...$('gene-rows').querySelectorAll('button')].some(el=>el.textContent==='CD163'),false);
change('gene-filter','all');
change('gene-search','USP9Y','input');
$('gene-rows').querySelector('button').click();
assert.match($('detail-caveat').textContent, /Y-linked/);
change('gene-search','TREM2','input');
assert.match($('gene-count').textContent, /^0 genes/);
assert.equal($('next-page').disabled, true);
change('gene-search','','input');
$('next-page').click();
assert.match($('gene-count').textContent,/showing 13–24/);
change('gene-sort','gene');
assert.equal($('page-state').textContent,'1 / 167');
const names = [...$('gene-rows').querySelectorAll('button')].map(el=>el.textContent);
assert.deepEqual(names, [...names].sort((a,b)=>a.localeCompare(b)));
change('gene-search','<script>','input');
assert.match($('gene-count').textContent,/^0 genes/);
assert.equal(window.document.querySelectorAll('#gene-rows script').length,0);
dom.window.close();
console.log('PASS: gene evidence counts, displayed numerical effects, accessible stage profiles, selection, sensitivity exclusions, absent genes, pagination, sorting, and literal search.');
