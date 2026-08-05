const documents = document.querySelector('#documents');
const uploadForm = document.querySelector('#upload-form');
const fileInput = document.querySelector('#file-input');
const uploadMessage = document.querySelector('#upload-message');
const queryForm = document.querySelector('#query-form');
const questionInput = document.querySelector('#question');
const conversation = document.querySelector('#conversation');
const health = document.querySelector('#health');

function escapeHTML(value) {
  return String(value).replace(/[&<>'"]/g, char => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', "'":'&#39;', '"':'&quot;'}[char]));
}

function citationMetadata(hit) {
  const metadata = hit.metadata || {};
  const labels = [];
  if (Array.isArray(metadata.heading_path) && metadata.heading_path.length > 0) {
    labels.push(metadata.heading_path.join(' > '));
  }
  if (metadata.page) labels.push(`第 ${metadata.page} 页`);
  if (metadata.block_type === 'code' && metadata.language) labels.push(`${metadata.language} code`);
  if (hit.vector_rank) labels.push(`向量 #${hit.vector_rank}`);
  if (hit.lexical_rank) labels.push(`关键词 #${hit.lexical_rank}`);
  if (hit.rrf_score) labels.push(`RRF ${Number(hit.rrf_score).toFixed(4)}`);
  if (hit.rerank_rank) labels.push(`重排 #${hit.rerank_rank}`);
  if (hit.rerank_score) labels.push(`Rerank ${Number(hit.rerank_score).toFixed(3)}`);
  return labels.length > 0 ? `<span class="citation-meta">${escapeHTML(labels.join(' · '))}</span>` : '';
}

async function loadHealth() {
  try {
    const response = await fetch('/readyz');
    health.textContent = response.ok ? '服务就绪' : '数据库未就绪';
    health.classList.toggle('ready', response.ok);
  } catch (_) { health.textContent = '服务离线'; }
}

async function loadDocuments() {
  const response = await fetch('/v1/documents');
  if (!response.ok) throw new Error((await response.json()).error || '加载文档失败');
  const data = await response.json();
  if (!data.documents || data.documents.length === 0) {
    documents.innerHTML = '<p class="hint">还没有文档。</p>';
    return;
  }
  documents.innerHTML = data.documents.map(doc => `<div class="doc"><span class="doc-icon">▤</span><span class="doc-name" title="${escapeHTML(doc.name)}">${escapeHTML(doc.name)}</span><span class="doc-size">${Math.ceil(doc.size_bytes / 1024)} KB</span><button class="reindex-button" type="button" data-reindex-id="${escapeHTML(doc.id)}" aria-label="重建索引" title="重建索引">↻</button></div>`).join('');
}

documents.addEventListener('click', async event => {
  const button = event.target.closest('.reindex-button');
  if (!button || button.disabled) return;
  button.disabled = true;
  button.textContent = '…';
  try {
    const response = await fetch(`/v1/documents/${encodeURIComponent(button.dataset.reindexId)}/reindex`, { method: 'POST' });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || '重建索引失败');
    const job = await waitForJob(data.job.id);
    uploadMessage.textContent = `已重建索引：${job.name}`;
    uploadMessage.className = 'hint';
  } catch (error) {
    uploadMessage.textContent = error.message;
    uploadMessage.className = 'hint error';
  } finally {
    button.disabled = false;
    button.textContent = '↻';
  }
});

uploadForm.addEventListener('submit', async event => {
  event.preventDefault();
  const file = fileInput.files[0];
  if (!file) { uploadMessage.textContent = '请先选择一个文件'; uploadMessage.className = 'hint error'; return; }
  uploadMessage.textContent = '正在解析、切分并建立索引...'; uploadMessage.className = 'hint';
  const form = new FormData(); form.append('file', file);
  try {
    const response = await fetch('/v1/documents', { method: 'POST', body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || '上传失败');
    uploadMessage.textContent = '任务已提交，正在后台建立索引...';
    const job = await waitForJob(data.job.id);
    uploadMessage.textContent = `已建立索引：${job.name}`;
    fileInput.value = '';
    await loadDocuments();
  } catch (error) { uploadMessage.textContent = error.message; uploadMessage.className = 'hint error'; }
});

async function waitForJob(jobID) {
  for (let attempt = 0; attempt < 180; attempt += 1) {
    const response = await fetch(`/v1/jobs/${encodeURIComponent(jobID)}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || '读取索引任务失败');
    const job = data.job;
    if (job.status === 'completed') return job;
    if (job.status === 'failed') throw new Error(job.error_message || '文档索引失败');
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
  throw new Error('索引任务超时，请稍后刷新文档列表');
}

queryForm.addEventListener('submit', async event => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;
  const empty = conversation.querySelector('.empty-state'); if (empty) empty.remove();
  conversation.insertAdjacentHTML('beforeend', `<div class="message user"><div class="message-label">Question</div><div class="bubble">${escapeHTML(question)}</div></div>`);
  const pending = document.createElement('div'); pending.className = 'message assistant'; pending.innerHTML = '<div class="message-label">MiniRAG</div><div class="bubble">检索中...</div>'; conversation.appendChild(pending); conversation.scrollTop = conversation.scrollHeight;
  questionInput.value = '';
  try {
    const response = await fetch('/v1/chat/stream', { method: 'POST', headers: {'Content-Type': 'application/json', 'Accept': 'text/event-stream'}, body: JSON.stringify({question, top_k: 5}) });
    if (!response.ok) {
      const data = await response.json();
      throw new Error(data.error || '查询失败');
    }
    if (!response.body) throw new Error('浏览器不支持流式响应');
    const bubble = pending.querySelector('.bubble');
    let answer = '';
    let citations = [];
    let done = null;
    await readSSE(response.body, (event, data) => {
      if (event === 'token') {
        answer += data.content || '';
        bubble.textContent = answer;
        conversation.scrollTop = conversation.scrollHeight;
      } else if (event === 'citations') {
        citations = data.citations || [];
      } else if (event === 'done') {
        done = data;
      } else if (event === 'error') {
        throw new Error(data.error || '查询失败');
      }
    });
    if (!done) throw new Error('流式响应未正常结束');
    const citationHTML = citations.map((hit, index) => `<div class="citation"><strong>[${index + 1}] ${escapeHTML(hit.document_name)}</strong>${citationMetadata(hit)}<br>${escapeHTML(hit.content.slice(0, 220))}${hit.content.length > 220 ? '…' : ''}</div>`).join('');
    pending.innerHTML = `<div class="message-label">MiniRAG · ${done.total_ms}ms</div><div class="bubble">${escapeHTML(answer)}</div><div class="citations">${citationHTML || '<span class="hint">没有召回来源</span>'}</div>`;
  } catch (error) { pending.innerHTML = `<div class="message-label">MiniRAG</div><div class="bubble error">${escapeHTML(error.message)}</div>`; }
  conversation.scrollTop = conversation.scrollHeight;
});

async function readSSE(body, onEvent) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const {value, done} = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), {stream: !done});
    let boundary;
    while ((boundary = buffer.indexOf('\n\n')) >= 0) {
      const block = buffer.slice(0, boundary).replace(/\r/g, '');
      buffer = buffer.slice(boundary + 2);
      const event = parseSSEBlock(block);
      if (event) onEvent(event.name, JSON.parse(event.data));
    }
    if (done) break;
  }
}

function parseSSEBlock(block) {
  let name = 'message';
  const data = [];
  for (const line of block.split('\n')) {
    if (line.startsWith('event:')) name = line.slice(6).trim();
    if (line.startsWith('data:')) data.push(line.slice(5).trim());
  }
  return data.length > 0 ? {name, data: data.join('\n')} : null;
}

document.querySelector('#refresh').addEventListener('click', () => loadDocuments().catch(error => { documents.innerHTML = `<p class="hint error">${escapeHTML(error.message)}</p>`; }));
loadHealth(); loadDocuments().catch(() => {});
