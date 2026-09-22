(() => {
  if (document.getElementById('pet-care-assistant')) return;
  const API = 'https://pet-care-rag-demo.onrender.com/api/chat';
  const panel = document.createElement('aside');
  panel.id = 'pet-care-assistant';
  panel.innerHTML = `<header><span>🐱🐶 宠物寄养智慧客服</span><button class="pet-toggle" title="收起">−</button></header><main><textarea placeholder="先选中顾客消息，或直接粘贴到这里"></textarea><label class="auto-send"><input id="pet-auto-send" type="checkbox" checked> 普通问题自动发送</label><label class="auto-send"><input id="pet-watch" type="checkbox"> 自动监听新消息</label><div><button id="pet-generate">生成回复</button><button class="secondary" id="pet-use-selection">读取选中文本</button></div><div class="status">普通咨询会自动回复；预订、取消、付款和健康问题需要人工确认。</div><div class="reply" hidden></div></main>`;
  document.body.appendChild(panel);
  const header = panel.querySelector('header');
  const toggle = panel.querySelector('.pet-toggle');
  toggle.onclick = (event) => { event.stopPropagation(); panel.classList.toggle('collapsed'); toggle.textContent = panel.classList.contains('collapsed') ? '+' : '−'; toggle.title = panel.classList.contains('collapsed') ? '展开' : '收起'; };
  let dragging = false, offsetX = 0, offsetY = 0;
  header.onpointerdown = (event) => { if (event.target === toggle) return; dragging = true; const rect = panel.getBoundingClientRect(); offsetX = event.clientX - rect.left; offsetY = event.clientY - rect.top; header.setPointerCapture(event.pointerId); };
  header.onpointermove = (event) => { if (!dragging) return; panel.style.left = `${Math.max(4, event.clientX - offsetX)}px`; panel.style.top = `${Math.max(4, event.clientY - offsetY)}px`; panel.style.right = 'auto'; panel.style.bottom = 'auto'; };
  header.onpointerup = () => { if (!dragging) return; dragging = false; localStorage.setItem('pet-care-panel-position', JSON.stringify({left: panel.style.left, top: panel.style.top})); };
  try { const saved = JSON.parse(localStorage.getItem('pet-care-panel-position') || 'null'); if (saved?.left && saved?.top) { panel.style.left = saved.left; panel.style.top = saved.top; panel.style.right = 'auto'; panel.style.bottom = 'auto'; } } catch (_) {}
  const textarea = panel.querySelector('textarea');
  const status = panel.querySelector('.status');
  const replyBox = panel.querySelector('.reply');
  const processedMessages = new Set();
  function fillReplyBox(text) {
    const candidates = [...document.querySelectorAll('textarea, [contenteditable="true"], input[type="text"]')];
    const target = candidates.find(node => /发消息|回复|输入/.test(node.getAttribute('placeholder') || node.getAttribute('aria-label') || '')) || candidates.find(node => node.offsetParent !== null && node !== textarea);
    if (!target) return null;
    target.focus();
    if (target.isContentEditable) {
      target.textContent = text;
      target.dispatchEvent(new InputEvent('input', {bubbles:true, inputType:'insertText', data:text}));
    } else {
      const setter = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(target), 'value')?.set;
      setter ? setter.call(target, text) : target.value = text;
      target.dispatchEvent(new Event('input', {bubbles:true}));
      target.dispatchEvent(new Event('change', {bubbles:true}));
    }
    return target;
  }
  function clickXhsSendButton(input) {
    if (!input) return false;
    let scope = input;
    for (let level = 0; level < 6 && scope; level += 1, scope = scope.parentElement) {
      const buttons = [...scope.querySelectorAll('button,[role="button"]')].filter(node => node.offsetParent !== null && !panel.contains(node));
      const send = buttons.find(node => /发送/.test((node.textContent || '').trim()) || /发送/.test(node.getAttribute('aria-label') || '') || /发送/.test(node.getAttribute('title') || ''));
      if (send) { send.click(); return true; }
    }
    // 小红书部分版本没有文字“发送”按钮，而是回车发送。
    input.dispatchEvent(new KeyboardEvent('keydown', {key:'Enter', code:'Enter', keyCode:13, which:13, bubbles:true}));
    input.dispatchEvent(new KeyboardEvent('keyup', {key:'Enter', code:'Enter', keyCode:13, which:13, bubbles:true}));
    return true;
  }
  panel.querySelector('#pet-use-selection').onclick = () => { textarea.value = window.getSelection()?.toString().trim() || ''; };
  async function handleQuestion(question) {
    if (!question) { status.textContent = '请先选中或输入顾客消息。'; return; }
    status.textContent = '正在分析顾客消息，请稍候……'; replyBox.hidden = true;
    try {
      const response = await fetch(API, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({question})});
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || '服务暂时不可用');
      const important = data.requires_confirmation;
      status.innerHTML = important ? '<span class="risk-high">重要操作：需要商家确认后发送</span>' : '<span class="risk-low">普通咨询：可以按店铺规则自动回复</span>';
      const answer = data.answer || data.error;
      replyBox.textContent = answer; replyBox.hidden = false;
      const filled = fillReplyBox(answer);
      if (filled && !important && panel.querySelector('#pet-auto-send').checked) {
        const sent = clickXhsSendButton(filled);
        status.innerHTML += sent ? '<br>普通问题已自动发送。' : '<br>已填入回复框，但未找到发送按钮，请手动发送。';
      } else {
        status.innerHTML += filled ? '<br>回复已填入小红书输入框，请检查后发送。' : '<br>未找到小红书回复框，请手动复制结果。';
      }
    } catch (error) { status.textContent = '助手暂时无法连接：' + error.message; }
  }
  panel.querySelector('#pet-generate').onclick = () => handleQuestion(textarea.value.trim());
  const candidate = text => {
    const value = text.replace(/\s+/g, ' ').trim();
    if (value.length < 4 || value.length > 300 || processedMessages.has(value)) return null;
    if (!/[猫狗犬寄养预订预定预约价格多少钱位置房间入住可以需要吗？?]/.test(value)) return null;
    return value;
  };
  const findCandidate = text => String(text).split(/\n+/).map(line => candidate(line)).filter(Boolean).pop();
  const observer = new MutationObserver(mutations => {
    if (!panel.querySelector('#pet-watch').checked) return;
    for (const mutation of mutations) {
      const nodes = mutation.type === 'characterData' ? [mutation.target.parentElement] : [...mutation.addedNodes];
      for (const node of nodes) {
        if (!node || panel.contains(node)) continue;
        const value = findCandidate(node.innerText || node.textContent || '');
        if (!value) continue;
        processedMessages.add(value);
        window.setTimeout(() => handleQuestion(value), 400);
        return;
      }
    }
  });
  const inputs = [...document.querySelectorAll('textarea, [contenteditable="true"], input[type="text"]')];
  const chatInput = inputs.find(node => /发消息|回复|输入/.test(node.getAttribute('placeholder') || node.getAttribute('aria-label') || '')) || inputs.find(node => { const rect = node.getBoundingClientRect(); return node.offsetParent !== null && !panel.contains(node) && rect.width > 500 && rect.bottom > window.innerHeight - 260; });
  let chatRoot = chatInput;
  for (let level = 0; chatRoot && level < 8; level += 1) {
    if (chatRoot.clientWidth > 500 && chatRoot.clientHeight > 300) break;
    chatRoot = chatRoot.parentElement;
  }
  if (chatRoot) observer.observe(chatRoot, {childList:true, characterData:true, subtree:true});
})();
