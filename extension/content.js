(() => {
  if (document.getElementById('pet-care-assistant')) return;
  const API = 'https://pet-care-rag-demo.onrender.com/api/chat';
  // Keep a visible build marker so it is easy to verify that Edge reloaded the
  // current unpacked extension instead of an older copy.
  const BUILD = 'watch-map-20260922';
  const panel = document.createElement('aside');
  panel.id = 'pet-care-assistant';
  panel.innerHTML = `<header><span>🐱🐶 宠物寄养智慧客服 <small class="pet-build">${BUILD}</small></span><button class="pet-toggle" title="收起">−</button></header><main><textarea placeholder="先选中顾客消息，或直接粘贴到这里"></textarea><label class="auto-send"><input id="pet-auto-send" type="checkbox" checked> 普通问题自动发送</label><label class="auto-send"><input id="pet-watch" type="checkbox"> 自动监听新消息</label><div><button id="pet-generate">生成回复</button><button class="secondary" id="pet-use-selection">读取选中文本</button></div><div class="status">普通咨询会自动回复；预订、取消、付款和健康问题需要人工确认。</div><div class="reply" hidden></div></main>`;
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
  // Message nodes are sometimes reused by Xiaohongshu's Vue renderer.  A
  // WeakSet therefore misses a new message when the same node's text changes;
  // retain the last text per node instead and only queue an actual transition.
  const nodeSnapshots = new WeakMap();
  const handledNodeTexts = new WeakMap();
  let sequenceSnapshot = null;
  let lastIncomingCount = null;
  const watchBox = panel.querySelector('#pet-watch');
  watchBox.onchange = () => {
    if (!watchBox.checked) {
      status.textContent = `新消息监听已关闭（${BUILD}）。`;
      return;
    }
    // Establish a fresh baseline at the moment the merchant opts in.  This
    // prevents messages that arrived while the checkbox was off from being
    // replayed as if they were new.
    scanIncoming(false);
    const count = incomingNodes().length;
    status.textContent = `已开启新消息监听（${BUILD}），已识别 ${count} 条顾客消息，等待新消息……`;
  };
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
  async function handleQuestion(question, context = {}) {
    if (!question) { status.textContent = '请先选中或输入顾客消息。'; return; }
    const prefix = context.auto ? `检测到新消息：“${question.slice(0, 32)}${question.length > 32 ? '…' : ''}”` : '';
    status.textContent = prefix ? `${prefix}\n正在请求助手，请稍候……` : '正在分析顾客消息，请稍候……';
    replyBox.hidden = true;
    try {
      const response = await fetch(API, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({question})});
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || '服务暂时不可用');
      const important = data.requires_confirmation;
      status.innerHTML = (prefix ? `${prefix}<br>` : '') + (important ? '<span class="risk-high">重要操作：需要商家确认后发送</span>' : '<span class="risk-low">普通咨询：可以按店铺规则自动回复</span>');
      const answer = data.answer || data.error;
      replyBox.textContent = answer; replyBox.hidden = false;
      const filled = fillReplyBox(answer);
      if (filled && !important && panel.querySelector('#pet-auto-send').checked) {
        const sent = clickXhsSendButton(filled);
        status.innerHTML += sent ? '<br>普通问题已自动发送。' : '<br>已填入回复框，但未找到发送按钮，请手动发送。';
      } else {
        status.innerHTML += filled ? '<br>回复已填入小红书输入框，请检查后发送。' : '<br>未找到小红书回复框，请手动复制结果。';
      }
    } catch (error) {
      status.textContent = `${prefix ? `${prefix}\n` : ''}助手暂时无法连接：${error.message}`;
    }
  }
  panel.querySelector('#pet-generate').onclick = () => handleQuestion(textarea.value.trim());
  const candidate = text => {
    const value = text.replace(/\s+/g, ' ').trim();
    if (value.length < 4 || value.length > 300) return null;
    if (!/[猫狗犬寄养预订预定预约价格多少钱位置房间入住可以需要吗？?]/.test(value)) return null;
    return value;
  };
  const findCandidate = text => String(text).split(/\n+/).map(line => candidate(line)).filter(Boolean).pop();
  const inputs = [...document.querySelectorAll('textarea, [contenteditable="true"], input[type="text"]')];
  const chatInput = inputs.find(node => /发消息|回复|输入/.test(node.getAttribute('placeholder') || node.getAttribute('aria-label') || '')) || inputs.find(node => { const rect = node.getBoundingClientRect(); return node.offsetParent !== null && !panel.contains(node) && rect.width > 500 && rect.bottom > window.innerHeight - 260; });
  let chatRoot = chatInput;
  for (let level = 0; chatRoot && level < 8; level += 1) {
    if (chatRoot.clientWidth > 500 && chatRoot.clientHeight > 300) break;
    chatRoot = chatRoot.parentElement;
  }
  const incomingNodes = () => {
    // Prefer the exact paragraph class from the current Xiaohongshu DOM.  A
    // broad fallback is only used on older layouts; otherwise the wrapper and
    // its paragraph would both be returned and trigger duplicate replies.
    const exact = [...document.querySelectorAll('.chat-item__body-left .xhs-im-bubble__text')];
    if (exact.length) return exact;
    const fallback = [...document.querySelectorAll('.chat-item__body-left [class*="bubble__text"], .chat-item__body-left [class*="bubble-text"]')];
    return fallback.filter(node => !fallback.some(parent => parent !== node && parent.contains(node)));
  };
  const normalizeNodeText = node => String(node?.textContent || '').replace(/\s+/g, ' ').trim();
  const handledTextsFor = node => {
    let texts = handledNodeTexts.get(node);
    if (!texts) { texts = new Set(); handledNodeTexts.set(node, texts); }
    return texts;
  };
  const queueIncomingNode = (node, text) => {
    const value = findCandidate(text);
    if (!value) return;
    const handled = handledTextsFor(node);
    if (handled.has(value)) return;
    handled.add(value);
    // MutationObserver and the polling fallback can see the same transition;
    // recording this node/text before scheduling prevents duplicate API calls
    // while a repeated question in a different node still works.
    status.textContent = `检测到新消息：“${value.slice(0, 32)}${value.length > 32 ? '…' : ''}”\n正在请求助手，请稍候……`;
    window.setTimeout(() => {
      handleQuestion(value, {auto: true});
    }, 250);
  };
  const scanIncoming = (allowTrigger = watchBox.checked) => {
    const nodes = incomingNodes();
    const texts = nodes.map(normalizeNodeText);
    const previousSequence = sequenceSnapshot;
    sequenceSnapshot = texts;
    if (allowTrigger && lastIncomingCount !== null && nodes.length !== lastIncomingCount) {
      status.textContent = `检测到消息列表变化（当前 ${nodes.length} 条），正在识别……`;
    }
    lastIncomingCount = nodes.length;

    // If Vue rebuilt every DOM node, node-based detection alone would treat
    // the entire history as new.  Only accept an appended tail or a change in
    // the final message when the ordered text snapshot confirms it.
    let allowedIndexes = null;
    if (allowTrigger && previousSequence) {
      const samePrefix = previousSequence.length <= texts.length && previousSequence.every((value, index) => value === texts[index]);
      if (samePrefix) {
        allowedIndexes = new Set();
        for (let index = previousSequence.length; index < texts.length; index += 1) allowedIndexes.add(index);
      } else if (previousSequence.length === texts.length) {
        const firstDiff = texts.findIndex((value, index) => value !== previousSequence[index]);
        // A reused node normally changes only the newest (tail) message.  Do
        // not answer old messages after an unrelated list reorder.
        if (firstDiff >= Math.max(0, texts.length - 2)) allowedIndexes = new Set([firstDiff]);
        else allowedIndexes = new Set();
      } else {
        allowedIndexes = new Set();
      }
    }
    for (const node of nodes) {
      if (panel.contains(node)) continue;
      const index = nodes.indexOf(node);
      const text = normalizeNodeText(node);
      if (!text) continue;
      const previous = nodeSnapshots.get(node);
      nodeSnapshots.set(node, text);
      if (!allowTrigger) continue;
      // A missing snapshot means a genuinely added message node.  A changed
      // snapshot means Xiaohongshu reused the node for a newly received text.
      if ((previous === undefined || previous !== text) && (!allowedIndexes || allowedIndexes.has(index))) queueIncomingNode(node, text);
    }
    return nodes.length;
  };
  // Baseline all messages already rendered so enabling the watcher does not
  // answer the conversation history.  The scan below keeps snapshots current
  // even while the checkbox is off.
  scanIncoming(false);
  const observer = new MutationObserver(() => {
    if (watchBox.checked) scanIncoming(true);
    else scanIncoming(false);
  });
  if (chatRoot) observer.observe(chatRoot, {childList:true, characterData:true, subtree:true});
  window.setInterval(() => {
    // Polling is intentional: Xiaohongshu can update text in a virtualized
    // list without emitting a useful mutation on the observed composer root.
    scanIncoming(watchBox.checked);
  }, 1500);
})();
