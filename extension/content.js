(() => {
  if (document.getElementById('pet-care-assistant')) return;
  const API = 'https://pet-care-rag-demo.onrender.com/api/chat';
  const panel = document.createElement('aside');
  panel.id = 'pet-care-assistant';
  panel.innerHTML = `<header>🐱🐶 宠物寄养智慧客服</header><main><textarea placeholder="先选中顾客消息，或直接粘贴到这里"></textarea><div><button id="pet-generate">生成回复</button><button class="secondary" id="pet-use-selection">读取选中文本</button></div><div class="status">普通咨询会标记为可自动回复；预订、取消、付款和健康问题需要人工确认。</div><div class="reply" hidden></div></main>`;
  document.body.appendChild(panel);
  const textarea = panel.querySelector('textarea');
  const status = panel.querySelector('.status');
  const replyBox = panel.querySelector('.reply');
  panel.querySelector('#pet-use-selection').onclick = () => { textarea.value = window.getSelection()?.toString().trim() || ''; };
  panel.querySelector('#pet-generate').onclick = async () => {
    const question = textarea.value.trim();
    if (!question) { status.textContent = '请先选中或输入顾客消息。'; return; }
    status.textContent = '正在分析顾客消息，请稍候……'; replyBox.hidden = true;
    try {
      const response = await fetch(API, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({question})});
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || '服务暂时不可用');
      const important = data.requires_confirmation;
      status.innerHTML = important ? '<span class="risk-high">重要操作：需要商家确认后发送</span>' : '<span class="risk-low">普通咨询：可以按店铺规则自动回复</span>';
      replyBox.textContent = data.answer || data.error; replyBox.hidden = false;
    } catch (error) { status.textContent = '助手暂时无法连接：' + error.message; }
  };
})();
