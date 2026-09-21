"""网页演示入口：python -m app.web，然后打开 http://127.0.0.1:8000。"""
from __future__ import annotations

import json
import os
import sqlite3
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from openai import APIConnectionError, APIStatusError, APITimeoutError
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.chat import ChatAssistant
from app.config import CHROMA_DIR, DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, TOP_K
from app.lightweight_retrieval import LightweightRetriever
from app.retrieval import PROMPT_TEMPLATE, create_llm, create_retriever


def build_assistant():
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY，请先检查 .env")
    llm = create_llm(DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL)
    if os.getenv('RETRIEVAL_BACKEND', 'vector').lower() == 'lightweight':
        retriever = LightweightRetriever(DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K)
    else:
        from app.vector_store import get_embeddings, get_vector_store, build_index
        embeddings = get_embeddings(EMBEDDING_MODEL)
        store = get_vector_store(embeddings, CHROMA_DIR)
        if store._collection.count() == 0:
            from app.ingestion import load_documents, split_documents
            store = build_index(split_documents(load_documents(DOCS_DIR), CHUNK_SIZE, CHUNK_OVERLAP), embeddings, CHROMA_DIR)
        retriever = create_retriever(store, k=TOP_K)
    chain = ChatPromptTemplate.from_template(PROMPT_TEMPLATE) | llm | StrOutputParser()
    return ChatAssistant(llm, retriever, chain)


HTML = """<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>宠物寄养助手</title><style>
body{margin:0;background:#f3f6fb;font:16px system-ui,'Microsoft YaHei',sans-serif;color:#263238}.app{max-width:900px;height:100vh;margin:auto;background:#fff;display:flex;flex-direction:column;box-shadow:0 0 25px #dce3ed}header{padding:22px 28px;border-bottom:1px solid #e5eaf1}h1{margin:0;font-size:22px}header p{color:#64748b;font-size:14px}#messages{flex:1;overflow:auto;padding:24px}.msg{max-width:78%;padding:13px 16px;border-radius:14px;margin:0 0 16px;white-space:pre-wrap;line-height:1.65}.user{margin-left:auto;background:#2563eb;color:#fff}.assistant{background:#eef2f7}.sources{font-size:13px;color:#64748b;margin-top:8px}form{display:flex;gap:10px;padding:16px 20px;border-top:1px solid #e5eaf1}input{flex:1;padding:13px;border:1px solid #cbd5e1;border-radius:10px;font-size:16px}button{border:0;border-radius:10px;padding:0 20px;background:#2563eb;color:#fff;font-size:15px}button.secondary{background:#eef2f7;color:#475569;padding:8px 12px;margin-left:15px}.welcome{text-align:center;color:#64748b;margin-top:70px}</style>
<main class='app'><header><h1>🐱🐶 宠物寄养助手</h1><p>咨询资料、计算费用、查询档期和管理本地演示预订 <button class='secondary' id='clear'>清空当前对话</button></p></header><section id='messages'><div class='welcome'>试试：猫可以吃巧克力吗？<br>或：猫标准间，2026年12月1日至4日有位置吗？</div></section><form id='form'><input id='question' placeholder='请输入你的问题，例如：我家狗寄养7天多少钱？'><button id='send'>发送</button></form></main>
<script>const box=document.querySelector('#messages'),input=document.querySelector('#question'),form=document.querySelector('#form'),send=document.querySelector('#send');function add(t,w,s=[]){document.querySelector('.welcome')?.remove();let e=document.createElement('div');e.className='msg '+w;e.textContent=t;if(s.length){let x=document.createElement('div');x.className='sources';x.textContent='参考资料：'+s.join('、');e.append(x)}box.append(e);box.scrollTop=box.scrollHeight}form.onsubmit=async e=>{e.preventDefault();let q=input.value.trim();if(!q)return;add(q,'user');input.value='';send.disabled=true;try{let r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});let d=await r.json();add(d.answer||d.error,'assistant',d.sources||[])}catch(x){add('网页暂时无法连接服务，请检查终端是否仍在运行。','assistant')}finally{send.disabled=false;input.focus()}};document.querySelector('#clear').onclick=async()=>{await fetch('/api/clear',{method:'POST'});box.innerHTML='<div class="welcome">当前对话已清空，已保存的预订仍然保留。</div>'}</script></html>"""

ASSISTANT = None


class Handler(BaseHTTPRequestHandler):
    def send_body(self, body, content_type='text/html; charset=utf-8', status=200):
        data = body.encode('utf-8') if isinstance(body, str) else body
        self.send_response(status); self.send_header('Content-Type', content_type); self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)

    def do_GET(self):
        if self.path == '/health':
            self.send_body(json.dumps({'status': 'ok'}), 'application/json; charset=utf-8')
        else:
            self.send_body(HTML if self.path == '/' else 'Not Found', status=200 if self.path == '/' else 404)

    def do_POST(self):
        global ASSISTANT
        length = int(self.headers.get('Content-Length', 0)); payload = json.loads(self.rfile.read(length) or b'{}')
        if self.path == '/api/clear':
            ASSISTANT.clear(); self.send_body(json.dumps({'ok': True}), 'application/json; charset=utf-8'); return
        if self.path != '/api/chat' or not str(payload.get('question', '')).strip():
            self.send_body(json.dumps({'error': '请输入问题'}, ensure_ascii=False), 'application/json; charset=utf-8', 400); return
        try:
            reply = ASSISTANT.reply(str(payload['question']).strip())
            self.send_body(json.dumps({'answer': reply.answer, 'sources': [Path(s).stem for s in reply.sources]}, ensure_ascii=False), 'application/json; charset=utf-8')
        except (APITimeoutError, APIConnectionError):
            self.send_body(json.dumps({'error': '模型服务暂时无法连接，请稍后重试。'}, ensure_ascii=False), 'application/json; charset=utf-8', 503)
        except APIStatusError as exc:
            self.send_body(json.dumps({'error': f'模型服务返回 HTTP {exc.status_code}。'}, ensure_ascii=False), 'application/json; charset=utf-8', 502)
        except sqlite3.Error:
            self.send_body(json.dumps({'error': '预订数据暂时无法保存，请稍后重试。'}, ensure_ascii=False), 'application/json; charset=utf-8', 503)

    def log_message(self, *_):
        return


def main():
    global ASSISTANT
    print('正在加载本地知识库和模型，请稍候...')
    ASSISTANT = build_assistant()
    host = os.getenv('HOST', '127.0.0.1')
    port = int(os.getenv('PORT', '8000'))
    server = ThreadingHTTPServer((host, port), Handler)
    print(f'网页已启动：http://{host}:{port}')
    print('按 Ctrl+C 停止服务。')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n网页服务已停止。')
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
