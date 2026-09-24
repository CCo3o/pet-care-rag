"""网页演示入口：python -m app.web，然后打开 http://127.0.0.1:8000。"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
from pathlib import Path

from openai import APIConnectionError, APIStatusError, APITimeoutError
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.chat import ChatAssistant
from app.display import terminal_text
from app.config import CHROMA_DIR, DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, TOP_K
from app.lightweight_retrieval import LightweightRetriever
from app.retrieval import PROMPT_TEMPLATE, create_llm, create_retriever


LOGGER = logging.getLogger("pet-care.web")
RATE_LIMIT_PER_MINUTE = max(1, int(os.getenv("RATE_LIMIT_PER_MINUTE", "30")))
RATE_WINDOW_SECONDS = 60.0
RATE_STATE: dict[str, deque[float]] = defaultdict(deque)
RATE_LOCK = Lock()
METRICS = {"requests": 0, "successes": 0, "errors": 0, "rate_limited": 0, "latency_ms_total": 0.0}
METRICS_LOCK = Lock()


def allow_request(client_key: str) -> bool:
    """Small in-memory guard for the demo service; production should use Redis/API gateway."""
    now = time.monotonic()
    with RATE_LOCK:
        bucket = RATE_STATE[client_key]
        while bucket and now - bucket[0] >= RATE_WINDOW_SECONDS:
            bucket.popleft()
        if len(bucket) >= RATE_LIMIT_PER_MINUTE:
            return False
        bucket.append(now)
        return True


def record_metric(status: int, elapsed_ms: float) -> None:
    with METRICS_LOCK:
        METRICS["requests"] += 1
        METRICS["latency_ms_total"] += elapsed_ms
        if status == 429:
            METRICS["rate_limited"] += 1
        elif status >= 400:
            METRICS["errors"] += 1
        else:
            METRICS["successes"] += 1


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
:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:linear-gradient(135deg,#eef4ff,#f8fafc);font:16px system-ui,'Microsoft YaHei',sans-serif;color:#263238}.app{width:min(1120px,100%);height:100vh;margin:auto;background:#fff;display:flex;flex-direction:column;box-shadow:0 0 32px #d8e1ef}header{padding:26px 42px;border-bottom:1px solid #e5eaf1}h1{margin:0;font-size:26px}header p{color:#64748b;font-size:15px;margin:10px 0 0}#messages{flex:1;overflow:auto;padding:32px 7%;scroll-behavior:smooth}.msg{max-width:82%;padding:15px 19px;border-radius:16px;margin:0 0 18px;white-space:pre-wrap;line-height:1.8;word-break:break-word}.user{margin-left:auto;background:#2563eb;color:#fff}.assistant{background:#eef2f7}.sources{font-size:13px;color:#64748b;margin-top:10px;padding-top:8px;border-top:1px solid #d9e1ec}form{display:flex;gap:12px;padding:18px 42px;border-top:1px solid #e5eaf1;background:#fff}input{flex:1;min-width:0;padding:14px 16px;border:1px solid #cbd5e1;border-radius:12px;font-size:16px;outline:none}input:focus{border-color:#2563eb;box-shadow:0 0 0 3px #dbeafe}button{border:0;border-radius:12px;padding:0 24px;background:#2563eb;color:#fff;font-size:15px;cursor:pointer}button:disabled{opacity:.6;cursor:wait}button.secondary{background:#eef2f7;color:#475569;padding:9px 13px;margin-left:15px}.welcome{text-align:center;color:#64748b;margin:90px auto 0;line-height:2}@media(max-width:700px){header{padding:20px}h1{font-size:22px}header p{line-height:1.8}#messages{padding:22px 16px}.msg{max-width:94%}form{padding:12px 16px}button{padding:0 17px}}
</style>
<main class='app'><header><h1>🐱🐶 宠物寄养助手</h1><p>咨询资料、计算费用、查询档期和管理本地演示预订 <button class='secondary' id='clear'>清空当前对话</button></p></header><section id='messages'><div class='welcome'>试试：猫可以吃巧克力吗？<br>或：猫标准间，2026年12月1日至4日有位置吗？</div></section><form id='form'><input id='question' placeholder='请输入你的问题，例如：我家狗寄养7天多少钱？'><button id='send'>发送</button></form></main>
<script>const box=document.querySelector('#messages'),input=document.querySelector('#question'),form=document.querySelector('#form'),send=document.querySelector('#send');function add(t,w,s=[]){document.querySelector('.welcome')?.remove();let e=document.createElement('div');e.className='msg '+w;e.textContent=t;if(s.length){let x=document.createElement('div');x.className='sources';x.textContent='参考资料：'+s.join('、');e.append(x)}box.append(e);box.scrollTop=box.scrollHeight}form.onsubmit=async e=>{e.preventDefault();let q=input.value.trim();if(!q)return;add(q,'user');input.value='';send.disabled=true;try{let r=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});let d=await r.json();add(d.answer||d.error,'assistant',d.sources||[])}catch(x){add('网页暂时无法连接服务，请检查终端是否仍在运行。','assistant')}finally{send.disabled=false;input.focus()}};document.querySelector('#clear').onclick=async()=>{await fetch('/api/clear',{method:'POST'});box.innerHTML='<div class="welcome">当前对话已清空，已保存的预订仍然保留。</div>'}</script></html>"""

ASSISTANT = None


class Handler(BaseHTTPRequestHandler):
    ALLOWED_ORIGINS = {'https://www.xiaohongshu.com', 'http://127.0.0.1:8000', 'http://localhost:8000'}

    def send_body(self, body, content_type='text/html; charset=utf-8', status=200):
        data = body.encode('utf-8') if isinstance(body, str) else body
        self.send_response(status); self.send_header('Content-Type', content_type)
        origin = self.headers.get('Origin')
        if origin in self.ALLOWED_ORIGINS: self.send_header('Access-Control-Allow-Origin', origin); self.send_header('Vary', 'Origin')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type'); self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)

    def send_json(self, payload, status=200):
        self.send_body(json.dumps(payload, ensure_ascii=False), 'application/json; charset=utf-8', status)

    def client_key(self):
        forwarded = self.headers.get('X-Forwarded-For', '').split(',')[0].strip()
        return forwarded or (self.client_address[0] if self.client_address else 'unknown')

    def do_OPTIONS(self):
        self.send_response(204); origin = self.headers.get('Origin')
        if origin in self.ALLOWED_ORIGINS: self.send_header('Access-Control-Allow-Origin', origin); self.send_header('Vary', 'Origin')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS'); self.send_header('Access-Control-Allow-Headers', 'Content-Type'); self.end_headers()

    def do_GET(self):
        if self.path == '/health':
            self.send_json({'status': 'ok'})
        elif self.path == '/metrics':
            with METRICS_LOCK:
                metrics = dict(METRICS)
            metrics['latency_ms_avg'] = round(metrics['latency_ms_total'] / metrics['requests'], 2) if metrics['requests'] else 0.0
            metrics.pop('latency_ms_total', None)
            self.send_json(metrics)
        else:
            self.send_body(HTML if self.path == '/' else 'Not Found', status=200 if self.path == '/' else 404)

    def do_POST(self):
        global ASSISTANT
        started = time.perf_counter()
        if self.path == '/api/chat' and not allow_request(self.client_key()):
            record_metric(429, (time.perf_counter() - started) * 1000)
            self.send_json({'error': '请求过于频繁，请稍后再试。'}, 429)
            return
        length = int(self.headers.get('Content-Length', 0))
        try:
            payload = json.loads(self.rfile.read(length) or b'{}')
        except (TypeError, ValueError):
            record_metric(400, (time.perf_counter() - started) * 1000)
            self.send_json({'error': '请求格式无效'}, 400)
            return
        if self.path == '/api/clear':
            ASSISTANT.clear(); record_metric(200, (time.perf_counter() - started) * 1000); self.send_json({'ok': True}); return
        if self.path != '/api/chat' or not str(payload.get('question', '')).strip():
            record_metric(400, (time.perf_counter() - started) * 1000)
            self.send_json({'error': '请输入问题'}, 400); return
        try:
            reply = ASSISTANT.reply(str(payload['question']).strip())
            question = str(payload['question']).strip()
            high_risk_terms = ('预订', '预定', '预约', '取消', '退款', '付款', '转账', '生病', '受伤', '用药', '过敏')
            risk = 'high' if any(term in question for term in high_risk_terms) else 'low'
            elapsed = (time.perf_counter() - started) * 1000
            record_metric(200, elapsed)
            LOGGER.info('chat completed status=200 latency_ms=%.0f', elapsed)
            self.send_json({'answer': terminal_text(reply.answer), 'sources': [Path(s).stem for s in reply.sources], 'risk': risk, 'requires_confirmation': risk == 'high'})
        except (APITimeoutError, APIConnectionError):
            elapsed = (time.perf_counter() - started) * 1000
            record_metric(503, elapsed); LOGGER.warning('chat failed status=503 latency_ms=%.0f', elapsed)
            self.send_json({'error': '模型服务暂时无法连接，请稍后重试。'}, 503)
        except APIStatusError as exc:
            elapsed = (time.perf_counter() - started) * 1000
            record_metric(502, elapsed); LOGGER.warning('chat failed status=502 latency_ms=%.0f', elapsed)
            self.send_json({'error': f'模型服务返回 HTTP {exc.status_code}。'}, 502)
        except sqlite3.Error:
            elapsed = (time.perf_counter() - started) * 1000
            record_metric(503, elapsed); LOGGER.warning('chat failed status=503 latency_ms=%.0f', elapsed)
            self.send_json({'error': '预订数据暂时无法保存，请稍后重试。'}, 503)

    def log_message(self, *_):
        return


def main():
    global ASSISTANT
    logging.basicConfig(
        level=os.getenv('LOG_LEVEL', 'INFO').upper(),
        format='%(asctime)s %(levelname)s %(name)s %(message)s',
    )
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
