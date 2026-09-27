"""Local reference workspace. Document processing stays in the existing HRM service."""
from __future__ import annotations

import json
import hashlib
import re
import math
import os
import sqlite3
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from finhrm_web import db
from finhrm_web.clarification import begin_clarification, resume_clarification
from finhrm_web.calculations import (
    LABEL_VI, FinancialExtractionResult, LineItem, compute_ratios, simulate_stress_test,
)
from finhrm_web.deps import get_current_user
from finhrm_web.routers import auth, users, ho_so, approvals, conversations

ROOT = Path(__file__).resolve().parents[2]
HRM_URL = os.environ.get('FINHRM_SERVICE_URL', 'http://127.0.0.1:8766')
INDEX = Path(os.environ.get('FINHRM_PAGE_INDEX', str(ROOT / 'cache/document_page_index.sqlite3')))


def hrm(path, payload=None, *, raw=None):
    data = raw if raw is not None else json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(HRM_URL + path, data=data,
        headers={'Content-Type': 'application/octet-stream' if raw is not None else 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=300 if path == '/v1/answer' else 15) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        try:
            error = json.load(exc).get('error', 'Yêu cầu HRM thất bại')
        except (ValueError, AttributeError):
            error = 'Yêu cầu HRM thất bại'
        raise HTTPException(exc.code, error) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(503, 'Chưa kết nối được dịch vụ HRM. Kiểm tra tiến trình và nhật ký khởi động.') from exc


def document_or_404(document_id):
    doc = db.get_document(document_id)
    if not doc:
        raise HTTPException(404, 'Không tìm thấy tài liệu')
    return doc


def prepared_artifact(doc):
    with db.get_conn() as conn:
        row = conn.execute('SELECT artifact, artifact_sha256, source_sha256 FROM web_jobs WHERE document_id=?', (doc['id'],)).fetchone()
    if not row or not row['artifact']:
        raise HTTPException(409, 'Chưa có Markdown MinerU. Bấm Chạy lại OCR ở màn hình tài liệu.')
    artifact = Path(row['artifact'])
    source = Path(doc['file_path'])
    if not artifact.is_file() or not source.is_file():
        raise HTTPException(409, 'Thiếu file nguồn hoặc artifact. Hãy chạy lại OCR.')
    if hashlib.sha256(artifact.read_bytes()).hexdigest() != row['artifact_sha256'] or hashlib.sha256(source.read_bytes()).hexdigest() != row['source_sha256']:
        raise HTTPException(409, 'Nguồn hoặc Markdown đã thay đổi. Hãy chạy lại OCR.')
    return artifact


def indexed_pages(doc):
    text = prepared_artifact(doc).read_text(encoding='utf-8')
    parts = re.split(r'^## Page (\d+)\s*$', text, flags=re.MULTILINE)
    return [{'page_number': int(parts[i]), 'text': parts[i+1].strip()} for i in range(1, len(parts), 2)]


def refresh_document(doc):
    with db.get_conn() as conn:
        row = conn.execute('SELECT job_id FROM web_jobs WHERE document_id=?', (doc['id'],)).fetchone()
    if doc['status'] in {'uploaded', 'processing'} and row:
        try:
            job = hrm('/v1/jobs/' + row['job_id'])
            status = {'ready': 'indexed', 'failed': 'failed'}.get(job['status'], 'processing')
            record = next(iter(job.get('index_results', [])), {})
            stats = record.get('stats', {})
            if status == 'indexed' and not record.get('artifact'):
                status = 'failed'
                job['error'] = 'Job cũ chưa tạo Markdown MinerU. Hãy bấm Chạy lại OCR.'
            db.update_document_status(doc['id'], status, total_pages=stats.get('total_pages'), chunk_count=stats.get('text_lines'))
            with db.get_conn() as conn:
                conn.execute('UPDATE web_jobs SET error=?,artifact=?,artifact_sha256=?,source_sha256=? WHERE document_id=?', (job.get('error'),record.get('artifact'),record.get('artifact_sha256'),record.get('source_sha256'),doc['id']))
        except HTTPException as exc:
            if exc.status_code == 404:
                # Server restarted: resubmit the same file; HRM reuses its verified cache.
                job = hrm('/v1/ingest', {'document': doc['file_path']})
                with db.get_conn() as conn:
                    conn.execute('UPDATE web_jobs SET job_id=? WHERE document_id=?', (job['job_id'], doc['id']))
    updated = document_or_404(doc['id'])
    with db.get_conn() as conn:
        row = conn.execute('SELECT error FROM web_jobs WHERE document_id=?', (doc['id'],)).fetchone()
    updated['error'] = row['error'] if row else None
    return updated


class Ask(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    document_ids: list[str] = Field(min_length=1, max_length=2)
    conversation_id: str | None = Field(default=None, max_length=100)
    reset_context: bool = False


class Period(BaseModel):
    reporting_period: str | None = Field(default=None, max_length=80)


class Overrides(BaseModel):
    values: dict[str, float]


class Stress(BaseModel):
    interest_rate_shock_pct: float = Field(default=0, ge=0, allow_inf_nan=False)
    exchange_rate_shock_pct: float = Field(default=0, ge=0, allow_inf_nan=False)
    cashflow_decline_pct: float = Field(default=0, ge=0, le=100, allow_inf_nan=False)
    cycle_extension_days: float = Field(default=0, ge=0, allow_inf_nan=False)
    foreign_currency_debt: float = Field(default=0, ge=0, allow_inf_nan=False)


def financial_data(document_id):
    doc = document_or_404(document_id)
    with db.get_conn() as conn:
        row = conn.execute('SELECT payload FROM web_values WHERE document_id=?', (document_id,)).fetchone()
    values = json.loads(row['payload']) if row else {}
    fe = FinancialExtractionResult({key: LineItem(key, label, values.get(key),
        confidence='manual' if key in values else 'missing') for key, label in LABEL_VI.items()})
    ratios = [asdict(item) for item in compute_ratios(fe)]
    # Reference thresholds are not validated financial judgments; show arithmetic only.
    for ratio in ratios:
        ratio['flag'] = None
        ratio['benchmark_note'] = 'Tính từ số liệu người dùng nhập; không phải kết luận thẩm định.'
    return fe, {'document_id': document_id, 'line_items': [asdict(item) for item in fe.line_items.values()],
        'ratios': ratios, 'red_flags': [], 'assessment': None, 'disclosures': [],
        'benchmark': {'source': 'Chưa tích hợp dữ liệu chuẩn ngành.', 'metrics': []},
        'reporting_period': doc['reporting_period'], 'has_manual_overrides': bool(values), 'cache_upgrading': False}


def create_app():
    Path(db.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    db.init_db()
    with db.get_conn() as conn:
        conn.executescript('''
            CREATE TABLE IF NOT EXISTS web_jobs(document_id TEXT PRIMARY KEY, job_id TEXT, error TEXT);
            CREATE TABLE IF NOT EXISTS web_values(document_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS web_question_context(
                conversation_id TEXT PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,
                document_id TEXT NOT NULL, artifact_sha256 TEXT NOT NULL, question TEXT NOT NULL);
        ''')
    with db.get_conn() as conn:
        context_columns = {row['name'] for row in conn.execute('PRAGMA table_info(web_question_context)')}
        if 'pending_clarification' not in context_columns:
            conn.execute('ALTER TABLE web_question_context ADD COLUMN pending_clarification TEXT')
        columns = {row['name'] for row in conn.execute('PRAGMA table_info(web_jobs)')}
        for name in ('artifact', 'artifact_sha256', 'source_sha256'):
            if name not in columns:
                conn.execute(f'ALTER TABLE web_jobs ADD COLUMN {name} TEXT')
    app = FastAPI(title='FinHRM local workspace')
    app.include_router(auth.router, prefix='/api')
    for router in (users.router, ho_so.router, approvals.router, conversations.router):
        app.include_router(router, prefix='/api', dependencies=[Depends(get_current_user)])
    api = APIRouter(prefix='/api', dependencies=[Depends(get_current_user)])

    @app.get('/health')
    def health():
        try:
            return hrm('/health')
        except HTTPException:
            return {'model_loaded': False, 'status': 'unavailable'}

    @api.get('/documents')
    def documents(ho_so_id: str | None = None):
        return [refresh_document(doc) for doc in db.list_documents(ho_so_id)]

    @api.post('/documents/upload')
    async def upload(file: UploadFile, ho_so_id: str | None = None, reporting_period: str | None = None):
        if ho_so_id and not db.get_ho_so(ho_so_id):
            raise HTTPException(404, 'Không tìm thấy hồ sơ')
        filename = Path((file.filename or '').replace('\\', '/')).name
        if not filename.lower().endswith('.pdf'):
            raise HTTPException(400, 'Chỉ hỗ trợ PDF qua MinerU')
        raw = await file.read(100 * 1024 * 1024 + 1)
        if len(raw) > 100 * 1024 * 1024:
            raise HTTPException(413, 'PDF vượt giới hạn 100 MiB')
        if not raw.startswith(b'%PDF-'):
            raise HTTPException(400, 'Nội dung không phải PDF')
        uploaded = hrm('/v1/upload?filename=' + urllib.parse.quote(filename), raw=raw)
        doc = db.create_document(ho_so_id, filename, uploaded['document'], reporting_period=reporting_period)
        try:
            job = hrm('/v1/ingest', {'document': uploaded['document']})
            with db.get_conn() as conn:
                conn.execute('INSERT INTO web_jobs(document_id,job_id) VALUES (?,?)', (doc['id'], job['job_id']))
            db.update_document_status(doc['id'], 'processing')
        except HTTPException:
            db.update_document_status(doc['id'], 'failed')
            raise
        return document_or_404(doc['id'])

    @api.post('/documents/{document_id}/retry')
    def retry(document_id: str):
        doc = document_or_404(document_id)
        job = hrm('/v1/ingest', {'document': doc['file_path']})
        with db.get_conn() as conn:
            conn.execute('INSERT INTO web_jobs(document_id,job_id) VALUES (?,?) ON CONFLICT(document_id) DO UPDATE SET job_id=excluded.job_id,error=NULL,artifact=NULL,artifact_sha256=NULL,source_sha256=NULL', (document_id,job['job_id']))
        db.update_document_status(document_id, 'processing')
        return document_or_404(document_id)

    @api.get('/documents/{document_id}/text')
    def preview(document_id: str):
        return {'chunks': indexed_pages(document_or_404(document_id))}

    @api.patch('/documents/{document_id}/reporting-period')
    def period(document_id: str, payload: Period):
        document_or_404(document_id)
        return db.update_document_reporting_period(document_id, payload.reporting_period)

    @api.delete('/documents/{document_id}')
    def delete_document(document_id: str):
        doc = refresh_document(document_or_404(document_id))
        if doc['status'] == 'processing':
            raise HTTPException(409, 'MinerU đang xử lý. Chờ hoàn tất hoặc dừng job trước khi xoá khỏi danh sách.')
        if doc['status'] == 'indexed':
            raise HTTPException(409, 'Chỉ xoá tài liệu chưa hoàn tất trong luồng này')
        with db.get_conn() as conn:
            conn.execute('DELETE FROM web_jobs WHERE document_id=?', (document_id,))
            conn.execute('DELETE FROM web_values WHERE document_id=?', (document_id,))
        db.delete_document(document_id)
        return {'deleted': True}

    @api.post('/rag/ask')
    def ask(payload: Ask, current_user: dict = Depends(get_current_user)):
        outputs, sources, diagnostic_sources = [], [], []
        request_timings, response_modes = [], []
        previous = None
        if payload.conversation_id:
            if not db.get_conversation(payload.conversation_id, current_user['id']):
                raise HTTPException(404, 'Không tìm thấy hội thoại')
            with db.get_conn() as conn:
                previous = conn.execute('SELECT * FROM web_question_context WHERE conversation_id=?', (payload.conversation_id,)).fetchone()
                # Clear first, including failures and multi-document requests.
                conn.execute('DELETE FROM web_question_context WHERE conversation_id=?', (payload.conversation_id,))
        for document_id in dict.fromkeys(payload.document_ids):
            doc = refresh_document(document_or_404(document_id))
            if doc['status'] != 'indexed':
                raise HTTPException(409, f"{doc['filename']}: MinerU chưa xử lý xong")
            artifact = prepared_artifact(doc)
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            request = {'document': str(artifact), 'question': payload.question,
                'top_k_chunks': 8, 'neighbor_chunks': 1, 'page_top_k': 6, 'page_neighbor_pages': 1}
            clarification_reply = None
            if (not payload.reset_context and len(payload.document_ids) == 1 and previous is not None
                    and previous['document_id'] == doc['id'] and previous['artifact_sha256'] == digest):
                request['previous_question'] = previous['question']
                resumed = resume_clarification(payload.question, previous['pending_clarification'], digest)
                if resumed and resumed.get('question') and not resumed.get('message'):
                    request['question'] = resumed['question']
                elif resumed and resumed.get('message'):
                    clarification_reply = resumed['message']
                else:
                    from hrm_finance_qa.conversation_context import resolve_followup
                    followup = resolve_followup(payload.question, previous['question'])
                    if followup:
                        request['question'] = followup['resolved_question']
            result = ({'status':'ABSTAIN','rendered':clarification_reply,'citations':[]}
                      if clarification_reply is not None else hrm('/v1/answer', request))
            request_timings.append({'document_id':doc['id'], 'timings_seconds':result.get('timings_seconds',{})})
            response_modes.append({'document_id':doc['id'], 'status':result.get('status'),
                'answer_type':result.get('answer_type','numeric_or_text' if result.get('status')=='ACCEPTED' else 'abstention'),
                'unit_state':result.get('unit_state'), 'metric_link_state':result.get('metric_link_state'),
                'has_canonical_value':result.get('canonical_value') is not None})
            # The existing public renderer owns publication/abstention. Never expose candidates.
            rendered = result.get('rendered')
            if not isinstance(rendered, str) or not rendered.strip():
                raise HTTPException(502, 'HRM chưa trả về phản hồi công khai hợp lệ')
            if payload.conversation_id and len(payload.document_ids) == 1:
                interpreted = (previous['question'] if clarification_reply is not None
                               else result.get('question_resolution', {}).get('resolved_question', request['question']))
                pending = (previous['pending_clarification'] if clarification_reply is not None
                           else begin_clarification(result, request['question'],
                                                    request.get('previous_question'), digest))
                with db.get_conn() as conn:
                    conn.execute('INSERT OR REPLACE INTO web_question_context '
                        '(conversation_id,document_id,artifact_sha256,question,pending_clarification) VALUES (?,?,?,?,?)',
                        (payload.conversation_id, doc['id'], digest, interpreted, pending))
            outputs.append((doc['filename'] + '\n' if len(payload.document_ids) > 1 else '') + rendered)
            parts = result.get('outputs')
            parts = parts if isinstance(parts, list) else [result]
            for part in parts:
                # Current HRM responses carry an explicit publication status.
                # Keep the legacy rendered-response contract when status is absent.
                destination = diagnostic_sources if part.get('status') in {'ABSTAIN', 'REJECT'} else sources
                for citation in part.get('citations', []):
                    item = {'filename': doc['filename'], 'page_number': citation.get('page', citation.get('page_number'))}
                    if item not in destination:
                        destination.append(item)
        prefix = 'Kết quả riêng cho từng tài liệu; chưa tổng hợp phép tính chéo tài liệu.\n\n' if len(outputs) > 1 else ''
        return {'answer': prefix + '\n\n'.join(outputs), 'sources': sources,
                'diagnostic_sources': diagnostic_sources, 'request_timings': request_timings,
                'response_modes': response_modes}

    @api.get('/financials/profile/{ho_so_id}/trend')
    def trend(ho_so_id: str):
        points = []
        for doc in db.list_documents(ho_so_id):
            if not doc['reporting_period']:
                continue
            fe, data = financial_data(doc['id'])
            if not data['has_manual_overrides']:
                continue
            ratios = {item['key']: item['value'] for item in data['ratios']}
            points.append({'document_id': doc['id'], 'period': doc['reporting_period'],
                'revenue': fe.get('doanh_thu_thuan'), 'profit_after_tax': fe.get('loi_nhuan_sau_thue'),
                'operating_cashflow': fe.get('luu_chuyen_tien_thuan_tu_hoat_dong_kinh_doanh'),
                'total_assets': fe.get('tong_tai_san'), 'current_ratio': ratios.get('current_ratio'),
                'debt_to_equity': ratios.get('debt_to_equity')})
        return {'points': sorted(points, key=lambda x: x['period']), 'needs_more_periods': len(points) < 2,
            'note': 'Số liệu người dùng nhập, đơn vị triệu đồng. Cần đối chiếu cùng phạm vi và kỳ báo cáo.'}

    @api.get('/financials/{document_id}')
    def financials(document_id: str):
        return financial_data(document_id)[1]

    @api.patch('/financials/{document_id}/override')
    def override(document_id: str, payload: Overrides):
        document_or_404(document_id)
        if any(key not in LABEL_VI or not math.isfinite(value) for key, value in payload.values.items()):
            raise HTTPException(422, 'Chỉ tiêu hoặc giá trị không hợp lệ')
        with db.get_conn() as conn:
            row = conn.execute('SELECT payload FROM web_values WHERE document_id=?', (document_id,)).fetchone()
            values = json.loads(row['payload']) if row else {}
            values.update(payload.values)
            conn.execute('INSERT OR REPLACE INTO web_values VALUES (?,?)', (document_id, json.dumps(values)))
        return financial_data(document_id)[1]

    @api.post('/financials/{document_id}/stress-test')
    def stress(document_id: str, payload: Stress):
        fe = financial_data(document_id)[0]
        required = ['luu_chuyen_tien_thuan_tu_hoat_dong_kinh_doanh', 'loi_nhuan_truoc_thue', 'chi_phi_lai_vay']
        missing = [LABEL_VI[key] for key in required if fe.get(key) is None]
        if fe.get('vay_ngan_han') is None and fe.get('no_ngan_han') is None:
            missing.append('Nợ ngắn hạn hoặc vay ngắn hạn')
        if payload.cycle_extension_days and fe.get('phai_thu_khach_hang') is None and fe.get('phai_thu_ngan_han') is None:
            missing.append('Phải thu khách hàng hoặc phải thu ngắn hạn')
        if missing:
            return {'risk_level': 'Chưa đủ dữ liệu', 'missing_inputs': missing, 'conclusions': [],
                    'disclaimer': 'Chưa tính kịch bản vì thiếu đầu vào; không tự thay bằng 0.'}
        result = simulate_stress_test(fe, **payload.model_dump())
        result['risk_level'] = 'Kịch bản tính toán' if not result['missing_inputs'] else 'Chưa đủ dữ liệu'
        result['disclaimer'] = 'Tính từ số liệu nhập tay và giả định trên form; không phải kết luận của HRM hoặc phê duyệt tín dụng.'
        return result

    app.include_router(api)
    dist = ROOT / 'web_demo/dist'
    if dist.is_dir():
        app.mount('/', StaticFiles(directory=dist, html=True), name='web')
    return app
