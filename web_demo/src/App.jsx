import ComparisonAnswer from './ComparisonAnswer.jsx';
import { answerDisplay } from './answerDisplay.js';
import { useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertTriangle, BarChart3, Bot, Check, CheckSquare2, ChevronLeft, Download,
  Eye, EyeOff, FileText, Folder, Home, LineChart, LockKeyhole, LogOut,
  MessageSquareText, Plus, Search, Send, Settings, ShieldCheck, Sparkles,
  UploadCloud, UserRound, UserPlus, X, RefreshCw, Trash2, Wifi, WifiOff
} from 'lucide-react';

const API_BASE = '/api';

const C = {
  wine: '#0d0d0d', wineDark: '#0d0d0d', wineLight: '#5d5d5d', gold: '#5d5d5d',
  ink: '#0d0d0d', muted: '#5d5d5d', border: '#e6e6e6', rose: '#f9f9f9', panel: '#f9f9f9'
};

const navItems = [
  { id: 'overview', label: 'Tổng quan', icon: Home },
  { id: 'profiles', label: 'Hồ sơ phân tích', icon: Folder },
  { id: 'ocr', label: 'Tài liệu & OCR', icon: FileText },
  { id: 'analysis', label: 'Phân tích', icon: BarChart3 },
  { id: 'reports', label: 'Thống kê & báo cáo', icon: LineChart },
  { id: 'assistant', label: 'Trợ lý phân tích', icon: MessageSquareText },
  { id: 'approval', label: 'Phê duyệt', icon: CheckSquare2 },
];

const pageCopy = {
  overview: ['Tổng quan', 'Dashboard doanh nghiệp và hồ sơ'],
  profiles: ['Hồ sơ phân tích', 'Quản lý hồ sơ doanh nghiệp đang theo dõi'],
  ocr: ['Tài liệu & OCR', 'Số hoá và bóc tách dữ liệu báo cáo tài chính'],
  analysis: ['Phân tích', 'Phân tích chỉ số tài chính chuyên sâu'],
  reports: ['Thống kê & báo cáo', 'Tổng hợp và xuất báo cáo định kỳ'],
  approval: ['Phê duyệt', 'Luồng phê duyệt kết quả phân tích'],
  admin: ['Quản lý', 'Người dùng, phân quyền và cấu hình hệ thống'],
};

const initialSettings = {
  endpoint: '/health', model: 'HRM V14.1 · Policy B',
  systemPrompt: 'MinerU → HRM → symbolic verification → câu trả lời có bằng chứng.'
};

function BrandMark({ small = false }) {
  return (
    <div className={`brand-mark ${small ? 'brand-mark--small' : ''}`} aria-label="FinHRM">
      <BarChart3 size={30} aria-hidden="true" />
    </div>
  );
}

function Brand({ inverse = false, compact = false }) {
  return <div className={`brand ${inverse ? 'brand--inverse' : ''}`}>
    <BrandMark small={compact} />
    <div className="brand-copy">
      <strong>FinHRM</strong>
      <span>{compact ? 'PHÂN TÍCH TÀI CHÍNH' : 'ĐỌC HIỂU BÁO CÁO TÀI CHÍNH'}</span>
    </div>
  </div>;
}

function ScoreCard() {
  const bars = [38, 54, 45, 70, 60, 86, 70];
  return <div className="score-card">
    <div className="score-card__head"><span>TỔNG QUAN PHÂN TÍCH</span><span className="score-live"><i /> MINH HOẠ GIAO DIỆN</span></div>
    <div className="score-card__body">
      <div className="score-ring"><span><BarChart3 size={36} /></span><small>phân tích</small></div>
      <div className="bars">{bars.map((h, i) => <i key={i} style={{ height: `${h}%` }} className={i > 4 ? 'hot' : ''} />)}</div>
    </div>
  </div>;
}

function Login({ onLogin }) {
  const [user, setUser] = useState('');
  const [pw, setPw] = useState('');
  const [show, setShow] = useState(false);
  const [err, setErr] = useState('');
  const submit = async e => {
    e.preventDefault();
    try {
      const response = await fetch(`${API_BASE}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: user, password: pw }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Đăng nhập thất bại.');
      setErr('');
      onLogin(data);
    } catch (error) {
      setErr(error.message || 'Không kết nối được tới backend.');
    }
  };
  return <div className="login-page">
    <div className="login-topline" />
    <section className="login-hero">
      <div className="hero-decor hero-decor--a" /><div className="hero-decor hero-decor--b" />
      <div>
        <div className="hero-brand-row">
          <Brand inverse />
          <div className="hero-divider" />
          <div className="hero-platform">NỀN TẢNG PHÂN<br />TÍCH TÀI CHÍNH</div>
        </div>
        <div className="hero-kicker">— &nbsp; GIẢI PHÁP DỮ LIỆU THÔNG MINH</div>
        <h1>Hiểu sâu dữ liệu.<br /><em>Vững vàng quyết định.</em></h1>
        <p>Không gian làm việc tập trung dành cho đội ngũ phân tích, giúp biến dữ liệu tài chính thành những quyết định tin cậy.</p>
      </div>
      <div className="hero-bottom">
        <ScoreCard />
        <div className="hero-values"><span>ĐỒNG HÀNH CÙNG PHÁT TRIỂN</span><i /><span>BẢO MẬT &amp; TIN CẬY</span></div>
      </div>
    </section>
    <section className="login-form-panel">
      <div className="login-card">
        <BrandMark />
        <div className="login-kicker">CỔNG THÔNG TIN NỘI BỘ</div>
        <h2>Chào mừng trở lại</h2>
        <p>Đăng nhập để tiếp tục vào hệ thống phân tích.</p>
        <form onSubmit={submit}>
          <label>Tên đăng nhập</label>
          <div className="field"><UserRound size={18} /><input value={user} onChange={e => setUser(e.target.value)} placeholder="Nhập tên đăng nhập" autoComplete="username" /></div>
          <label>Mật khẩu</label>
          <div className="field"><LockKeyhole size={18} /><input value={pw} onChange={e => setPw(e.target.value)} type={show ? 'text' : 'password'} placeholder="Nhập mật khẩu" autoComplete="current-password" /><button type="button" onClick={() => setShow(v => !v)}>{show ? <EyeOff size={18} /> : <Eye size={18} />}</button></div>
          {err && <div className="form-error">{err}</div>}
          <button className="login-button">Đăng nhập <span>→</span></button>
        </form>
        <div className="secure-note"><ShieldCheck size={18} /><div><strong>Truy cập dành cho cán bộ được cấp quyền</strong><span>Vui lòng sử dụng tài khoản nội bộ để đăng nhập.</span></div></div>
      </div>
    </section>
  </div>;
}

const STATUS_META = {
  moi_tao: { label: 'Mới tạo', color: '#5d5d5d', bg: '#f9f9f9' },
  cho_kiem_soat: { label: 'Chờ kiểm soát', color: '#0d0d0d', bg: '#e6e6e6' },
  cho_phe_duyet: { label: 'Chờ phê duyệt', color: '#0d0d0d', bg: '#e6e6e6' },
  da_duyet: { label: 'Đã duyệt', color: '#0d0d0d', bg: '#f9f9f9' },
  tu_choi: { label: 'Từ chối', color: '#5d5d5d', bg: '#f9f9f9' },
};

const DOC_STATUS_META = {
  uploaded: { label: 'Đã tải lên', color: '#5d5d5d', bg: '#f9f9f9' },
  processing: { label: 'Đang xử lý', color: '#0d0d0d', bg: '#e6e6e6' },
  indexed: { label: 'Đã lập chỉ mục', color: '#0d0d0d', bg: '#f9f9f9' },
  failed: { label: 'Lỗi xử lý', color: '#5d5d5d', bg: '#f9f9f9' },
};

const ROLE_LABELS = {
  can_bo: 'Cán bộ phân tích',
  kiem_soat: 'Kiểm soát viên',
  truong_phong: 'Trưởng phòng',
  quan_tri: 'Quản trị hệ thống',
};

const ASSESSMENT_META = {
  good: { label: 'Tích cực', color: '#0d0d0d' },
  warning: { label: 'Cần theo dõi', color: '#0d0d0d' },
  danger: { label: 'Rủi ro cao', color: '#5d5d5d' },
  missing: { label: 'Thiếu dữ liệu', color: '#5d5d5d' },
};

const approvalLabels = {
  submit: 'Gửi hồ sơ',
  kiem_soat: 'Kiểm soát viên duyệt',
  phe_duyet: 'Trưởng phòng phê duyệt',
  tu_choi: 'Từ chối',
};

async function apiRequest(token, path, options = {}) {
  const headers = options.body instanceof FormData
    ? { ...(options.headers || {}) }
    : { 'Content-Type': 'application/json', ...(options.headers || {}) };
  if (token) headers.Authorization = `Bearer ${token}`;
  const response = await fetch(`${API_BASE}${path}`, { ...options, headers });
  let data = null;
  try { data = await response.json(); } catch { /* response không có body */ }
  if (!response.ok) throw new Error(data?.detail || `Yêu cầu thất bại (HTTP ${response.status})`);
  return data;
}

function formatDate(value) {
  if (!value) return '—';
  return new Date(value).toLocaleString('vi-VN', { dateStyle: 'short', timeStyle: 'short' });
}

function formatChatTime(value) {
  if (!value) return 'Vừa xong';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return 'Vừa xong';
  return date.toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' });
}

function StatusPill({ status, document = false }) {
  const meta = (document ? DOC_STATUS_META : STATUS_META)[status] || STATUS_META.moi_tao;
  return <span className="status-pill" style={{ color: meta.color, background: meta.bg }}>{meta.label}</span>;
}

function PageHeader({ title, description, actions }) {
  return <div className="business-header">
    <div><span className="eyebrow">NỀN TẢNG PHÂN TÍCH TÀI CHÍNH</span><h1>{title}</h1><p>{description}</p></div>
    {actions && <div className="business-actions">{actions}</div>}
  </div>;
}

function ErrorNotice({ message }) {
  return message ? <div className="notice notice-error"><AlertTriangle size={16} />{message}</div> : null;
}

function LoadingState() {
  return <div className="loading-state"><RefreshCw size={20} className="spin" /> Đang tải dữ liệu...</div>;
}

function EmptyState({ icon: Icon = Folder, children }) {
  return <div className="empty-state"><Icon size={28} /><span>{children}</span></div>;
}

function OverviewPage({ token, onNavigate, onSelectHoSo }) {
  const [hoSo, setHoSo] = useState([]);
  const [docs, setDocs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const load = async () => {
    setLoading(true); setError('');
    try {
      const [profiles, documents] = await Promise.all([apiRequest(token, '/ho-so'), apiRequest(token, '/documents')]);
      setHoSo(profiles); setDocs(documents);
    } catch (err) { setError(err.message); } finally { setLoading(false); }
  };
  useEffect(() => { load(); }, [token]);

  const counts = Object.keys(STATUS_META).reduce((result, key) => ({ ...result, [key]: hoSo.filter(h => h.trang_thai === key).length }), {});
  const pendingDocs = docs.filter(d => d.status === 'uploaded' || d.status === 'processing').length;
  return <main className="business-page">
    <PageHeader title="Tổng quan" description="Dashboard doanh nghiệp và hồ sơ" actions={<button className="btn btn-ghost" onClick={load}><RefreshCw size={15} /> Làm mới</button>} />
    <ErrorNotice message={error} />
    {loading ? <LoadingState /> : <>
      <div className="stats-grid">
        <div className="stat-card"><span>TỔNG HỒ SƠ</span><strong>{hoSo.length}</strong><small>Doanh nghiệp đang theo dõi</small></div>
        <div className="stat-card stat-card--gold"><span>ĐANG XỬ LÝ</span><strong>{(counts.moi_tao || 0) + (counts.cho_kiem_soat || 0) + (counts.cho_phe_duyet || 0)}</strong><small>{pendingDocs} tài liệu đang OCR/index</small></div>
        <div className="stat-card stat-card--green"><span>ĐÃ DUYỆT</span><strong>{counts.da_duyet || 0}</strong><small>Hồ sơ hoàn tất quy trình</small></div>
        <div className="stat-card stat-card--red"><span>CẦN LƯU Ý</span><strong>{counts.tu_choi || 0}</strong><small>Hồ sơ bị từ chối</small></div>
      </div>
      <div className="dashboard-grid">
        <section className="panel"><div className="panel-title"><div><span className="section-kicker">QUY TRÌNH</span><h2>Trạng thái hồ sơ</h2></div><button className="link-button" onClick={() => onNavigate('profiles')}>Xem tất cả</button></div>
          {Object.entries(STATUS_META).map(([key, meta]) => <div className="bar-row" key={key}><div><span>{meta.label}</span><b>{counts[key] || 0}</b></div><div className="bar-track"><i style={{ width: `${hoSo.length ? Math.max(4, ((counts[key] || 0) / hoSo.length) * 100) : 0}%`, background: meta.color }} /></div></div>)}
        </section>
        <section className="panel quick-panel"><div className="panel-title"><div><span className="section-kicker">THAO TÁC NHANH</span><h2>Bắt đầu công việc</h2></div></div>
          <button onClick={() => onNavigate('profiles')}><Folder size={18} /><span><b>Tạo hồ sơ doanh nghiệp</b><small>Nhập thông tin khách hàng mới</small></span></button>
          <button onClick={() => onNavigate('ocr')}><FileText size={18} /><span><b>Tải tài liệu và chạy OCR</b><small>Số hoá báo cáo tài chính</small></span></button>
          <button onClick={() => onNavigate('analysis')}><BarChart3 size={18} /><span><b>Phân tích chỉ số</b><small>Kiểm tra số liệu và cảnh báo</small></span></button>
        </section>
      </div>
      <section className="panel"><div className="panel-title"><div><span className="section-kicker">CẬP NHẬT GẦN ĐÂY</span><h2>Hồ sơ mới cập nhật</h2></div></div>
        {hoSo.length === 0 ? <EmptyState icon={Folder}>Chưa có hồ sơ. Hãy tạo hồ sơ đầu tiên để bắt đầu.</EmptyState> : <div className="table-wrap"><table className="data-table"><thead><tr><th>Doanh nghiệp</th><th>Ngành nghề</th><th>Phụ trách</th><th>Trạng thái</th><th>Cập nhật</th><th /></tr></thead><tbody>{hoSo.slice(0, 8).map(h => <tr key={h.id}><td><b>{h.ten_doanh_nghiep}</b><small>MST {h.ma_so_thue || '—'}</small></td><td>{h.nganh_nghe || '—'}</td><td>{h.nguoi_phu_trach_ten || '—'}</td><td><StatusPill status={h.trang_thai} /></td><td>{formatDate(h.updated_at)}</td><td><button className="table-action" onClick={() => { onSelectHoSo(h.id); onNavigate('profiles'); }}>Mở</button></td></tr>)}</tbody></table></div>}
      </section>
    </>}
  </main>;
}

function HoSoPage({ token, selectedHoSoId, onSelectHoSo, onNavigate }) {
  const [profiles, setProfiles] = useState([]); const [users, setUsers] = useState([]); const [detail, setDetail] = useState(null);
  const [form, setForm] = useState({ ten_doanh_nghiep: '', ma_so_thue: '', nganh_nghe: '', nguoi_phu_trach_id: '' });
  const [showForm, setShowForm] = useState(false); const [loading, setLoading] = useState(true); const [saving, setSaving] = useState(false); const [error, setError] = useState('');
  const load = async () => { setLoading(true); try { const [hs, us] = await Promise.all([apiRequest(token, '/ho-so'), apiRequest(token, '/users').catch(() => [])]); setProfiles(hs); setUsers(us); } catch (err) { setError(err.message); } finally { setLoading(false); } };
  useEffect(() => { load(); }, [token]);
  useEffect(() => { if (!selectedHoSoId) { setDetail(null); return; } apiRequest(token, `/ho-so/${selectedHoSoId}`).then(setDetail).catch(err => setError(err.message)); }, [selectedHoSoId, token]);
  const create = async e => { e.preventDefault(); if (!form.ten_doanh_nghiep.trim()) return setError('Vui lòng nhập tên doanh nghiệp.'); setSaving(true); setError(''); try { const created = await apiRequest(token, '/ho-so', { method: 'POST', body: JSON.stringify({ ...form, nguoi_phu_trach_id: form.nguoi_phu_trach_id || null }) }); setForm({ ten_doanh_nghiep: '', ma_so_thue: '', nganh_nghe: '', nguoi_phu_trach_id: '' }); setShowForm(false); await load(); onSelectHoSo(created.id); } catch (err) { setError(err.message); } finally { setSaving(false); } };
  const submit = async () => { try { await apiRequest(token, `/approvals/${detail.id}/submit`, { method: 'POST', body: JSON.stringify({ comment: '' }) }); const fresh = await apiRequest(token, `/ho-so/${detail.id}`); setDetail(fresh); await load(); } catch (err) { setError(err.message); } };
  return <main className="business-page"><PageHeader title="Hồ sơ phân tích" description="Quản lý hồ sơ doanh nghiệp đang theo dõi" actions={<button className="btn btn-primary" onClick={() => setShowForm(v => !v)}><Plus size={16} /> Tạo hồ sơ</button>} /><ErrorNotice message={error} />
    {showForm && <form className="panel profile-form" onSubmit={create}><div className="panel-title"><div><span className="section-kicker">HỒ SƠ MỚI</span><h2>Thông tin doanh nghiệp</h2></div></div><div className="form-grid"><label>Tên doanh nghiệp<input value={form.ten_doanh_nghiep} onChange={e => setForm({ ...form, ten_doanh_nghiep: e.target.value })} placeholder="Công ty cổ phần..." /></label><label>Mã số thuế<input value={form.ma_so_thue} onChange={e => setForm({ ...form, ma_so_thue: e.target.value })} placeholder="Mã số thuế" /></label><label>Ngành nghề<input value={form.nganh_nghe} onChange={e => setForm({ ...form, nganh_nghe: e.target.value })} placeholder="Sản xuất, thương mại..." /></label><label>Người phụ trách<select value={form.nguoi_phu_trach_id} onChange={e => setForm({ ...form, nguoi_phu_trach_id: e.target.value })}><option value="">— Chọn cán bộ —</option>{users.filter(u => u.role === 'can_bo').map(u => <option key={u.id} value={u.id}>{u.full_name}</option>)}</select></label></div><div className="form-actions"><button type="button" className="btn btn-ghost" onClick={() => setShowForm(false)}>Huỷ</button><button className="btn btn-primary" disabled={saving}>{saving ? 'Đang lưu...' : 'Tạo hồ sơ'}</button></div></form>}
    {loading ? <LoadingState /> : <div className="split-layout"><section className="panel profile-list"><div className="panel-title"><div><span className="section-kicker">DANH MỤC</span><h2>{profiles.length} hồ sơ</h2></div><button className="icon-button" onClick={load}><RefreshCw size={15} /></button></div>{profiles.length === 0 ? <EmptyState>Chưa có hồ sơ nào.</EmptyState> : profiles.map(p => <button className={`profile-item ${p.id === selectedHoSoId ? 'active' : ''}`} key={p.id} onClick={() => onSelectHoSo(p.id)}><div><b>{p.ten_doanh_nghiep}</b><small>{p.nganh_nghe || 'Chưa phân loại'} · MST {p.ma_so_thue || '—'}</small></div><StatusPill status={p.trang_thai} /></button>)}</section><section className="panel detail-panel">{!detail ? <EmptyState icon={Folder}>Chọn một hồ sơ để xem chi tiết.</EmptyState> : <><div className="detail-head"><div><span className="section-kicker">CHI TIẾT HỒ SƠ</span><h2>{detail.ten_doanh_nghiep}</h2><p>MST {detail.ma_so_thue || '—'} · {detail.nganh_nghe || '—'} · Phụ trách: {detail.nguoi_phu_trach_ten || '—'}</p></div><StatusPill status={detail.trang_thai} /></div><div className="detail-actions"><button className="btn btn-ghost" onClick={() => onNavigate('ocr')}><FileText size={15} /> Tài liệu ({detail.documents?.length || 0})</button><button className="btn btn-ghost" onClick={() => onNavigate('analysis')}><BarChart3 size={15} /> Phân tích</button>{detail.trang_thai === 'moi_tao' && <button className="btn btn-primary" onClick={submit}>Gửi kiểm soát</button>}</div><div className="timeline"><h3>Lịch sử phê duyệt</h3>{!detail.approval_events?.length ? <p className="muted">Chưa có sự kiện phê duyệt.</p> : detail.approval_events.map(ev => <div className="timeline-item" key={ev.id}><i /><div><b>{approvalLabels[ev.step] || ev.step}</b><small>{ev.actor_ten || 'Hệ thống'} · {formatDate(ev.created_at)}</small>{ev.comment && <span>{ev.comment}</span>}</div></div>)}</div></>}</section></div>}
  </main>;
}

function DocumentsPage({ token, selectedHoSoId, onSelectHoSo, onNavigate }) {
  const [profiles, setProfiles] = useState([]); const [docs, setDocs] = useState([]); const [active, setActive] = useState(null); const [chunks, setChunks] = useState([]); const [loading, setLoading] = useState(false); const [uploading, setUploading] = useState(false); const [deleting, setDeleting] = useState(false); const [reportingPeriod, setReportingPeriod] = useState(''); const [error, setError] = useState(''); const fileRef = useRef(null);
  useEffect(() => { apiRequest(token, '/ho-so').then(data => { setProfiles(data); if (!selectedHoSoId && data[0]) onSelectHoSo(data[0].id); }).catch(err => setError(err.message)); }, [token, selectedHoSoId, onSelectHoSo]);
  const load = async () => { if (!selectedHoSoId) return setDocs([]); setLoading(true); try { const fresh = await apiRequest(token, `/documents?ho_so_id=${encodeURIComponent(selectedHoSoId)}`); setDocs(fresh); setActive(previous => previous ? (fresh.find(d => d.id === previous.id) || null) : null); } catch (err) { setError(err.message); } finally { setLoading(false); } };
  useEffect(() => { setActive(null); setChunks([]); load(); }, [selectedHoSoId, token]);
  useEffect(() => { if (!docs.some(d => d.status === 'uploaded' || d.status === 'processing')) return undefined; const timer = setInterval(load, 2500); return () => clearInterval(timer); }, [docs, selectedHoSoId, token]);
  const upload = async e => { const file = e.target.files?.[0]; e.target.value = ''; if (!file || !selectedHoSoId) return; if (!file.name.toLowerCase().endsWith('.pdf')) return setError('Chỉ hỗ trợ file PDF.'); setUploading(true); setError(''); try { const body = new FormData(); body.append('file', file); const period = reportingPeriod.trim(); await apiRequest(token, `/documents/upload?ho_so_id=${encodeURIComponent(selectedHoSoId)}${period ? `&reporting_period=${encodeURIComponent(period)}` : ''}`, { method: 'POST', body }); await load(); } catch (err) { setError(err.message); } finally { setUploading(false); } };
  const preview = async doc => { setActive(doc); setReportingPeriod(doc.reporting_period || ''); setChunks([]); if (doc.status !== 'indexed') return; try { const data = await apiRequest(token, `/documents/${doc.id}/text`); setChunks(data.chunks || []); } catch (err) { setError(err.message); } };
  useEffect(() => { if (active?.status === 'indexed') apiRequest(token, `/documents/${active.id}/text`).then(data => setChunks(data.chunks || [])).catch(e => setError(e.message)); }, [active?.id, active?.status, token]);
  const retryOcr = async () => { if (!active) return; setError(''); try { await apiRequest(token, `/documents/${active.id}/retry`, {method:'POST'}); await load(); } catch(e) { setError(e.message); } };
  const saveReportingPeriod = async () => { if (!active) return; try { const updated = await apiRequest(token, `/documents/${active.id}/reporting-period`, { method: 'PATCH', body: JSON.stringify({ reporting_period: reportingPeriod.trim() || null }) }); setActive(updated); await load(); } catch (err) { setError(err.message); } };
  const deleteIncomplete = async () => { if (!active || !['uploaded', 'failed'].includes(active.status)) return; if (!window.confirm(`Xoá ${active.filename}? Tài liệu sẽ được gỡ khỏi danh sách. File nguồn và cache được giữ lại.`)) return; setDeleting(true); setError(''); try { await apiRequest(token, `/documents/${active.id}`, { method: 'DELETE' }); setActive(null); setChunks([]); await load(); } catch (err) { setError(err.message); } finally { setDeleting(false); } };
  const activeOcrPages = new Set(active?.ocr_pages ? JSON.parse(active.ocr_pages) : []);
  const canDeleteIncomplete = active && ['uploaded', 'failed'].includes(active.status);
  return <main className="business-page"><PageHeader title="Tài liệu & OCR" description="Tải lên, gắn kỳ báo cáo, nhận dạng và lập chỉ mục báo cáo tài chính" actions={<><select className="select-control" value={selectedHoSoId || ''} onChange={e => onSelectHoSo(e.target.value || null)}><option value="">— Chọn hồ sơ —</option>{profiles.map(p => <option key={p.id} value={p.id}>{p.ten_doanh_nghiep}</option>)}</select><input className="select-control period-control" value={reportingPeriod} onChange={e => setReportingPeriod(e.target.value)} placeholder="Kỳ báo cáo, vd 2025" /><button className="btn btn-primary" disabled={!selectedHoSoId || uploading} onClick={() => fileRef.current?.click()}><UploadCloud size={16} /> {uploading ? 'Đang tải...' : 'Tải PDF'}</button><input ref={fileRef} type="file" accept="application/pdf" hidden onChange={upload} /></>} /><ErrorNotice message={error} />{!selectedHoSoId ? <div className="empty-state empty-state--action"><Folder size={28} /><span>{profiles.length === 0 ? 'Chưa có hồ sơ để gắn tài liệu.' : 'Chọn hồ sơ để tải lên và xem tài liệu.'}</span>{profiles.length === 0 && <button className="btn btn-primary" onClick={() => onNavigate('profiles')}><Plus size={15} /> Tạo hồ sơ trước</button>}</div> : <div className="split-layout docs-layout"><section className="panel profile-list"><div className="panel-title"><div><span className="section-kicker">TÀI LIỆU</span><h2>{docs.length} tệp PDF</h2></div><button className="icon-button" onClick={load}><RefreshCw size={15} /></button></div>{loading ? <LoadingState /> : docs.length === 0 ? <EmptyState icon={FileText}>Hồ sơ chưa có tài liệu.</EmptyState> : docs.map(d => <button className={`profile-item ${active?.id === d.id ? 'active' : ''}`} key={d.id} onClick={() => preview(d)}><div><b>{d.filename}</b><small>{d.reporting_period ? `Kỳ ${d.reporting_period} · ` : ''}{d.total_pages || '—'} trang · {d.chunk_count || 0} đoạn</small></div><StatusPill status={d.status} document /></button>)}</section><section className="panel detail-panel">{!active ? <EmptyState icon={FileText}>Chọn tài liệu để xem nội dung đã trích xuất.</EmptyState> : <><div className="detail-head"><div><span className="section-kicker">XEM TRƯỚC OCR</span><h2>{active.filename}</h2><p>{active.total_pages || '—'} trang · {active.chunk_count || 0} đoạn văn bản</p></div><StatusPill status={active.status} document /></div><div className="document-period-editor"><label>Kỳ báo cáo<input value={reportingPeriod} onChange={e => setReportingPeriod(e.target.value)} placeholder="Ví dụ: 2025" /></label><button className="btn btn-ghost" onClick={saveReportingPeriod}>Lưu kỳ</button>{active.status === 'failed' && <button className="btn btn-primary" onClick={retryOcr}>Chạy lại OCR</button>}{canDeleteIncomplete && <button className="btn btn-danger" onClick={deleteIncomplete} disabled={deleting}><Trash2 size={15} /> {deleting ? 'Đang xoá...' : 'Xoá tài liệu treo'}</button>}</div>{active.status !== 'indexed' ? <div className="notice notice-warning"><RefreshCw size={16} /> {active.status === 'failed' ? (active.error || 'Xử lý thất bại. Kiểm tra MinerU.') : 'MinerU đang xử lý. Chọn lại tài liệu sau khi danh sách cập nhật.'}</div> : chunks.length === 0 ? <EmptyState>Không có nội dung trích xuất.</EmptyState> : <div className="chunk-list">{chunks.map((chunk, i) => <article className="chunk" key={chunk.chunk_id || i}><div><span className="page-pill">Trang {chunk.page_number}</span>{activeOcrPages.has(chunk.page_number) && <span className="ocr-pill">OCR · cần kiểm tra</span>}</div><p>{chunk.text}</p></article>)}</div>}</>}</section></div>}</main>;
}

function AnalysisPage({ token, selectedHoSoId, onSelectHoSo }) {
  const [profiles, setProfiles] = useState([]);
  const [docs, setDocs] = useState([]);
  const [docId, setDocId] = useState('');
  const [data, setData] = useState(null);
  const [trend, setTrend] = useState(null);
  const [stress, setStress] = useState({ interest_rate_shock_pct: 2, exchange_rate_shock_pct: 3, cashflow_decline_pct: 10, cycle_extension_days: 15, foreign_currency_debt: 0 });
  const [stressResult, setStressResult] = useState(null);
  const [stressLoading, setStressLoading] = useState(false);
  const [editing, setEditing] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    apiRequest(token, '/ho-so').then(setProfiles).catch(err => setError(err.message));
  }, [token]);

  useEffect(() => {
    setDocs([]); setDocId(''); setData(null);
    if (selectedHoSoId) {
      apiRequest(token, `/documents?ho_so_id=${encodeURIComponent(selectedHoSoId)}`)
        .then(setDocs).catch(err => setError(err.message));
    }
  }, [selectedHoSoId, token]);

  useEffect(() => {
    setTrend(null);
    if (selectedHoSoId) apiRequest(token, `/financials/profile/${selectedHoSoId}/trend`).then(setTrend).catch(err => setError(err.message));
  }, [selectedHoSoId, token]);

  const load = async id => {
    setDocId(id);
    if (!id) { setData(null); return; }
    setLoading(true); setError('');
    try { const result = await apiRequest(token, `/financials/${id}`); setData(result); if (result.cache_upgrading) window.setTimeout(() => load(id), 5000); }
    catch (err) { setData(null); setError(err.message); }
    finally { setLoading(false); }
  };

  const save = async () => {
    const values = Object.fromEntries(
      Object.entries(editing)
        .filter(([, value]) => value !== '' && Number.isFinite(Number(value)))
        .map(([key, value]) => [key, Number(value)])
    );
    if (!Object.keys(values).length) return;
    try {
      setData(await apiRequest(token, `/financials/${docId}/override`, {
        method: 'PATCH', body: JSON.stringify({ values }),
      }));
      setEditing({});
    } catch (err) { setError(err.message); }
  };

  const runStressTest = async e => {
    e.preventDefault(); if (!docId) return;
    setStressLoading(true); setError('');
    try { setStressResult(await apiRequest(token, `/financials/${docId}/stress-test`, { method: 'POST', body: JSON.stringify(Object.fromEntries(Object.entries(stress).map(([key, value]) => [key, Number(value) || 0]))) })); }
    catch (err) { setError(err.message); } finally { setStressLoading(false); }
  };

  return <main className="business-page">
    <PageHeader title="Phân tích" description="Đánh giá sức khỏe tài chính, rủi ro, dòng tiền và điểm cần thẩm định" actions={<>
      <select className="select-control" value={selectedHoSoId || ''} onChange={e => onSelectHoSo(e.target.value || null)}>
        <option value="">— Chọn hồ sơ —</option>
        {profiles.map(profile => <option key={profile.id} value={profile.id}>{profile.ten_doanh_nghiep}</option>)}
      </select>
      <select className="select-control" value={docId} onChange={e => load(e.target.value)} disabled={!selectedHoSoId}>
        <option value="">— Chọn tài liệu —</option>
        {docs.map(doc => <option key={doc.id} value={doc.id}>{doc.filename}</option>)}
      </select>
    </>} />
    <ErrorNotice message={error} />
    {!docId ? <EmptyState icon={BarChart3}>Chọn hồ sơ và tài liệu đã lập chỉ mục để xem phân tích.</EmptyState> : loading ? <LoadingState /> : data ? <>
      {data.cache_upgrading && <div className="notice notice-warning"><RefreshCw size={16} className="spin" /> Đang nâng cấp phân tích đầy đủ trong nền. Số liệu cache hiện có đã được hiển thị; trang sẽ tự tải lại khi hoàn tất.</div>}
      <div className="ratio-grid">
        {data.ratios.map(ratio => <div className={`ratio-card ${ratio.flag || ''}`} key={ratio.key}>
          <span>{ratio.label}</span>
          <strong>{ratio.value == null ? '—' : `${Number(ratio.value).toFixed(2)} ${ratio.unit}`}</strong>
          <small>{ratio.benchmark_note}</small>
        </div>)}
      </div>
      {data.assessment && <>
        <section className={`assessment-hero assessment-hero--${data.assessment.health?.level === 'Rủi ro cao' ? 'danger' : data.assessment.health?.level === 'Cần theo dõi' ? 'warning' : 'good'}`}>
          <div><span className="section-kicker">NHẬN ĐỊNH TÍN DỤNG SƠ BỘ</span><h2>{data.assessment.health?.level || 'Chưa đủ dữ liệu'}</h2><p>{data.assessment.health?.summary}</p></div>
          <div className="assessment-score"><strong>{data.assessment.health?.score ?? '—'}</strong><span>/ 100</span><small>điểm chỉ báo</small></div>
        </section>
        <div className="assessment-grid">
          <section className="panel"><div className="panel-title"><div><span className="section-kicker">4 TRỤ CỘT</span><h2>Sức khỏe và rủi ro trọng yếu</h2></div></div>
            <div className="assessment-dimensions">{(data.assessment.dimensions || []).map(dimension => { const meta = ASSESSMENT_META[dimension.level] || ASSESSMENT_META.missing; return <article className={`assessment-dimension assessment-dimension--${dimension.level}`} key={dimension.key}><div><h3>{dimension.title}</h3><span style={{ color: meta.color }}>{meta.label}</span></div><p>{dimension.summary}</p>{dimension.evidence?.length > 0 && <ul>{dimension.evidence.map((item, index) => <li key={index}>{item}</li>)}</ul>}</article>; })}</div>
          </section>
          <section className="panel assessment-actions"><div className="panel-title"><div><span className="section-kicker">KHUYẾN NGHỊ NGÂN HÀNG</span><h2>Việc cần làm trước khi quyết định</h2></div></div>
            <div className="assessment-list"><h3>Điểm cần kiểm tra</h3><ul>{(data.assessment.investigation_points || []).map((item, index) => <li key={index}>{item}</li>)}</ul></div>
            <div className="assessment-list"><h3>Định hướng cấp tín dụng</h3><ul>{(data.assessment.banking_recommendations || []).map((item, index) => <li key={index}>{item}</li>)}</ul></div>
            <p className="assessment-note">{data.assessment.data_quality?.note}</p>
          </section>
        </div>
      </>}
      <div className="analysis-extended-grid">
        <section className="panel"><div className="panel-title"><div><span className="section-kicker">THUYẾT MINH BCTC</span><h2>Bằng chứng cần thẩm định</h2></div></div>
          <p className="muted">Chưa tích hợp tự động thẩm định thuyết minh vào màn hình này. Dùng Trợ lý để hỏi và đối chiếu bằng chứng.</p>
          <div className="disclosure-list">{(data.disclosures || []).map(item => <article className={`disclosure ${item.found ? 'found' : 'missing'}`} key={item.key}><div><b>{item.title}</b><span>{item.found ? `Có đề cập · tr. ${item.source_page}` : 'Chưa tìm thấy từ khóa'}</span></div>{item.snippet && <p>“{item.snippet}”</p>}<small>{item.review}</small></article>)}</div>
        </section>
        <section className="panel"><div className="panel-title"><div><span className="section-kicker">SO SÁNH NGÀNH</span><h2>{data.benchmark?.industry_label || 'Chưa phân loại'}</h2></div></div>
          <p className="benchmark-source">{data.benchmark?.source}</p>
          <div className="benchmark-list">{(data.benchmark?.metrics || []).map(metric => <div key={metric.key}><span>{metric.label}</span><b>{metric.value == null ? '—' : Number(metric.value).toFixed(2)}</b><small className={metric.status}>{metric.min != null ? `≥ ${metric.min}` : `≤ ${metric.max}`} · {metric.status === 'good' ? 'Trong ngưỡng' : metric.status === 'warning' ? 'Cần theo dõi' : 'Thiếu dữ liệu'}</small></div>)}</div>
        </section>
      </div>
      <section className="panel trend-panel"><div className="panel-title"><div><span className="section-kicker">XU HƯỚNG NHIỀU KỲ</span><h2>So sánh các báo cáo của cùng doanh nghiệp</h2></div></div>
        {!trend ? <LoadingState /> : trend.needs_more_periods ? <EmptyState icon={LineChart}>Cần gắn kỳ báo cáo và tải ít nhất 2 PDF đã lập chỉ mục để phân tích xu hướng.</EmptyState> : <div className="table-wrap"><table className="data-table"><thead><tr><th>Kỳ báo cáo</th><th>Doanh thu</th><th>LNST</th><th>LCTT HĐKD</th><th>Tổng tài sản</th><th>Thanh khoản</th><th>Nợ/VCSH</th></tr></thead><tbody>{trend.points.map(point => <tr key={point.document_id}><td><b>{point.period}</b></td><td>{point.revenue == null ? '—' : Number(point.revenue).toLocaleString('vi-VN')}</td><td>{point.profit_after_tax == null ? '—' : Number(point.profit_after_tax).toLocaleString('vi-VN')}</td><td>{point.operating_cashflow == null ? '—' : Number(point.operating_cashflow).toLocaleString('vi-VN')}</td><td>{point.total_assets == null ? '—' : Number(point.total_assets).toLocaleString('vi-VN')}</td><td>{point.current_ratio == null ? '—' : `${Number(point.current_ratio).toFixed(2)}x`}</td><td>{point.debt_to_equity == null ? '—' : `${Number(point.debt_to_equity).toFixed(2)}x`}</td></tr>)}</tbody></table></div>}
        <p className="assessment-note">{trend?.note}</p>
      </section>
      <section className="panel stress-panel"><div className="panel-title"><div><span className="section-kicker">STRESS TEST</span><h2>Áp lực lãi suất, tỷ giá, dòng tiền và chu kỳ kinh doanh</h2></div></div>
        <form onSubmit={runStressTest}><div className="stress-inputs"><label>Tăng chi phí lãi vay (%)<input type="number" min="0" step="0.1" value={stress.interest_rate_shock_pct} onChange={e => setStress({ ...stress, interest_rate_shock_pct: e.target.value })} /></label><label>Shock tỷ giá (%)<input type="number" min="0" step="0.1" value={stress.exchange_rate_shock_pct} onChange={e => setStress({ ...stress, exchange_rate_shock_pct: e.target.value })} /></label><label>Suy giảm dòng tiền (%)<input type="number" min="0" max="100" step="1" value={stress.cashflow_decline_pct} onChange={e => setStress({ ...stress, cashflow_decline_pct: e.target.value })} /></label><label>Kéo dài chu kỳ thu tiền (ngày)<input type="number" min="0" step="1" value={stress.cycle_extension_days} onChange={e => setStress({ ...stress, cycle_extension_days: e.target.value })} /></label><label>Dư nợ ngoại tệ (triệu đồng)<input type="number" min="0" step="1" value={stress.foreign_currency_debt} onChange={e => setStress({ ...stress, foreign_currency_debt: e.target.value })} /></label><button className="btn btn-primary" disabled={stressLoading}>{stressLoading ? 'Đang tính...' : 'Chạy kịch bản'}</button></div></form>
        {stressResult && <div className={`stress-result ${stressResult.risk_level === 'Rủi ro cao' ? 'danger' : stressResult.risk_level === 'Cần theo dõi' ? 'warning' : ''}`}><div><b>Kết quả: {stressResult.risk_level}</b>{stressResult.conclusions?.map((item, index) => <span key={index}>{item}</span>)}</div>{stressResult.missing_inputs?.length > 0 && <small>Thiếu dữ liệu: {stressResult.missing_inputs.join('; ')}</small>}<p>{stressResult.disclaimer}</p></div>}
      </section>
      {data.red_flags?.length > 0 && <div className="notice notice-error warning-list"><AlertTriangle size={18} /><div><b>Dấu hiệu bất thường cần lưu ý</b>{data.red_flags.map((flag, index) => <span key={index}>{flag}</span>)}</div></div>}
      <section className="panel">
        <div className="panel-title"><div><span className="section-kicker">DỮ LIỆU GỐC</span><h2>Chỉ tiêu tài chính</h2></div>{Object.keys(editing).length > 0 && <button className="btn btn-primary" onClick={save}>Lưu điều chỉnh</button>}</div>
        <p className="muted">Bảng này dùng số liệu do người dùng nhập và xác nhận, đơn vị triệu đồng. Chưa tự động điền từ HRM. Hỏi số liệu có bằng chứng ở màn hình Trợ lý phân tích.</p>
        <div className="table-wrap"><table className="data-table"><thead><tr><th>Chỉ tiêu</th><th>Giá trị</th><th>Đơn vị</th><th>Nguồn</th></tr></thead><tbody>
          {data.line_items.map(item => <tr key={item.key}><td><b>{item.label}</b></td><td><input className="table-input" type="number" key={`${item.key}-${item.value}`} defaultValue={item.value ?? ''} onChange={e => setEditing({ ...editing, [item.key]: e.target.value })} /></td><td>{item.unit}</td><td><span className="confidence-pill">{item.confidence === 'manual' ? 'Người dùng nhập' : item.value == null ? 'Chưa có số liệu' : 'HRM xác minh'}{item.source_page ? ` · tr. ${item.source_page}` : ''}</span></td></tr>)}
        </tbody></table></div>
      </section>
    </> : null}
  </main>;
}

function ReportsPage({ token, onSelectHoSo, onNavigate }) {
  const [profiles, setProfiles] = useState([]); const [loading, setLoading] = useState(true); const [error, setError] = useState('');
  const load = () => { setLoading(true); apiRequest(token, '/ho-so').then(setProfiles).catch(err => setError(err.message)).finally(() => setLoading(false)); };
  useEffect(() => { load(); }, [token]);
  const counts = Object.keys(STATUS_META).reduce((a, key) => ({ ...a, [key]: profiles.filter(p => p.trang_thai === key).length }), {}); const max = Math.max(1, ...Object.values(counts)); const approved = profiles.length ? ((counts.da_duyet / profiles.length) * 100).toFixed(1) : '0.0';
  const exportCsv = () => { const rows = [['Doanh nghiệp', 'Mã số thuế', 'Ngành nghề', 'Trạng thái', 'Người phụ trách'], ...profiles.map(p => [p.ten_doanh_nghiep, p.ma_so_thue || '', p.nganh_nghe || '', STATUS_META[p.trang_thai]?.label || p.trang_thai, p.nguoi_phu_trach_ten || ''])]; const csv = '\ufeff' + rows.map(row => row.map(v => `"${String(v).replaceAll('"', '""')}"`).join(',')).join('\n'); const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' })); const a = document.createElement('a'); a.href = url; a.download = `bao-cao-ho-so-${new Date().toISOString().slice(0, 10)}.csv`; a.click(); URL.revokeObjectURL(url); };
  return <main className="business-page"><PageHeader title="Thống kê & báo cáo" description="Tổng hợp danh mục và tiến độ xử lý hồ sơ" actions={<><button className="btn btn-ghost" onClick={load}><RefreshCw size={15} /> Làm mới</button><button className="btn btn-primary" onClick={exportCsv} disabled={!profiles.length}><Download size={15} /> Xuất CSV</button></>} /><ErrorNotice message={error} />{loading ? <LoadingState /> : <><div className="stats-grid report-stats"><div className="stat-card"><span>TỔNG HỒ SƠ</span><strong>{profiles.length}</strong></div><div className="stat-card stat-card--green"><span>TỶ LỆ ĐÃ DUYỆT</span><strong>{approved}%</strong></div><div className="stat-card stat-card--gold"><span>ĐANG XỬ LÝ</span><strong>{counts.moi_tao + counts.cho_kiem_soat + counts.cho_phe_duyet}</strong></div><div className="stat-card stat-card--red"><span>TỪ CHỐI</span><strong>{counts.tu_choi}</strong></div></div><div className="dashboard-grid"><section className="panel"><div className="panel-title"><div><span className="section-kicker">TIẾN ĐỘ</span><h2>Phân bố theo trạng thái</h2></div></div>{Object.entries(STATUS_META).map(([key, meta]) => <div className="bar-row" key={key}><div><span>{meta.label}</span><b>{counts[key]}</b></div><div className="bar-track"><i style={{ width: `${Math.max(counts[key] ? 5 : 0, (counts[key] / max) * 100)}%`, background: meta.color }} /></div></div>)}</section><section className="panel"><div className="panel-title"><div><span className="section-kicker">NGÀNH NGHỀ</span><h2>Phân bố danh mục</h2></div></div>{Object.entries(profiles.reduce((a, p) => ({ ...a, [p.nganh_nghe || 'Chưa phân loại']: (a[p.nganh_nghe || 'Chưa phân loại'] || 0) + 1 }), {})).map(([name, count]) => <div className="bar-row" key={name}><div><span>{name}</span><b>{count}</b></div><div className="bar-track"><i style={{ width: `${Math.max(5, (count / max) * 100)}%`, background: C.wine }} /></div></div>)}{!profiles.length && <p className="muted">Chưa có dữ liệu.</p>}</section></div><section className="panel"><div className="panel-title"><div><span className="section-kicker">DANH SÁCH</span><h2>Hồ sơ trong báo cáo</h2></div></div>{profiles.length ? <div className="table-wrap"><table className="data-table"><thead><tr><th>Doanh nghiệp</th><th>Ngành nghề</th><th>Phụ trách</th><th>Trạng thái</th><th /></tr></thead><tbody>{profiles.map(p => <tr key={p.id}><td><b>{p.ten_doanh_nghiep}</b></td><td>{p.nganh_nghe || '—'}</td><td>{p.nguoi_phu_trach_ten || '—'}</td><td><StatusPill status={p.trang_thai} /></td><td><button className="table-action" onClick={() => { onSelectHoSo(p.id); onNavigate('profiles'); }}>Xem hồ sơ</button></td></tr>)}</tbody></table></div> : <EmptyState>Chưa có dữ liệu báo cáo.</EmptyState>}</section></>}</main>;
}

function ApprovalPage({ token, user }) {
  const [profiles, setProfiles] = useState([]); const [queue, setQueue] = useState([]); const [comment, setComment] = useState({}); const [role, setRole] = useState(user.role === 'quan_tri' ? 'kiem_soat' : user.role); const [loading, setLoading] = useState(false); const [error, setError] = useState('');
  const canSubmit = user.role === 'can_bo' || user.role === 'quan_tri'; const canReview = ['kiem_soat', 'truong_phong', 'quan_tri'].includes(user.role); const reviewRole = user.role === 'quan_tri' ? role : user.role;
  const load = async () => { setLoading(true); try { setProfiles(await apiRequest(token, '/ho-so')); if (['kiem_soat', 'truong_phong'].includes(reviewRole)) setQueue(await apiRequest(token, `/approvals/queue/${reviewRole}`)); else setQueue([]); } catch (err) { setError(err.message); } finally { setLoading(false); } };
  useEffect(() => { load(); }, [token, reviewRole]);
  const action = async (id, endpoint, body) => { try { await apiRequest(token, `/approvals/${id}/${endpoint}`, { method: 'POST', body: JSON.stringify(body) }); setComment({ ...comment, [id]: '' }); await load(); } catch (err) { setError(err.message); } };
  const submit = p => action(p.id, 'submit', { comment: '' }); const review = (p, approve) => action(p.id, reviewRole === 'kiem_soat' ? 'kiem-soat' : 'phe-duyet', { approve, comment: comment[p.id] || '' });
  const counts = Object.keys(STATUS_META).reduce((a, key) => ({ ...a, [key]: profiles.filter(p => p.trang_thai === key).length }), {});
  return <main className="business-page"><PageHeader title="Phê duyệt" description={`Luồng 3 cấp · Vai trò hiện tại: ${ROLE_LABELS[user.role] || user.role}`} actions={user.role === 'quan_tri' && <select className="select-control" value={role} onChange={e => setRole(e.target.value)}><option value="kiem_soat">Xem hàng chờ kiểm soát</option><option value="truong_phong">Xem hàng chờ phê duyệt</option></select>} /><ErrorNotice message={error} /><div className="approval-counts">{Object.entries(STATUS_META).map(([key, meta]) => <div key={key}><strong style={{ color: meta.color }}>{counts[key]}</strong><span>{meta.label}</span></div>)}</div>{loading ? <LoadingState /> : <>{canSubmit && profiles.filter(p => p.trang_thai === 'moi_tao').length > 0 && <section className="panel approval-section"><div className="panel-title"><div><span className="section-kicker">CÁN BỘ PHÂN TÍCH</span><h2>Hồ sơ chờ gửi kiểm soát</h2></div></div>{profiles.filter(p => p.trang_thai === 'moi_tao').map(p => <div className="approval-row" key={p.id}><div><b>{p.ten_doanh_nghiep}</b><small>{p.nguoi_phu_trach_ten || 'Chưa phân công'}</small></div><button className="btn btn-primary" onClick={() => submit(p)}>Gửi kiểm soát</button></div>)}</section>}{canReview && <section className="panel approval-section"><div className="panel-title"><div><span className="section-kicker">HÀNG CHỜ XỬ LÝ</span><h2>{queue.length} hồ sơ</h2></div></div>{queue.length === 0 ? <EmptyState icon={CheckSquare2}>Không có hồ sơ đang chờ xử lý.</EmptyState> : queue.map(p => <div className="approval-card" key={p.id}><div className="detail-head"><div><h3>{p.ten_doanh_nghiep}</h3><p>MST {p.ma_so_thue || '—'} · {p.nganh_nghe || '—'} · {p.nguoi_phu_trach_ten || '—'}</p></div><StatusPill status={p.trang_thai} /></div><textarea value={comment[p.id] || ''} onChange={e => setComment({ ...comment, [p.id]: e.target.value })} placeholder="Nhận xét (không bắt buộc)..." rows="2" /><div className="approval-buttons"><button className="btn btn-success" onClick={() => review(p, true)}><Check size={15} /> Duyệt</button><button className="btn btn-danger" onClick={() => review(p, false)}>Từ chối</button></div></div>)}</section>}{!canSubmit && !canReview && <EmptyState icon={CheckSquare2}>Vai trò hiện tại không có thao tác phê duyệt.</EmptyState>}</>}</main>;
}

function UserManagementPage({ token, user }) {
  const [users, setUsers] = useState([]); const [show, setShow] = useState(false); const [form, setForm] = useState({ username: '', full_name: '', role: 'can_bo', password: '' }); const [error, setError] = useState(''); const [loading, setLoading] = useState(true);
  const load = () => { setLoading(true); apiRequest(token, '/users').then(setUsers).catch(err => setError(err.message)).finally(() => setLoading(false)); }; useEffect(() => { load(); }, [token]);
  const create = async e => { e.preventDefault(); try { await apiRequest(token, '/users', { method: 'POST', body: JSON.stringify(form) }); setShow(false); setForm({ username: '', full_name: '', role: 'can_bo', password: '' }); load(); } catch (err) { setError(err.message); } };
  const toggle = async item => { try { await apiRequest(token, `/users/${item.id}/active`, { method: 'PATCH', body: JSON.stringify({ active: !item.active }) }); load(); } catch (err) { setError(err.message); } };
  return <main className="business-page"><PageHeader title="Quản lý người dùng" description="Phân quyền cán bộ, kiểm soát viên, trưởng phòng và quản trị hệ thống" actions={user.role === 'quan_tri' && <button className="btn btn-primary" onClick={() => setShow(v => !v)}><UserPlus size={15} /> Thêm người dùng</button>} /><ErrorNotice message={error} />{user.role !== 'quan_tri' && <div className="notice notice-warning"><ShieldCheck size={16} /> Chỉ quản trị hệ thống mới được thêm hoặc khoá người dùng.</div>}{show && <form className="panel profile-form" onSubmit={create}><div className="form-grid"><label>Tên đăng nhập<input required value={form.username} onChange={e => setForm({ ...form, username: e.target.value })} /></label><label>Họ và tên<input required value={form.full_name} onChange={e => setForm({ ...form, full_name: e.target.value })} /></label><label>Vai trò<select value={form.role} onChange={e => setForm({ ...form, role: e.target.value })}>{Object.entries(ROLE_LABELS).map(([key, label]) => <option key={key} value={key}>{label}</option>)}</select></label><label>Mật khẩu<input required minLength="6" type="password" value={form.password} onChange={e => setForm({ ...form, password: e.target.value })} /></label></div><div className="form-actions"><button type="button" className="btn btn-ghost" onClick={() => setShow(false)}>Huỷ</button><button className="btn btn-primary">Tạo người dùng</button></div></form>}{loading ? <LoadingState /> : <section className="panel"><div className="panel-title"><div><span className="section-kicker">TÀI KHOẢN</span><h2>{users.length} người dùng</h2></div><button className="icon-button" onClick={load}><RefreshCw size={15} /></button></div><div className="table-wrap"><table className="data-table"><thead><tr><th>Tài khoản</th><th>Họ tên</th><th>Vai trò</th><th>Trạng thái</th><th /></tr></thead><tbody>{users.map(item => <tr key={item.id}><td><b>{item.username}</b></td><td>{item.full_name}</td><td>{ROLE_LABELS[item.role] || item.role}</td><td><span className={`status-pill ${item.active ? 'active-status' : 'inactive-status'}`}>{item.active ? 'Đang hoạt động' : 'Đã khoá'}</span></td><td>{user.role === 'quan_tri' && item.id !== user.id && <button className="table-action" onClick={() => toggle(item)}>{item.active ? 'Khoá' : 'Mở khoá'}</button>}</td></tr>)}</tbody></table></div></section>}</main>;
}

function SettingsModal({ settings, onSave, onClose }) {
  const [draft, setDraft] = useState(settings);
  const [testing, setTesting] = useState(false);
  const [state, setState] = useState('idle');
  const test = async () => {
    setTesting(true); setState('idle');
    try {
      const r = await fetch('/health'); const health = await r.json(); if (!health.model_loaded) throw new Error('Model chưa nạp');
      if (!r.ok) throw new Error('bad'); setState('ok');
    } catch { setState('fail'); } finally { setTesting(false); }
  };
  return <div className="modal-backdrop" onMouseDown={onClose}>
    <div className="settings-modal" onMouseDown={e => e.stopPropagation()}>
      <div className="modal-head"><div><h3>Cấu hình model local</h3><p>Trạng thái dịch vụ HRM và MinerU trên máy này.</p></div><button onClick={onClose}><X size={20} /></button></div>
      <label>Endpoint API</label><input readOnly value={draft.endpoint} onChange={e => setDraft({ ...draft, endpoint: e.target.value })} />
      <label>Tên model</label><input readOnly value={draft.model} onChange={e => setDraft({ ...draft, model: e.target.value })} />
      <label>Luồng xử lý</label><textarea readOnly rows="4" value={draft.systemPrompt} onChange={e => setDraft({ ...draft, systemPrompt: e.target.value })} />
      <button className="test-button" onClick={test} disabled={testing}><RefreshCw size={16} className={testing ? 'spin' : ''} />{testing ? 'Đang kiểm tra...' : 'Kiểm tra kết nối'}</button>
      {state === 'ok' && <div className="connection ok"><Wifi size={16} /> Kết nối thành công.</div>}
      {state === 'fail' && <div className="connection fail"><WifiOff size={16} /> Không kết nối được. Kiểm tra tiến trình HRM; hệ thống không tạo câu trả lời thay thế.</div>}
      <div className="modal-actions"><button onClick={onClose}>Huỷ</button><button className="primary" onClick={() => onSave(draft)}>Đóng</button></div>
    </div>
  </div>;
}

export function MessageContent({ message }) {
  if (message.role !== 'assistant') return <div className="bubble">{message.text}</div>;
  const display = answerDisplay(message.text, message.source);
  if (display.comparison) return <ComparisonAnswer display={display} />;
  return <>
    <div className="bubble">{display.text}</div>
    {display.sourceLabel && <div className="answer-source">Nguồn: {display.sourceLabel}.</div>}
    {display.details && <details className="answer-verification"><summary>Chi tiết xác minh</summary><div>{display.details}</div></details>}
  </>;
}

function AssistantPage({ settings, token, onSettings }) {
  const [resetContext, setResetContext] = useState(false);
  const [mobilePanel, setMobilePanel] = useState(null);
  const [docs, setDocs] = useState([]);
  const [chats, setChats] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [connection, setConnection] = useState('idle');
  const [uploading, setUploading] = useState(false);
  const scrollRef = useRef(null);
  const fileRef = useRef(null);
  const chat = chats.find(c => c.id === activeId) || { messages: [] };
  const selected = docs.filter(d => d.active);
  useEffect(() => { if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight; }, [chat?.messages.length, loading]);

  const conversationRequest = async (path, options = {}) => {
    const response = await fetch(`${API_BASE}${path}`, {
      ...options,
      headers: { Authorization: `Bearer ${token}`, ...(options.headers || {}) },
    });
    if (!response.ok) throw new Error('Không tải được dữ liệu hội thoại.');
    return response.json();
  };

  const loadConversations = async () => {
    const summaries = await conversationRequest('/conversations');
    const data = await Promise.all(summaries.map(summary => conversationRequest(`/conversations/${summary.id}`)));
    const mapped = data.map(conversation => ({
      id: conversation.id,
      title: conversation.title,
      time: new Date(conversation.updated_at).toLocaleDateString('vi-VN'),
      messages: conversation.messages.map(message => ({
        id: message.id, role: message.role,
        text: message.text?.startsWith('Chưa thể trả lời từ tài liệu thật:') ? 'Chưa trả lời được từ tài liệu đã chọn.' : message.text,
        source: message.text?.startsWith('Chưa thể trả lời từ tài liệu thật:') ? null : message.source,
        created_at: message.created_at,
      })),
    }));
    if (mapped.length === 0) {
      const created = await conversationRequest('/conversations', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({}) });
      mapped.push({ id: created.id, title: created.title, time: 'Vừa xong', messages: [] });
    }
    setChats(mapped);
    setActiveId(current => mapped.some(conversation => conversation.id === current) ? current : mapped[0].id);
  };

  const loadDocuments = async () => {
    const response = await fetch(`${API_BASE}/documents`, { headers: { Authorization: `Bearer ${token}` } });
    if (!response.ok) throw new Error('Không tải được danh sách tài liệu.');
    const data = await response.json();
    setDocs(previous => {
      const selectedIds = new Set(previous.filter(document => document.active).map(document => document.id));
      return data.map(document => ({
        id: document.id,
        name: document.filename,
        size: 'PDF',
        date: new Date(document.uploaded_at).toLocaleDateString('vi-VN'),
        status: document.status,
        active: selectedIds.has(document.id),
      }));
    });
  };

  useEffect(() => {
    loadConversations().catch(e => setConnection(e.message));
    loadDocuments().catch(() => setConnection('unavailable'));
    fetch('/health').then(r => r.json()).then(h => setConnection(h.model_loaded ? 'live' : 'unavailable')).catch(() => setConnection('unavailable'));
    const timer = setInterval(() => loadDocuments().catch(() => {}), 5000);
    return () => clearInterval(timer);
  }, [token]);

  const uploadDocument = async event => {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file) return;
    setUploading(true);
    try {
      const form = new FormData();
      form.append('file', file);
      const response = await fetch(`${API_BASE}/documents/upload`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` },
        body: form,
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Upload thất bại.');
      await loadDocuments();
    } catch (error) {
      setConnection(error.message);
    } finally {
      setUploading(false);
    }
  };

  const toggleDoc = id => {
    if (loading) return;
    setResetContext(true);
    setDocs(prev => {
    const current = prev.find(d => d.id === id);
    if (!current.active && prev.filter(d => d.active).length >= 2) return prev;
    return prev.map(d => d.id === id ? { ...d, active: !d.active } : d);
    });
  };

  const askModel = async q => {
    try {
      const response = await fetch(`${API_BASE}/rag/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({ question: q, document_ids: selected.map(document => document.id), conversation_id: activeId, reset_context: resetContext, model: settings.model }),
      });
      const raw = await response.text();
      let data = null;
      try { data = raw ? JSON.parse(raw) : null; } catch { /* Proxy có thể trả lỗi dạng text/HTML. */ }
      if (!response.ok || !data?.answer) throw new Error(data?.detail || 'HRM chưa sẵn sàng.');
      setConnection('live');
      const sourceLabel = items => items.map(source => `${source.filename || ''}${source.page_number ? ` · trang ${source.page_number}` : ''}`).join('; ');
      const publishedSource = sourceLabel(data.sources || []);
      const diagnosticSource = sourceLabel(data.diagnostic_sources || []);
      return { text: data.answer, source: publishedSource || (diagnosticSource ? 'Ứng viên chưa xác minh: ' + diagnosticSource : '') };
    } catch (error) {
      setConnection('unavailable');
      return { text: error.message || 'Chưa trả lời được từ tài liệu đã chọn.', source: null };
    }
  };

  const send = async () => {
    const q = text.trim(); if (!q || loading || !selected.length || !activeId) return;
    setText(''); setLoading(true);
    const appendMessage = message => setChats(prev => prev.map(c => c.id === activeId ? { ...c, messages: [...c.messages, message] } : c));
    try {
      const userMessage = await conversationRequest(`/conversations/${activeId}/messages`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ role: 'user', text: q }) });
      appendMessage(userMessage);
      const ans = await askModel(q);
      setResetContext(false);
      const assistantMessage = await conversationRequest(`/conversations/${activeId}/messages`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ role: 'assistant', text: ans.text, source: ans.source }) });
      appendMessage(assistantMessage);
    } catch {
      appendMessage({ id: `local-${Date.now()}`, role: 'assistant', text: 'Chưa trả lời được từ tài liệu đã chọn.', source: null, created_at: new Date().toISOString() });
    } finally { setLoading(false); }
  };

  const newChat = async () => {
    const conversation = await conversationRequest('/conversations', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({}) });
    setChats(prev => [{ id: conversation.id, title: conversation.title, time: 'Vừa xong', messages: [] }, ...prev]);
    setActiveId(conversation.id);
  };

  return <div className="assistant-page">
    <div className="assistant-tools"><button onClick={() => setMobilePanel(mobilePanel === 'history' ? null : 'history')}>Lịch sử</button><button onClick={() => setMobilePanel(mobilePanel === 'docs' ? null : 'docs')}>Tài liệu ({selected.length}/2)</button></div>
    <aside className={`history-panel ${mobilePanel === 'history' ? 'mobile-open' : ''}`}>

      <div className="section-top"><div><span>LỊCH SỬ</span><strong>Hội thoại</strong></div><button disabled={loading} onClick={() => newChat().catch(e => setConnection(e.message))}><Plus size={18} /></button></div>
      <div className="history-list">{chats.map(c => <button key={c.id} className={c.id === activeId ? 'active' : ''} disabled={loading} onClick={() => setActiveId(c.id)}><MessageSquareText size={17} /><div><strong>{c.title}</strong><span>{c.time}</span></div></button>)}</div>
      <div className="sim-note"><i /> Hội thoại được lưu trên máy này</div>
    </aside>
    <section className="chat-panel">
      <header className="chat-head">
        <div className="bot-badge"><Bot size={20} /></div>
        <div><strong>FinHRM Assistant</strong><span><i className={connection === 'live' ? 'live' : ''} /> {connection === 'live' ? 'Đã kết nối model local' : 'Chưa xác nhận kết nối HRM'}</span></div>
        <button className="settings-mini" onClick={onSettings}><Settings size={18} /></button>
      </header>
      <div className="messages" ref={scrollRef}>
        {chat.messages.length === 0 && <div className="empty-chat"><Sparkles size={26} /><strong>Bắt đầu cuộc hội thoại mới</strong><span>Upload PDF, chọn tài liệu đã index rồi đặt câu hỏi.</span></div>}
        {chat.messages.map(m => <div className={`message-row ${m.role}`} key={m.id}>
          {m.role === 'user' ? <div className="avatar user-avatar">A</div> : <div className="avatar bot-avatar"><Bot size={15} /></div>}
          <div className="message-wrap"><div className="message-meta">{m.role === 'user' ? 'Bạn' : 'Trợ lý phân tích'} <span>{formatChatTime(m.created_at)}</span></div><MessageContent message={m} /></div>
        </div>)}
        {loading && <div className="typing"><RefreshCw size={15} className="spin" /> Đang phân tích tài liệu...</div>}
      </div>
      <div className="composer">
        {selected.length > 0 && <div className="composer-files">{selected.map(d => <span key={d.id}><FileText size={13} />{d.name}</span>)}</div>}
        <div className="composer-box"><textarea rows="2" value={text} onChange={e => setText(e.target.value.slice(0, 1000))} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } }} placeholder="Đặt câu hỏi về tài liệu đã chọn..." /><button onClick={send} disabled={!text.trim() || loading || !selected.length || !activeId}><Send size={18} /></button></div>
        <div className="composer-hint"><span>Enter để gửi · Shift + Enter để xuống dòng</span><span>{text.length}/1000</span></div>
      </div>
    </section>
    <aside className={`context-panel ${mobilePanel === 'docs' ? 'mobile-open' : ''}`}>
      <div className="context-head"><div><span>NGỮ CẢNH</span><strong>Tài liệu đã phân tích</strong></div><b>{selected.length}/2</b></div>
      <p>Chọn từ 1 đến 2 tài liệu làm nguồn cho mỗi câu hỏi.</p>
      <input ref={fileRef} type="file" accept="application/pdf" hidden onChange={uploadDocument} />
      <button className="upload-button" onClick={() => fileRef.current?.click()} disabled={uploading}><Plus size={16} />{uploading ? 'Đang tải lên...' : 'Upload PDF'}</button>
      <div className="doc-list">{docs.map(d => <button key={d.id} className={d.active ? 'active' : ''} disabled={loading || d.status !== 'indexed'} onClick={() => toggleDoc(d.id)}><span className="checkbox">{d.active && <Check size={12} />}</span><FileText size={18} /><div><strong>{d.name}</strong><span>{d.status === 'indexed' ? `${d.size} · ${d.date}` : `Đang xử lý: ${d.status}`}</span></div></button>)}</div>
      <div className="context-note"><ShieldCheck size={18} /><span>Chỉ dùng tài liệu được chọn. Phản hồi không tự suy diễn ngoài nguồn.</span></div>
    </aside>
  </div>;
}

function Shell({ session, onLogout }) {
  const [page, setPage] = useState('overview');
  const [settings, setSettings] = useState(initialSettings);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [selectedHoSoId, setSelectedHoSoId] = useState(null);
  const user = session.user || {};
  const label = page === 'admin' ? 'Quản lý' : navItems.find(n => n.id === page)?.label || '';
  return <div className="app-shell">
    <header className="topbar">
      <div className="topbar-brand"><Brand inverse compact /></div>
      <div className="breadcrumb"><ChevronLeft size={18} /><span>Không gian làm việc</span><b>/</b><strong>{label}</strong></div>
      <div className="user-area"><div className="user-badge">{(user.full_name || user.username || 'U').slice(0, 1).toUpperCase()}</div><div><strong>{user.full_name || user.username}</strong><span>{ROLE_LABELS[user.role] || user.role}</span></div><button onClick={onLogout}><LogOut size={17} />Đăng xuất</button></div>
    </header>
    <aside className="sidebar">
      <div className="nav-group"><span>KHÔNG GIAN NGHIỆP VỤ</span>{navItems.map(item => { const Icon = item.icon; return <button key={item.id} className={page === item.id ? 'active' : ''} onClick={() => setPage(item.id)}><Icon size={20} />{item.label}</button> })}</div>
      <div className="nav-divider" />
      <div className="nav-group"><span>HỆ THỐNG</span><button className={page === 'admin' ? 'active' : ''} onClick={() => setPage('admin')}><Settings size={20} />Quản lý</button></div>
      <div className="system-status"><i /><div><strong>Không gian làm việc local</strong><span>FinHRM · MinerU · Policy B</span></div></div>
    </aside>
    <div className="main-area">
      {page === 'overview' && <OverviewPage token={session.access_token} onNavigate={setPage} onSelectHoSo={setSelectedHoSoId} />}
      {page === 'profiles' && <HoSoPage token={session.access_token} selectedHoSoId={selectedHoSoId} onSelectHoSo={setSelectedHoSoId} onNavigate={setPage} />}
      {page === 'ocr' && <DocumentsPage token={session.access_token} selectedHoSoId={selectedHoSoId} onSelectHoSo={setSelectedHoSoId} onNavigate={setPage} />}
      {page === 'analysis' && <AnalysisPage token={session.access_token} selectedHoSoId={selectedHoSoId} onSelectHoSo={setSelectedHoSoId} />}
      {page === 'reports' && <ReportsPage token={session.access_token} onSelectHoSo={setSelectedHoSoId} onNavigate={setPage} />}
      {page === 'assistant' && <AssistantPage settings={settings} token={session.access_token} onSettings={() => setSettingsOpen(true)} />}
      {page === 'approval' && <ApprovalPage token={session.access_token} user={user} />}
      {page === 'admin' && <UserManagementPage token={session.access_token} user={user} />}
    </div>
    {settingsOpen && <SettingsModal settings={settings} onClose={() => setSettingsOpen(false)} onSave={s => { setSettings(s); setSettingsOpen(false) }} />}
  </div>;
}

export default function App() {
  const [session, setSession] = useState(null);
  return session ? <Shell session={session} onLogout={() => setSession(null)} /> : <Login onLogin={setSession} />;
}
