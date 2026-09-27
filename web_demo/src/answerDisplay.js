import { fieldDisplay } from './fieldDisplay.js';
import { comparisonDisplay } from './comparisonDisplay.js';
// Presentation only. The persisted public HRM response remains unchanged.
const ACTUAL_NOTE = 'Đang hiểu câu hỏi là số thực hiện. ';
const UNIT_NOTE = 'Đơn vị mặc định là đồng vì bằng chứng được dùng không ghi đơn vị.';

export function answerDisplay(raw, source = '') {
  const comparison = comparisonDisplay(raw || '');
  if (comparison) return comparison;
  const fields = fieldDisplay(raw);
  if (fields) return fields;
  let text = raw || '';
  const notes = [];
  if (text.startsWith(ACTUAL_NOTE)) {
    notes.push(ACTUAL_NOTE.trim());
    text = text.slice(ACTUAL_NOTE.length);
  }
  if (text.endsWith('\n' + UNIT_NOTE)) {
    notes.push(UNIT_NOTE);
    text = text.slice(0, -('\n' + UNIT_NOTE).length);
  }
  // Move only recognized metadata/formula lines; preserve unfamiliar warnings.
  const lines = text.split('\n');
  const scalarAnswer = !text.includes('Số liệu thứ nhất:');
  text = lines.filter(line => {
    if (scalarAnswer && (line === UNIT_NOTE || /^Đang hiểu “[^"]+” là .+, theo (?:hàng và kỳ trong tài liệu|câu hỏi trước trên cùng tài liệu)\.$/.test(line)
        || /^Công thức: \([−+\d., ]+\) \/ [\d.,]+ × 100 = [−+\d.,-]+%\.$/.test(line))) {
      notes.push(line); return false;
    }
    return true;
  }).join('\n');
  // Compact only the known single-source, single-answer renderer format.
  // Leave summaries, multiple documents and unfamiliar warnings intact.
  const inline = text.match(/\s+Nguồn: ([^\n;]+), trang (\d+(?:, \d+)*)\.$/);
  let sourceLabel = source || '';
  const diagnosticPrefix = 'Ứng viên chưa xác minh: ';
  const legacyRefusal = !text.includes('\n') && /^(?:Không đủ bằng chứng|Chưa (?:xác minh|xác nhận|xác định|đủ bằng chứng|có đủ bằng chứng))/.test(text);
  if (sourceLabel.startsWith(diagnosticPrefix) || (legacyRefusal && sourceLabel)) {
    notes.push(sourceLabel.startsWith(diagnosticPrefix) ? sourceLabel : diagnosticPrefix + sourceLabel);
    sourceLabel = '';
  }
  if (inline && !legacyRefusal && !text.includes('\n') && !sourceLabel.includes(';')) {
    text = text.slice(0, inline.index);
    sourceLabel = 'trang ' + inline[2];
  } else if (!text.includes('Nguồn:') && sourceLabel && !sourceLabel.includes(';')) {
    const page = sourceLabel.match(/(?:·\s*)?trang (\d+(?:, \d+)*)$/);
    if (page) sourceLabel = 'trang ' + page[1];
  }
  return { text, sourceLabel, details: notes.join('\n') };
}
