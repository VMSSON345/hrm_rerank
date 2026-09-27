// Compact only the complete, known comparison renderer. Never infer a value.
const UNIT_NOTE = 'Đơn vị mặc định là đồng vì bằng chứng được dùng không ghi đơn vị.';
const COMPLETE = 'Đối chiếu từng kỳ/loại số liệu và chênh lệch.';
const ACTUAL = 'Đang dùng số thực hiện theo cách hiểu mặc định.';
const CITATION = /Nguồn: (.+?), trang (\d+(?:, \d+)*)\.$/;

function sourceCitation(line) {
  const match = line.match(CITATION);
  return match ? { name: match[1], pages: match[2].split(', ') } : null;
}

function scalar(section) {
  const lines = section.trim().split('\n');
  if (lines.slice(1).some(line => line !== UNIT_NOTE)) return null;
  const match = lines[0].match(/^(.+?) năm ((?:19|20)\d{2}) là ([+−-]?[\d.,]+) (nghìn đồng|triệu đồng|tỷ đồng|đồng|%)\.\s+Nguồn:/);
  const citation = sourceCitation(lines[0]);
  if (!match || !citation) return null;
  const role = match[1].match(/ (kế hoạch|thực hiện)$/)?.[1] || '';
  return { metric: role ? match[1].slice(0, -(role.length + 1)) : match[1],
    year: match[2], value: match[3], unit: match[4], role, citation,
    unitAssumed: lines.includes(UNIT_NOTE) };
}

export function comparisonDisplay(raw) {
  const match = raw.match(/^([\s\S]*?)\n\nSố liệu thứ nhất:\n([\s\S]*?)\n\nSố liệu thứ hai:\n([\s\S]*?)\n\nChênh lệch:\n([\s\S]*)$/);
  if (!match) return null;
  const prefix = match[1].split('\n');
  if (!prefix.includes(COMPLETE) || prefix.some(line => line !== COMPLETE && line !== ACTUAL
      && !/^Đang hiểu “doanh thu” là .+\.$/.test(line))) return null;
  const first = scalar(match[2]); const second = scalar(match[3]);
  if (!first || !second || first.metric !== second.metric || first.unit !== second.unit) return null;
  const difference = match[4].trim().split('\n');
  const summary = difference[0];
  // Keep refusals, incomplete answers and unfamiliar warnings in their original form.
  if (!summary.startsWith(first.metric + ' năm ') || !/[\d]/.test(summary)
      || difference.slice(1).some(line => line !== UNIT_NOTE && !line.startsWith('Công thức: ')
        && !sourceCitation(line))) return null;
  if (!difference.some(line => line.startsWith('Công thức: '))) return null;
  const citations = [first.citation, second.citation, ...difference.map(sourceCitation).filter(Boolean)];
  const bySource = new Map();
  for (const citation of citations) {
    if (!bySource.has(citation.name)) bySource.set(citation.name, new Set());
    for (const page of citation.pages) bySource.get(citation.name).add(page);
  }
  const sourceLabel = [...bySource].map(([name, pages]) => {
    const label = name.replace(/^[a-f\d]{8,16}-/i, '').replace(/\.(?:md|pdf)$/i, '');
    return `${label} · trang ${[...pages].sort((a, b) => Number(a) - Number(b)).join(', ')}`;
  }).join('; ');
  const sameYear = first.year === second.year;
  if (sameYear && (!first.role || !second.role || first.role === second.role)) return null;
  const rows = [first, second].map(item => ({
    period: sameYear ? `${item.role === 'kế hoạch' ? 'Kế hoạch' : 'Thực hiện'} ${item.year}` : item.year,
    value: item.value,
  }));
  const unitAssumed = first.unitAssumed || second.unitAssumed || difference.includes(UNIT_NOTE);
  const actual = prefix.includes(ACTUAL) || (first.role === 'thực hiện' && second.role === 'thực hiện');
  const details = [...new Set([
    ...difference.filter(line => line.startsWith('Công thức: ')),
    ...prefix.filter(line => line !== COMPLETE && line !== ACTUAL),
    ...citations.map(item => `Nguồn gốc: ${item.name}, trang ${item.pages.join(', ')}.`),
  ])].join('\n');
  return { text: summary, sourceLabel, details, comparison: {
    metric: first.metric, unit: first.unit, periodHeading: sameYear ? 'Loại số liệu' : 'Năm', rows,
    note: [actual ? 'Số thực hiện.' : '', unitAssumed ? 'Áp dụng đơn vị đồng mặc định cho số liệu không ghi đơn vị.' : ''].filter(Boolean).join(' '),
  } };
}
