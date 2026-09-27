import test from 'node:test';
import assert from 'node:assert/strict';
import { answerDisplay } from './answerDisplay.js';

test('user example: unchanged number, one page citation, collapsed assumptions', () => {
  const raw = 'Đang hiểu câu hỏi là số thực hiện. Doanh thu thuần năm 2024 là 1.172.121.547.172 đồng. Nguồn: d416d19e0a-AAH_24CN_BCTN.md, trang 17.\nĐơn vị mặc định là đồng vì bằng chứng được dùng không ghi đơn vị.';
  const result = answerDisplay(raw, 'AAH_24CN_BCTN.pdf · trang 17');
  assert.equal(result.text, 'Doanh thu thuần năm 2024 là 1.172.121.547.172 đồng.');
  assert.equal(result.sourceLabel, 'trang 17');
  assert.match(result.details, /Đơn vị mặc định/);
  assert.match(result.details, /số thực hiện/);
});
test('refusal is retained', () => {
  const raw = 'Đang hiểu câu hỏi là số thực hiện. Chưa xác nhận được đơn vị và thang đo của số liệu từ nguồn; chưa thể quy đổi để trả lời.';
  assert.equal(answerDisplay(raw).text, raw.replace('Đang hiểu câu hỏi là số thực hiện. ', ''));
});
test('unfamiliar warnings and multiple documents are retained', () => {
  const raw = 'Báo cáo A\nDoanh thu 1 đồng. Nguồn: a.md, trang 1.\nBáo cáo B\nChưa xác minh đủ dữ liệu.';
  assert.equal(answerDisplay(raw, 'a.pdf · trang 1; b.pdf · trang 2').text, raw);
});

import { readFileSync } from 'node:fs';
const responses = JSON.parse(readFileSync(new URL('../../data/test/suites/revenue_context/reports/finhrm_B_revenue_context_http_v1.json', import.meta.url), 'utf8')).rows;

test('recorded AAH response: result first, exact strings, one source and unit note', () => {
  const raw = responses.find(row => row.case_id === 'aah-first').response.rendered;
  const display = answerDisplay(raw);
  assert.equal(display.text, 'Doanh thu thuần năm 2024 tăng 939.755.158.742 đồng so với năm 2023.');
  assert.deepEqual(display.comparison.rows, [
    { period: '2023', value: '232.366.388.430' },
    { period: '2024', value: '1.172.121.547.172' },
  ]);
  assert.equal(display.comparison.metric, 'Doanh thu thuần');
  assert.equal(display.comparison.unit, 'đồng');
  assert.equal(display.sourceLabel, 'AAH_24CN_BCTN · trang 17');
  assert.match(display.comparison.note, /đồng mặc định/);
  assert.match(display.details, /Công thức: \(1\.172\.121\.547\.172 đồng - 232\.366\.388\.430 đồng\)/);
  assert.match(display.details, /d416d19e0a-AAH_24CN_BCTN.md/);
  assert.equal(raw, responses[0].response.rendered);
});

test('all six recorded complete comparisons use the compact view', () => {
  for (const row of responses.filter(row => row.response.status === 'ACCEPTED')) {
    const display = answerDisplay(row.response.rendered);
    assert.ok(display.comparison, row.case_id);
    for (const item of display.comparison.rows) assert.ok(row.response.rendered.includes(item.value));
    assert.ok(row.response.rendered.includes(display.text));
  }
});

test('partial comparison retains the verified year and refusal instead of a complete table', () => {
  const raw = responses.find(row => row.case_id === 'aah-missing').response.rendered;
  const display = answerDisplay(raw);
  assert.equal(display.comparison, undefined);
  assert.match(display.text, /2030/);
  assert.match(display.text, /1\.172\.121\.547\.172/);
  assert.equal(display.text, raw); // Keep the complete missing-year response.
});

test('unknown warnings, changed metric, and incomplete source are not silently discarded', () => {
  const raw = responses[0].response.rendered;
  for (const changed of [raw + '\nCó điều chỉnh hồi tố chưa xác nhận.',
    raw.replace('Số liệu thứ hai:\nDoanh thu thuần', 'Số liệu thứ hai:\nTổng doanh thu'),
    raw.replace('trang 17.', 'không xác định trang.'),
    raw.replace('Đối chiếu từng kỳ/loại số liệu và chênh lệch.', 'Chưa xác minh đủ các mục.')]) {
    assert.equal(answerDisplay(changed).comparison, undefined);
  }
});

test('withheld candidate citation is diagnostic, including old stored messages', () => {
  for (const source of ['AAH.pdf · trang 24', 'Ứng viên chưa xác minh: AAH.pdf · trang 24']) {
    const display = answerDisplay('Không đủ bằng chứng đã xác minh để trả lời.', source);
    assert.equal(display.sourceLabel, '');
    assert.match(display.details, /Ứng viên chưa xác minh: AAH.pdf · trang 24/);
  }
});

test('source-bound field answer keeps its public citation', () => {
  const display = answerDisplay('Theo thông tin được nêu trong báo cáo, vốn điều lệ là 1.179.000.000.000 đồng. Nguồn: AAH.md, trang 1.');
  assert.equal(display.sourceLabel, 'trang 1');
  assert.match(display.text, /1\.179\.000\.000\.000 đồng/);
  assert.doesNotMatch(display.details, /Ứng viên/);
});


test('verified growth keeps one source and moves known formula into details', () => {
  const raw = 'Lợi nhuận sau thuế năm 2032 tăng khoảng 50% so với năm 2031. Nguồn: REPORT, trang 2.\nĐơn vị mặc định là đồng vì bằng chứng được dùng không ghi đơn vị.\nCông thức: (1.800 − 1.200) / 1.200 × 100 = 50%.';
  const result = answerDisplay(raw, 'REPORT.pdf · trang 2');
  assert.equal(result.text, 'Lợi nhuận sau thuế năm 2032 tăng khoảng 50% so với năm 2031.');
  assert.equal(result.sourceLabel, 'trang 2');
  assert.match(result.details, /1.800 − 1.200/);
  assert.match(result.details, /đơn vị/i);
});
