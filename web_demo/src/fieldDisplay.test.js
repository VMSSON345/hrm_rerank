import test from 'node:test';
import assert from 'node:assert/strict';
import {answerDisplay} from './answerDisplay.js';
test('two source fields retain identifiers and show a single shared citation',()=>{
 const r=answerDisplay('Điện thoại: 000 123 Nguồn: NEW, trang 1.\n\nWebsite: example.test Nguồn: NEW, trang 1.');
 assert.equal(r.text,'Điện thoại: 000 123\n\nWebsite: example.test');assert.equal(r.sourceLabel,'NEW · trang 1');
});
test('partial fields and unknown warning are not stripped',()=>{
 const s='Website: example.test Nguồn: NEW, trang 1.\n\nChưa xác nhận được email.';
 assert.equal(answerDisplay(s).text,s);
});
test('separate pages and documents remain visible',()=>{
 const r=answerDisplay('Điện thoại: 000 123 Nguồn: NEW, trang 1.\n\nWebsite: example.test Nguồn: OTHER, trang 4.');
 assert.match(r.sourceLabel,/NEW · trang 1; OTHER · trang 4/);
});
