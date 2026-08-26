#!/usr/bin/env bash
set -u

MODEL_PATH="${MODEL_PATH:-models/document_hrm_V8_candidate.pt}"
REPORT_PATH="${REPORT_PATH:-reports/document_hrm_external_pdf_v1.txt}"
CASE_FILTER="${CASE_FILTER:-}"
MINERU_OUTPUT_ROOT="${MINERU_OUTPUT_ROOT:-reports/mineru_external_v1}"
DOCUMENT_SERVICE_URL="${DOCUMENT_SERVICE_URL:-}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "venv/bin/python" ]]; then
    PYTHON_BIN="venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

mkdir -p "$(dirname "$REPORT_PATH")"
: > "$REPORT_PATH"

run_case() {
  local case_id="$1"
  local document="$2"
  local expected_operation="$3"
  local expected_value="$4"
  local expected_pages="$5"
  local question="$6"
  local ticker="${case_id%%-*}"
  local page_offset=0
  case "$ticker" in
    ABI) page_offset=24 ;;
    ABR) page_offset=32 ;;
    ABS) page_offset=30 ;;
    *)
      echo "Unknown MinerU smoke ticker: $ticker" >&2
      return 2
      ;;
  esac
  local -a ocr_args=(
    --ocr
    --force-ocr
    --mineru-output-dir "$MINERU_OUTPUT_ROOT/$ticker"
    --mineru-page-offset "$page_offset"
  )

  if [[ -n "$CASE_FILTER" && ! "$case_id" =~ $CASE_FILTER ]]; then
    return
  fi

  {
    echo "================================================================================"
    echo "CASE: $case_id"
    echo "DOCUMENT: $document"
    echo "QUESTION: $question"
    echo "GOLD_OPERATION: $expected_operation"
    echo "GOLD_VALUE: $expected_value"
    echo "GOLD_PDF_PAGE: $expected_pages"
    echo "--------------------------------------------------------------------------------"
  } | tee -a "$REPORT_PATH"

  local command_status
  if [[ -n "$DOCUMENT_SERVICE_URL" ]]; then
    "$PYTHON_BIN" scripts/request_document_service.py \
      "$document" \
      "$question" \
      --base-url "$DOCUMENT_SERVICE_URL" \
      --top-k-chunks 12 \
      --neighbor-chunks 2 2>&1 | tee -a "$REPORT_PATH"
    command_status=${PIPESTATUS[0]}
  else
    "$PYTHON_BIN" main.py \
      --document-model-path "$MODEL_PATH" \
      answer-document "$question" \
      --document "$document" \
      "${ocr_args[@]}" \
      --top-k-chunks 12 \
      --neighbor-chunks 2 2>&1 | tee -a "$REPORT_PATH"
    command_status=${PIPESTATUS[0]}
  fi
  echo "COMMAND_EXIT: $command_status" | tee -a "$REPORT_PATH"
  echo | tee -a "$REPORT_PATH"
}

run_case \
  "ABI-SELECT-001" \
  "pdf/ABI_25CN_BCTN.pdf" \
  "SELECT" \
  "4771980517070" \
  "25" \
  "Theo bảng cân đối kế toán, tổng cộng tài sản của ABI tại ngày 31/12/2025 là bao nhiêu VND?"

run_case \
  "ABI-SUBTRACT-001" \
  "pdf/ABI_25CN_BCTN.pdf" \
  "SUBTRACT" \
  "49590826575" \
  "26" \
  "Theo báo cáo kết quả hoạt động kinh doanh, lợi nhuận sau thuế của ABI năm 2025 tăng bao nhiêu VND so với năm 2024?"

run_case \
  "ABI-RATIO-001" \
  "pdf/ABI_25CN_BCTN.pdf" \
  "RATIO_PERCENT" \
  "63.2092987480601" \
  "25" \
  "Theo bảng cân đối kế toán, nợ phải trả chiếm bao nhiêu phần trăm tổng cộng tài sản của ABI tại ngày 31/12/2025?"

run_case \
  "ABI-PERCENT-CHANGE-001" \
  "pdf/ABI_25CN_BCTN.pdf" \
  "PERCENT_CHANGE" \
  "13.9033531319969" \
  "26" \
  "Theo báo cáo kết quả hoạt động kinh doanh, doanh thu thuần hoạt động kinh doanh bảo hiểm của ABI thay đổi bao nhiêu phần trăm từ năm 2024 đến năm 2025?"

run_case \
  "ABR-SELECT-001" \
  "pdf/ABR_25CN_BCTN.pdf" \
  "SELECT" \
  "284331609643" \
  "33" \
  "Theo bảng cân đối kế toán, tổng cộng tài sản của ABR cuối năm 2025 là bao nhiêu VND?"

run_case \
  "ABR-SUBTRACT-001" \
  "pdf/ABR_25CN_BCTN.pdf" \
  "SUBTRACT" \
  "70449815400" \
  "33" \
  "Theo bảng cân đối kế toán, tổng cộng tài sản của ABR cuối năm 2025 giảm bao nhiêu VND so với đầu năm 2025?"

run_case \
  "ABR-RATIO-001" \
  "pdf/ABR_25CN_BCTN.pdf" \
  "RATIO_PERCENT" \
  "18.8351384867977" \
  "33,34" \
  "Theo bảng cân đối kế toán, nợ phải trả chiếm bao nhiêu phần trăm tổng cộng tài sản của ABR cuối năm 2025?"

run_case \
  "ABR-PERCENT-CHANGE-001" \
  "pdf/ABR_25CN_BCTN.pdf" \
  "PERCENT_CHANGE" \
  "3.03463290552239" \
  "35" \
  "Theo báo cáo kết quả hoạt động kinh doanh, doanh thu bán hàng và cung cấp dịch vụ của ABR thay đổi bao nhiêu phần trăm từ năm 2024 đến năm 2025?"

run_case \
  "ABS-SELECT-001" \
  "pdf/ABS_25CN_BCTN.pdf" \
  "SELECT" \
  "173171512356" \
  "36" \
  "Theo báo cáo kết quả hoạt động, lợi nhuận kế toán sau thuế thu nhập doanh nghiệp của ABS năm 2025 là bao nhiêu VND?"

run_case \
  "ABS-SUBTRACT-001" \
  "pdf/ABS_25CN_BCTN.pdf" \
  "SUBTRACT" \
  "1323989362641" \
  "31" \
  "Theo báo cáo tình hình tài chính, tổng cộng tài sản của ABS cuối năm 2025 tăng bao nhiêu VND so với đầu năm 2025?"

run_case \
  "ABS-RATIO-001" \
  "pdf/ABS_25CN_BCTN.pdf" \
  "RATIO_PERCENT" \
  "66.1381040526222" \
  "31,32" \
  "Theo báo cáo tình hình tài chính, nợ phải trả chiếm bao nhiêu phần trăm tổng cộng tài sản của ABS tại ngày 31/12/2025?"

run_case \
  "ABS-PERCENT-CHANGE-001" \
  "pdf/ABS_25CN_BCTN.pdf" \
  "PERCENT_CHANGE" \
  "41.5246991934066" \
  "35" \
  "Theo báo cáo kết quả hoạt động, cộng doanh thu hoạt động của ABS thay đổi bao nhiêu phần trăm từ năm 2024 đến năm 2025?"

echo "Saved report: $REPORT_PATH"
"$PYTHON_BIN" scripts/score_external_pdf_smoke.py "$REPORT_PATH"
