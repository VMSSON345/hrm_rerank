export default function ComparisonAnswer({ display }) {
  const { comparison } = display;
  return <>
    <div className="bubble comparison-answer">
      <p className="comparison-summary">{display.text}</p>
      <div className="comparison-table-wrap">
        <table className="comparison-table" aria-label={`So sánh ${comparison.metric.toLowerCase()}`}>
          <thead><tr><th scope="col">{comparison.periodHeading}</th><th scope="col">{comparison.metric}<span className="comparison-unit"> ({comparison.unit})</span></th></tr></thead>
          <tbody>{comparison.rows.map((row, index) => <tr key={index}><th scope="row">{row.period}</th><td>{row.value}</td></tr>)}</tbody>
        </table>
      </div>
      <div className="comparison-source">Nguồn: {display.sourceLabel}.</div>
      {comparison.note && <p className="comparison-note">{comparison.note}</p>}
    </div>
    <details className="answer-verification"><summary>Công thức và chi tiết xác minh</summary><div>{display.details}</div></details>
  </>;
}
