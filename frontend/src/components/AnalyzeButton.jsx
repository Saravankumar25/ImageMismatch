export default function AnalyzeButton({ loading, disabled }) {
  return (
    <button
      type="submit"
      className={`btn btn-primary btn-analyze${loading ? ' is-loading' : ''}`}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
    >
      {loading ? (
        <>
          <span className="spinner" aria-hidden="true" />
          Analyzing…
        </>
      ) : (
        <>
          Analyze
          <svg className="btn-arrow" viewBox="0 0 16 16" aria-hidden="true">
            <path d="M3 8h9.5M8.5 4l4 4-4 4" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </>
      )}
    </button>
  );
}
