import datetime, json, os, math
import pandas as pd

def render_report(design_csv, cfg_dict, fit_summary_txt=None, metrics=None, out_html="outputs/report.html"):
    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    d0 = pd.read_csv(design_csv, nrows=2000)  # preview
    cols = d0.columns.tolist()
    nrows = sum(1 for _ in open(design_csv)) - 1
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # small HTML
    html = f"""
<!doctype html><html><head><meta charset="utf-8">
<title>Shark iSSA Report</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:980px;margin:2rem auto;line-height:1.5}}
code,pre{{background:#f6f8fa;padding:.2rem .4rem;border-radius:.3rem}}
table{{border-collapse:collapse;width:100%}}
th,td{{border:1px solid #ddd;padding:6px}}
th{{background:#f0f0f0}}
.kv td:first-child{{font-weight:600;width:28%}}
</style></head><body>
<h1>Shark iSSA Analysis Report</h1>
<p><em>Generated {now}</em></p>

<h2>Run configuration</h2>
<table class="kv">
<tr><td>Δt (min)</td><td>{cfg_dict['regularization']['dt_minutes']}</td></tr>
<tr><td>K available</td><td>{cfg_dict['steps']['K_available']}</td></tr>
<tr><td>Social radius (m)</td><td>{cfg_dict['social']['radius_m']}</td></tr>
<tr><td>Cone half-angle (deg)</td><td>{cfg_dict['social']['cone_half_angle_deg']}</td></tr>
<tr><td>Formula</td><td><code>{" + ".join(cfg_dict["fit"]["formula"])}</code></td></tr>
</table>

<h2>Design matrix</h2>
<p>File: <code>{design_csv}</code><br>Rows: {nrows:,}<br>Columns: {len(cols)}<br><small>{", ".join(cols[:20])} …</small></p>

<h2>Fit summary</h2>
<pre>{fit_summary_txt or "Model fit not available (exported for R)."}
</pre>

<h2>Model metrics</h2>
<table class="kv">
<tr><td>AIC</td><td>{(metrics or {}).get("AIC","—")}</td></tr>
<tr><td>BIC</td><td>{(metrics or {}).get("BIC","—")}</td></tr>
<tr><td>logLik</td><td>{(metrics or {}).get("ll","—")}</td></tr>
<tr><td>n (rows)</td><td>{(metrics or {}).get("n","—")}</td></tr>
<tr><td>df (k)</td><td>{(metrics or {}).get("k","—")}</td></tr>
<tr><td>CV mean rank (used)</td><td>{(metrics or {}).get("rank_mean","—")}</td></tr>
<tr><td>CV % top-1</td><td>{(metrics or {}).get("rank_top1","—")}</td></tr>
<tr><td>CV mean log score</td><td>{(metrics or {}).get("log_score","—")}</td></tr>
</table>

<h2>Social behavior interpretation</h2>
<ul>
<li><b>Following (iSSA-compliant)</b>: positive β for <code>ahead_any</code>, <code>n_forward</code>, or <code>mean_align_fwd</code> implies selection for endpoints with conspecifics ahead/aligned at time t.</li>
<li><b>Leading (iSSA-approx)</b>: positive β for <code>behind_any</code> or <code>n_behind</code> implies the focal tends to be ahead of others at time t.</li>
<li><b>Future-leading (post-hoc)</b>: descriptive metric <code>lead_future</code> = neighbors appear behind at t+Δ near the focal’s endpoint (not used in the likelihood).</li>
</ul>

<p><small>This report keeps the iSSA likelihood free of future information (no leakage). Any future-based lead metrics are shown descriptively only.</small></p>

</body></html>
"""
    with open(out_html, "w") as f:
        f.write(html)
    return out_html
