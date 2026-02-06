import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, shapiro
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pingouin as pg
import warnings, io, os
warnings.filterwarnings('ignore')

# ============================================================
# THEME
# ============================================================
st.set_page_config(
    page_title="AI Performance Evaluation Dashboard",
    page_icon="📊", layout="wide", initial_sidebar_state="expanded"
)

C = {
    "bg":      "#0f1117",
    "card":    "#161820",
    "border":  "#2a2d3a",
    "txt":     "#eef0f4",
    "dim":     "#7e8494",
    "gold":    "#e8b86d",
    "teal":    "#4ecdc4",
    "coral":   "#ff6b6b",
    "blue":    "#6c9fe6",
    "purple":  "#a87de8",
    "orange":  "#f0a55a",
}

PLAT_CLR = {
    "HR (Baseline)":     C["gold"],
    "Chat GPT 5.2":      C["teal"],
    "Claude Sonnet 4.5": C["blue"],
    "Deep Seek V3.2":    C["purple"],
    "GLM 4.7":           C["coral"],
    "Gemini 3 Pro":      C["orange"],
}
PLAT_ORDER = list(PLAT_CLR.keys())
VARIABLES  = ["Kemampuan komunikasi", "Pengalaman Kerja", "Keterampilan Teknis"]

plt.style.use('dark_background')

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] {{ font-family:'Sora',sans-serif; background:{C['bg']} !important; color:{C['txt']} !important; }}

.stSidebar {{ background:#131520 !important; border-right:1px solid {C['border']}; }}
.stSidebar h2 {{ color:{C['gold']}; font-size:0.7rem; text-transform:uppercase; letter-spacing:1.4px; margin-bottom:0.4rem; padding-bottom:0.25rem; border-bottom:1px solid {C['border']}; font-weight:600; }}

.main .block-container {{ padding-top:1.6rem; padding-left:1.8rem; padding-right:1.8rem; max-width:1440px; }}

.card {{ background:{C['card']}; border:1px solid {C['border']}; border-radius:12px; padding:1rem 1.2rem; transition:border-color .2s,box-shadow .2s; }}
.card:hover {{ border-color:{C['gold']}44; box-shadow:0 4px 20px rgba(232,184,109,.07); }}
.card .lbl {{ font-size:0.68rem; text-transform:uppercase; letter-spacing:1.1px; color:{C['dim']}; margin-bottom:0.25rem; }}
.card .val {{ font-size:1.55rem; font-weight:700; color:{C['txt']}; line-height:1.15; }}
.card .sub {{ font-size:0.68rem; color:{C['dim']}; margin-top:0.2rem; }}
.badge {{ display:inline-block; font-size:0.58rem; font-weight:600; text-transform:uppercase; letter-spacing:0.7px; padding:0.18rem 0.48rem; border-radius:16px; margin-top:0.3rem; }}
.b-sig {{ background:#4ecdc422; color:#4ecdc4; }}
.b-ns  {{ background:#ff6b6b22; color:#ff6b6b; }}

.sh {{ display:flex; align-items:center; gap:0.6rem; margin-bottom:0.7rem; margin-top:0.5rem; }}
.sh .dot {{ width:9px; height:9px; border-radius:50%; background:{C['gold']}; }}
.sh h3 {{ font-size:0.95rem; font-weight:600; color:{C['txt']}; margin:0; }}
.sh .tag {{ font-size:0.58rem; font-weight:600; text-transform:uppercase; letter-spacing:0.9px; color:{C['gold']}; background:{C['gold']}18; padding:0.18rem 0.45rem; border-radius:4px; }}

.upload-zone {{ border:2px dashed {C['border']}; border-radius:14px; padding:1.6rem; text-align:center; background:{C['card']}; transition:border-color .25s; }}
.upload-zone:hover {{ border-color:{C['gold']}55; }}

.step {{ display:flex; gap:0.75rem; align-items:flex-start; padding:0.6rem 0; border-bottom:1px solid {C['border']}22; }}
.step:last-child {{ border-bottom:none; }}
.step .num {{ min-width:24px; height:24px; border-radius:50%; background:{C['gold']}; color:{C['bg']}; font-size:0.7rem; font-weight:700; display:flex; align-items:center; justify-content:center; }}
.step .txt {{ font-size:0.73rem; color:{C['dim']}; line-height:1.6; }}
.step .txt strong {{ color:{C['txt']}; }}

.rank-row {{ display:flex; align-items:center; gap:0.6rem; padding:0.5rem 0.7rem; border-radius:8px; margin-bottom:0.28rem; background:{C['card']}; border:1px solid {C['border']}; }}
.rank-row .rn {{ font-size:0.95rem; font-weight:700; width:26px; text-align:center; font-family:'JetBrains Mono',monospace; }}
.rank-row .rname {{ font-size:0.77rem; font-weight:500; flex:1; }}
.rank-row .rscore {{ font-size:0.74rem; font-family:'JetBrains Mono',monospace; color:{C['dim']}; }}

::-webkit-scrollbar {{ width:5px; }}
::-webkit-scrollbar-track {{ background:transparent; }}
::-webkit-scrollbar-thumb {{ background:{C['border']}; border-radius:3px; }}
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPERS
# ============================================================
def card(label, value, sub="", badge="", bcls=""):
    b = f'<span class="badge {bcls}">{badge}</span>' if badge else ""
    return f'<div class="card"><div class="lbl">{label}</div><div class="val">{value}</div><div class="sub">{sub}</div>{b}</div>'

def sh(title, tag=""):
    t = f'<span class="tag">{tag}</span>' if tag else ""
    return f'<div class="sh"><div class="dot"></div><h3>{title}</h3>{t}</div>'

def fig_dark(fig, axes):
    fig.patch.set_facecolor(C["card"])
    for ax in (axes if isinstance(axes, list) else [axes]):
        ax.set_facecolor(C["card"])
        ax.tick_params(colors=C["dim"], labelsize=7)
        for sp in ['top','right']: ax.spines[sp].set_visible(False)
        ax.spines['left'].set_color(C["border"])
        ax.spines['bottom'].set_color(C["border"])
        ax.xaxis.label.set_color(C["dim"])
        ax.yaxis.label.set_color(C["dim"])
        ax.title.set_color(C["txt"]); ax.title.set_fontsize(8.2); ax.title.set_fontweight('600')
        lg = ax.get_legend()
        if lg:
            lg.get_frame().set_facecolor(C["bg"]); lg.get_frame().set_edgecolor(C["border"])
            for t in lg.get_texts(): t.set_color(C["dim"]); t.set_fontsize(6.5)


# ============================================================
# DATA LOADING FUNCTIONS  (replicates notebook logic exactly)
# ============================================================
def load_kelulusan(uploaded_files):
    """Load multiple CSV files (kelulusan folder)."""
    dfs = []
    for f in uploaded_files:
        dfs.append(pd.read_csv(f))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def load_hr(uploaded_file):
    """Load single HR .xlsx, keep & rename required columns."""
    df = pd.read_excel(uploaded_file)
    needed = ['Nama','ID','Kemampuan komunikasi','Pengalaman Kerja',
              'Keterampilan Teknis','Penilaian Kinerja Kepegawaian']
    df = df[needed].copy()
    df['Platform'] = 'HR (Baseline)'
    df = df.dropna()
    return df

def load_ai_files(uploaded_files):
    """Load 5 AI .xlsx files, normalise columns, concatenate."""
    alt_cols = {'Kemampuan Komunikasi': 'Kemampuan komunikasi'}
    keep = ['Platform','ID','Nama','Kemampuan komunikasi',
            'Pengalaman Kerja','Keterampilan Teknis',
            'Penilaian Kinerja Kepegawaian','Waktu']
    dfs = []
    for f in uploaded_files:
        name = os.path.splitext(f.name)[0].replace('_',' ')
        df = pd.read_excel(f)
        df = df.rename(columns=alt_cols)
        if 'AI Tools' in df.columns:
            df['Platform'] = df['AI Tools']
        else:
            df['Platform'] = name
        avail = [c for c in keep if c in df.columns]
        dfs.append(df[avail].copy())
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def load_member(uploaded_file):
    """Load member export CSV, keep Nama + Tingkat_Kelulusan."""
    df = pd.read_csv(uploaded_file)
    df = df[['Nama Fasilitator','Persentase Kelulusan Peserta']].copy()
    df.rename(columns={'Nama Fasilitator':'Nama',
                        'Persentase Kelulusan Peserta':'Tingkat_Kelulusan'}, inplace=True)
    return df


# ============================================================
# ANALYSIS FUNCTIONS  (replicates notebook logic exactly)
# ============================================================
def run_all_analysis(df_complete, df_member):
    """Run full pipeline, return dict of all result DataFrames."""
    res = {}

    # --- Normality ---
    norm_rows = []
    for var in VARIABLES:
        for plat in df_complete['Platform'].unique():
            data = df_complete[df_complete['Platform']==plat][var].dropna()
            if len(data) >= 3:
                w, p = shapiro(data)
                norm_rows.append({"Variabel":var,"Platform":plat,"W":round(w,4),"P":round(p,4),"Normal": p>0.05})
    res['normality'] = pd.DataFrame(norm_rows)

    # --- RM-ANOVA ---
    anova_rows = []
    for var in VARIABLES:
        rm = pg.rm_anova(data=df_complete, dv=var, within='Platform', subject='ID', detailed=True)
        F  = rm['F'].values[0]
        p  = rm['p-unc'].values[0]
        eta = rm['np2'].values[0] if 'np2' in rm.columns else None
        anova_rows.append({"Variabel":var,"F":round(F,4),"p":round(p,6),"eta2p":round(eta,4) if eta else None,"Sig": p<0.05})
    res['anova'] = pd.DataFrame(anova_rows)

    # --- Post-hoc ---
    posthoc_all = {}
    for var in VARIABLES:
        ph = pg.pairwise_tests(data=df_complete, dv=var, within='Platform',
                               subject='ID', padjust='bonf', effsize='hedges')
        posthoc_all[var] = ph
    res['posthoc'] = posthoc_all

    # --- Accuracy vs HR ---
    hr_base = df_complete[df_complete['Platform']=='HR (Baseline)'].copy()
    acc_rows = []
    for plat in df_complete['Platform'].unique():
        if plat == 'HR (Baseline)': continue
        df_ai = df_complete[df_complete['Platform']==plat].copy()
        merged = df_ai.merge(hr_base[['ID']+VARIABLES], on='ID', suffixes=('_AI','_HR'))
        for var in VARIABLES:
            ai_col, hr_col = f"{var}_AI", f"{var}_HR"
            mae  = mean_absolute_error(merged[hr_col], merged[ai_col])
            rmse = np.sqrt(mean_squared_error(merged[hr_col], merged[ai_col]))
            r2   = r2_score(merged[hr_col], merged[ai_col])
            pr, pp = pearsonr(merged[hr_col], merged[ai_col])
            sr, sp = spearmanr(merged[hr_col], merged[ai_col])
            md = (merged[ai_col]-merged[hr_col]).mean()
            acc_rows.append({"Platform":plat,"Variabel":var,
                             "MAE":round(mae,4),"RMSE":round(rmse,4),"R2":round(r2,4),
                             "Pearson_r":round(pr,4),"P_Pearson":round(pp,4),
                             "Spearman_rho":round(sr,4),"P_Spearman":round(sp,4),
                             "Mean_Diff":round(md,4)})
    res['accuracy'] = pd.DataFrame(acc_rows)

    # --- Overall Ranking (composite) ---
    df_acc = res['accuracy']
    ov = df_acc.groupby('Platform').agg({"MAE":"mean","RMSE":"mean","R2":"mean",
                                          "Pearson_r":"mean","Spearman_rho":"mean"}).copy()
    def n01(s, inv=False):
        mn,mx = s.min(),s.max()
        if mx==mn: return pd.Series(0.5, index=s.index)
        n = (s-mn)/(mx-mn)
        return 1-n if inv else n
    ov['cs'] = (n01(ov['MAE'],True)+n01(ov['RMSE'],True)+n01(ov['R2'])+n01(ov['Pearson_r'])+n01(ov['Spearman_rho']))/5
    ov = ov.sort_values('cs', ascending=False).reset_index()
    ov.columns = ['Platform','MAE','RMSE','R²','Pearson r','Spearman ρ','Composite Score']
    res['ranking'] = ov.round(4)

    # --- Correlation with Tingkat Kelulusan ---
    pivot = df_complete.pivot_table(index=['ID','Nama'], columns='Platform',
                                     values='Penilaian Kinerja Kepegawaian').reset_index()
    pivot.columns = ['ID','Nama'] + [f'PK_{c}' for c in pivot.columns[2:]]
    merged_corr = pivot.merge(df_member, on='Nama', how='left').dropna()
    corr_rows = []
    for col in [c for c in merged_corr.columns if c.startswith('PK_')]:
        plat = col.replace('PK_','')
        pr, pp = pearsonr(merged_corr[col], merged_corr['Tingkat_Kelulusan'])
        sr, sp = spearmanr(merged_corr[col], merged_corr['Tingkat_Kelulusan'])
        corr_rows.append({"Platform":plat,"Pearson_r":round(pr,4),"P_Pearson":round(pp,4),
                          "Spearman_rho":round(sr,4),"P_Spearman":round(sp,4)})
    res['correlation'] = pd.DataFrame(corr_rows)

    return res


# ============================================================
# SESSION-STATE BOOTSTRAP  —  upload screen rendered once
# ============================================================
def upload_screen():
    st.markdown(f"""
    <div style="max-width:680px;margin:3rem auto 0;text-align:center;">
        <div style="font-size:1.6rem;font-weight:700;color:{C['txt']};margin-bottom:0.25rem;">
            Upload Dataset Penelitian
        </div>
        <div style="font-size:0.75rem;color:{C['dim']};margin-bottom:1.8rem;">
            Masukkan semua file yang diperlukan untuk menjalankan analisis lengkap
        </div>
    </div>
    """, unsafe_allow_True=True)

    # Show required-file checklist
    steps = [
        ("1", "<strong>Folder Kelulusan</strong> — semua file <code>.csv</code> laporan akhir lokus (619 peserta total)"),
        ("2", "<strong>HR Baseline</strong> — 1 file <code>.xlsx</code> Matriks Penilaian HR Real"),
        ("3", "<strong>Penilaian AI</strong> — 5 file <code>.xlsx</code> (ChatGPT, Claude, DeepSeek, GLM, Gemini)"),
        ("4", "<strong>Member Export</strong> — 1 file <code>.csv</code> berisi <em>Nama Fasilitator</em> & <em>Persentase Kelulusan</em>"),
    ]
    st.markdown(f'<div style="max-width:680px;margin:0 auto;background:{C["card"]};border:1px solid {C["border"]};border-radius:12px;padding:1rem 1.2rem;margin-bottom:1.6rem;">' +
                ''.join(f'<div class="step"><div class="num">{n}</div><div class="txt">{t}</div></div>' for n,t in steps) +
                '</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.markdown(f'<div style="font-size:0.68rem;text-transform:uppercase;letter-spacing:1px;color:{C["gold"]};margin-bottom:0.35rem;font-weight:600;">① Kelulusan (CSV folder)</div>', unsafe_allow_html=True)
        kelulusan_files = st.file_uploader("", type="csv", accept_multiple_files=True, key="upload_kel", label_visibility="collapsed")

        st.markdown(f'<div style="font-size:0.68rem;text-transform:uppercase;letter-spacing:1px;color:{C["gold"]};margin-bottom:0.35rem;margin-top:0.9rem;font-weight:600;">② HR Baseline (XLSX)</div>', unsafe_allow_html=True)
        hr_file = st.file_uploader("", type="xlsx", key="upload_hr", label_visibility="collapsed")

    with col2:
        st.markdown(f'<div style="font-size:0.68rem;text-transform:uppercase;letter-spacing:1px;color:{C["gold"]};margin-bottom:0.35rem;font-weight:600;">③ Penilaian AI — 5 file (XLSX)</div>', unsafe_allow_html=True)
        ai_files = st.file_uploader("", type="xlsx", accept_multiple_files=True, key="upload_ai", label_visibility="collapsed")

        st.markdown(f'<div style="font-size:0.68rem;text-transform:uppercase;letter-spacing:1px;color:{C["gold"]};margin-bottom:0.35rem;margin-top:0.9rem;font-weight:600;">④ Member Export (CSV)</div>', unsafe_allow_html=True)
        member_file = st.file_uploader("", type="csv", key="upload_mem", label_visibility="collapsed")

    # Validation & status bar
    st.markdown("<br>")
    checks = {
        "Kelulusan CSV": len(kelulusan_files) if kelulusan_files else 0,
        "HR Baseline":   1 if hr_file else 0,
        "AI Files (5)":  len(ai_files) if ai_files else 0,
        "Member Export": 1 if member_file else 0,
    }
    targets = {"Kelulusan CSV": 1, "HR Baseline": 1, "AI Files (5)": 5, "Member Export": 1}  # min required

    status_html = '<div style="display:flex;gap:0.5rem;flex-wrap:wrap;justify-content:center;">'
    all_ok = True
    for label, count in checks.items():
        ok = count >= targets[label]
        if not ok: all_ok = False
        clr = C["teal"] if ok else C["coral"]
        bg  = "#4ecdc418" if ok else "#ff6b6b18"
        icon = "✓" if ok else "✗"
        status_html += f'<div style="background:{bg};border:1px solid {clr}33;border-radius:6px;padding:0.28rem 0.7rem;font-size:0.68rem;color:{clr};font-weight:600;">{icon} {label} ({count})</div>'
    status_html += '</div>'
    st.markdown(status_html, unsafe_allow_html=True)

    if all_ok:
        st.markdown("<br>")
        col_btn = st.columns([1,2,1])[1]
        if col_btn.button("🚀  Jalankan Analisis", use_container_width=True):
            with st.spinner("Memproses data dan menjalankan analisis…"):
                df_kel    = load_kelulusan(kelulusan_files)
                df_hr     = load_hr(hr_file)
                df_ai     = load_ai_files(ai_files)
                df_member = load_member(member_file)

                df_combined = pd.concat([df_hr, df_ai], ignore_index=True)
                df_combined = df_combined.dropna(subset=['ID']+VARIABLES)

                # Keep only IDs present in all 6 platforms
                plat_count = df_combined.groupby('ID')['Platform'].nunique()
                n_plat = df_combined['Platform'].nunique()
                complete_ids = plat_count[plat_count==n_plat].index
                df_complete = df_combined[df_combined['ID'].isin(complete_ids)].copy()
                df_complete = df_complete.sort_values(['ID','Platform']).reset_index(drop=True)

                results = run_all_analysis(df_complete, df_member)

                # Persist in session state
                st.session_state['df_complete']  = df_complete
                st.session_state['df_kelulusan'] = df_kel
                st.session_state['df_member']    = df_member
                st.session_state['results']      = results
                st.session_state['analysed']     = True
            st.rerun()
    else:
        st.markdown(f'<div style="text-align:center;margin-top:0.8rem;font-size:0.72rem;color:{C["dim"]};">Lengkapi semua file di atas untuk mengaktifkan analisis.</div>', unsafe_allow_html=True)


# ============================================================
# DASHBOARD PAGES  (rendered only after analysis is done)
# ============================================================
def page_overview():
    df   = st.session_state['df_complete']
    res  = st.session_state['results']
    df_k = st.session_state['df_kelulusan']
    n_mem = df['ID'].nunique()
    n_plat = df['Platform'].nunique()
    n_loc  = df_k.iloc[:,0].nunique() if len(df_k) else 0   # proxy: unique first-col values
    n_pes  = len(df_k)

    st.markdown(f'<div style="margin-bottom:1.2rem;"><div style="font-size:1.4rem;font-weight:700;color:{C["txt"]};">Evaluasi Akurasi Rekomendasi AI</div><div style="font-size:0.74rem;color:{C["dim"]};margin-top:0.2rem;">Comparative Multi-AI Evaluation · Yayasan Sakata Innovation Center</div></div>', unsafe_allow_html=True)

    # KPI row
    cols = st.columns(6)
    kpis = [
        ("Anggota Tim", str(n_mem), "Within-subjects", "", ""),
        ("Platform", str(n_plat), "1 HR + 5 AI", "", ""),
        ("Lokasi", str(n_loc), "Kegiatan diklat", "", ""),
        ("Peserta", str(n_pes), "Total diklat", "", ""),
        ("Variabel", "3", "Kompetensi individu", "", ""),
        ("RM-ANOVA", "p < .001", "Semua signifikan", "Sig.", "b-sig"),
    ]
    for c,(l,v,s,b,bc) in zip(cols, kpis):
        c.markdown(card(l,v,s,b,bc), unsafe_allow_html=True)

    st.markdown("<br>")

    # Platform list + 3 overview charts
    col_l, col_r = st.columns([1, 1.15], gap="medium")

    with col_l:
        st.markdown(sh("Platform yang Dievaluasi","6 Metode"), unsafe_allow_html=True)
        for p in PLAT_ORDER:
            tag = "⚖️ Baseline" if "HR" in p else "🤖 AI"
            st.markdown(f'<div style="display:flex;align-items:center;gap:0.55rem;padding:0.38rem 0.6rem;background:{C["card"]};border:1px solid {C["border"]};border-left:3px solid {PLAT_CLR[p]};border-radius:7px;margin-bottom:0.22rem;"><span style="font-size:0.82rem;">{tag.split()[0]}</span><div style="flex:1;"><div style="font-size:0.74rem;font-weight:600;color:{C["txt"]}">{p}</div></div></div>', unsafe_allow_html=True)

    with col_r:
        st.markdown(sh("Struktur Data","Komposisi"), unsafe_allow_html=True)

        # --- Donut ---
        fig, ax = plt.subplots(figsize=(4.2, 3.4))
        fig.patch.set_facecolor(C["card"]); ax.set_facecolor(C["card"])
        sizes  = [n_mem]*n_plat
        labels = list(df['Platform'].unique())
        colors = [PLAT_CLR.get(l,"#aaa") for l in labels]
        w, t, at = ax.pie(sizes, autopct='%1.0f%%', colors=colors, startangle=90,
                          pctdistance=0.82, wedgeprops=dict(width=0.35, edgecolor=C["card"], linewidth=2))
        for a in at: a.set_color(C["txt"]); a.set_fontsize(6.5); a.set_fontweight('600')
        ax.legend(w, labels, loc="center left", bbox_to_anchor=(1.05,0.5), fontsize=5.8, frameon=True)
        lg=ax.get_legend(); lg.get_frame().set_facecolor(C["bg"]); lg.get_frame().set_edgecolor(C["border"])
        for tt in lg.get_texts(): tt.set_color(C["dim"])
        ax.set_title("Evaluasi per Platform", color=C["txt"], fontsize=7.8, fontweight='600', pad=10)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

        # --- Horizontal bar: komposisi ---
        fig, ax = plt.subplots(figsize=(4.2, 2.2))
        fig.patch.set_facecolor(C["card"]); ax.set_facecolor(C["card"])
        cats = ["Anggota Tim","Evaluasi Total","Lokasi","Peserta"]
        vals = [n_mem, n_mem*n_plat, n_loc, n_pes]
        clrs = [C["gold"],C["teal"],C["blue"],C["purple"]]
        y    = np.arange(len(cats))
        bars = ax.barh(y, vals, color=clrs, height=0.4, edgecolor=C["card"])
        ax.set_yticks(y); ax.set_yticklabels(cats, fontsize=6.8, color=C["dim"])
        for b,v in zip(bars,vals): ax.text(b.get_width()+6, b.get_y()+b.get_height()/2, str(v), va='center', fontsize=6.8, fontweight='600', color=C["txt"])
        ax.set_xlim(0, max(vals)*1.18); ax.set_title("Komposisi Dataset", color=C["txt"], fontsize=7.8, fontweight='600')
        fig_dark(fig, ax); plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()


def page_distribusi():
    df  = st.session_state['df_complete']
    sel = st.session_state.get('sel_vars', VARIABLES)
    sel_p = st.session_state.get('sel_plats', PLAT_ORDER[1:])
    plats = ['HR (Baseline)'] + [p for p in PLAT_ORDER[1:] if p in sel_p]

    st.markdown(f'<div style="margin-bottom:1rem;"><div style="font-size:1.3rem;font-weight:700;color:{C["txt"]};">Distribusi & Statistik Deskriptif</div><div style="font-size:0.72rem;color:{C["dim"]};margin-top:0.15rem;">Sebaran skor penilaian per platform dan variabel</div></div>', unsafe_allow_html=True)

    # --- Boxplot ---
    st.markdown(sh("Distribusi Skor","Boxplot"), unsafe_allow_html=True)
    n = len(sel) if sel else 1
    fig, axes = plt.subplots(1, n, figsize=(5.2*n, 3.6))
    fig.patch.set_facecolor(C["card"])
    if n==1: axes=[axes]

    for idx, var in enumerate(sel if sel else VARIABLES[:1]):
        ax = axes[idx]; ax.set_facecolor(C["card"])
        data_list, lbl_list, clr_list = [], [], []
        for p in plats:
            d = df[df['Platform']==p][var].dropna()
            if len(d):
                data_list.append(d.values)
                lbl_list.append(p.replace(' (Baseline)','*').replace('Chat ','').replace(' Sonnet 4.5','*').replace(' V3.2','*').replace(' 4.7','*').replace(' 3 Pro','*'))
                clr_list.append(PLAT_CLR.get(p,"#aaa"))
        bp = ax.boxplot(data_list, patch_artist=True, widths=0.5,
                        medianprops=dict(color=C["txt"],linewidth=1.4),
                        whiskerprops=dict(color=C["dim"]),capprops=dict(color=C["dim"]),
                        flierprops=dict(marker='o',markerfacecolor=C["dim"],markersize=2.5,linestyle='none'))
        for patch,clr in zip(bp['boxes'],clr_list): patch.set_facecolor(clr); patch.set_alpha(0.5); patch.set_edgecolor(clr)
        ax.set_xticklabels(lbl_list, fontsize=5.6, rotation=22, color=C["dim"])
        ax.set_title(var, color=C["txt"], fontsize=7.8, fontweight='600')
        ax.set_ylabel("Skor", fontsize=6.8, color=C["dim"])
        ax.grid(True, alpha=0.1, color=C["border"])
        fig_dark(fig, ax)
    plt.tight_layout(pad=1.2); st.pyplot(fig, use_container_width=True); plt.close()

    # --- Mean profile plot ---
    st.markdown("<br>"); st.markdown(sh("Rata-rata Skor","Profile Plot"), unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(9, 3.4)); fig.patch.set_facecolor(C["card"]); ax.set_facecolor(C["card"])
    x = np.arange(len(plats))
    markers = ['o','s','^']
    vc = [C["gold"],C["teal"],C["blue"]]
    for i, var in enumerate(sel if sel else VARIABLES[:1]):
        means = [df[df['Platform']==p][var].mean() for p in plats]
        ax.plot(x, means, marker=markers[i], markersize=6, linewidth=1.8, label=var, color=vc[i], markeredgecolor=C["card"], markeredgewidth=1.2)
        for xi, yi in zip(x, means): ax.text(xi, yi+0.6, f'{yi:.1f}', ha='center', va='bottom', fontsize=5.8, color=C["dim"])
    short = [p.replace(' (Baseline)','*').replace('Chat ','').replace(' Sonnet 4.5','*').replace(' V3.2','*').replace(' 4.7','*').replace(' 3 Pro','*') for p in plats]
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=6.8, color=C["dim"])
    ax.set_ylabel("Rata-rata Skor", fontsize=7, color=C["dim"])
    ax.set_title("Perbandingan Rata-rata Skor Antar Platform", color=C["txt"], fontsize=8, fontweight='600')
    ax.legend(fontsize=6, frameon=True); ax.grid(True, alpha=0.1, color=C["border"])
    fig_dark(fig, ax); plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    # --- Deskriptif tabel ---
    st.markdown("<br>"); st.markdown(sh("Statistik Deskriptif","Tabel"), unsafe_allow_html=True)
    rows = []
    for var in (sel if sel else VARIABLES):
        for p in plats:
            d = df[df['Platform']==p][var].dropna()
            if len(d):
                rows.append({"Variabel":var,"Platform":p,"N":len(d),"Mean":round(d.mean(),2),
                             "SD":round(d.std(),2),"Min":round(d.min(),2),"Median":round(d.median(),2),"Max":round(d.max(),2)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def page_anova():
    res = st.session_state['results']
    df_norm = res['normality']
    df_anova = res['anova']

    st.markdown(f'<div style="margin-bottom:1rem;"><div style="font-size:1.3rem;font-weight:700;color:{C["txt"]};">Uji Asumsi & Repeated Measures ANOVA</div><div style="font-size:0.72rem;color:{C["dim"]};margin-top:0.15rem;">Normalitas · Omnibus F-test · Post-hoc Heatmap</div></div>', unsafe_allow_html=True)

    # Normalitas summary
    total = len(df_norm)
    n_normal = df_norm['Normal'].sum()
    cols = st.columns(3)
    cols[0].markdown(card("Total Pengujian", str(total), "kombinasi var × platform"), unsafe_allow_html=True)
    cols[1].markdown(card("Normal", str(n_normal), "p > 0.05", "Normal", "b-sig"), unsafe_allow_html=True)
    cols[2].markdown(card("Tidak Normal", str(total-n_normal), "p ≤ 0.05", "Tidak Normal", "b-ns"), unsafe_allow_html=True)

    st.markdown("<br>")
    sel = st.session_state.get('sel_vars', VARIABLES)

    # Per-variable normalitas table
    for var in (sel if sel else VARIABLES):
        sub = df_norm[df_norm['Variabel']==var]
        st.markdown(sh(var), unsafe_allow_html=True)
        rows_h = ""
        for _, r in sub.iterrows():
            bc = "b-sig" if r['Normal'] else "b-ns"
            bt = "Normal" if r['Normal'] else "Tidak Normal"
            pc = C["teal"] if r['Normal'] else C["coral"]
            rows_h += f'<tr><td style="padding:.35rem .6rem;color:{C["txt"]};font-size:.74rem;font-weight:500;">{r["Platform"]}</td><td style="padding:.35rem .6rem;color:{C["dim"]};font-size:.72rem;font-family:\'JetBrains Mono\',monospace;">{r["W"]:.4f}</td><td style="padding:.35rem .6rem;color:{pc};font-size:.72rem;font-family:\'JetBrains Mono\',monospace;">{r["P"]:.4f}</td><td style="padding:.35rem .6rem;"><span class="badge {bc}">{bt}</span></td></tr>'
        st.markdown(f'<div style="background:{C["card"]};border:1px solid {C["border"]};border-radius:9px;overflow:hidden;margin-bottom:0.5rem;"><table style="width:100%;border-collapse:collapse;"><thead><tr style="background:{C["bg"]};border-bottom:1px solid {C["border"]};"><th style="padding:.35rem .6rem;text-align:left;font-size:.62rem;text-transform:uppercase;letter-spacing:1px;color:{C["dim"]};font-weight:500;">Platform</th><th style="padding:.35rem .6rem;text-align:left;font-size:.62rem;text-transform:uppercase;letter-spacing:1px;color:{C["dim"]};font-weight:500;">W</th><th style="padding:.35rem .6rem;text-align:left;font-size:.62rem;text-transform:uppercase;letter-spacing:1px;color:{C["dim"]};font-weight:500;">p-value</th><th style="padding:.35rem .6rem;text-align:left;font-size:.62rem;text-transform:uppercase;letter-spacing:1px;color:{C["dim"]};font-weight:500;">Kesimpulan</th></tr></thead><tbody>{rows_h}</tbody></table></div>', unsafe_allow_html=True)

    # ANOVA cards
    st.markdown("<br>"); st.markdown(sh("Hasil RM-ANOVA","Omnibus F-test"), unsafe_allow_html=True)
    cols = st.columns(3)
    for i,(_, r) in enumerate(df_anova.iterrows()):
        sig = "Sig." if r['Sig'] else "n.s."
        bcls = "b-sig" if r['Sig'] else "b-ns"
        cols[i].markdown(card(r['Variabel'], f"F = {r['F']:.2f}", f"p = {r['p']:.4f} · η²p = {r['eta2p']}", sig, bcls), unsafe_allow_html=True)

    # Post-hoc heatmap
    st.markdown("<br>"); st.markdown(sh("Post-Hoc Heatmap","Hedges d · Bonferroni"), unsafe_allow_html=True)
    var_ph = st.selectbox("Variabel:", sel if sel else VARIABLES, index=0, label_visibility="collapsed")
    ph = res['posthoc'][var_ph]
    plats_h = PLAT_ORDER
    mat = pd.DataFrame(0.0, index=plats_h, columns=plats_h)
    for _, row in ph.iterrows():
        a, b = row['A'], row['B']
        if a in plats_h and b in plats_h:
            d = row['hedges'] if 'hedges' in row else 0
            mat.loc[a,b] =  d
            mat.loc[b,a] = -d

    vals = mat.values.copy().astype(float)
    np.fill_diagonal(vals, np.nan)
    fig, ax = plt.subplots(figsize=(6.5, 5.2)); fig.patch.set_facecolor(C["card"])
    im = ax.imshow(vals, cmap='RdBu_r', vmin=-2.5, vmax=2.5, aspect='auto')
    short = [p.replace(' (Baseline)','*').replace('Chat ','').replace(' Sonnet 4.5','*').replace(' V3.2','*').replace(' 4.7','*').replace(' 3 Pro','*') for p in plats_h]
    ax.set_xticks(range(len(plats_h))); ax.set_xticklabels(short, fontsize=6, color=C["dim"], rotation=28, ha='right')
    ax.set_yticks(range(len(plats_h))); ax.set_yticklabels(short, fontsize=6, color=C["dim"])
    for i in range(len(plats_h)):
        for j in range(len(plats_h)):
            v = mat.values[i,j]
            sig = ph[((ph['A']==plats_h[i])&(ph['B']==plats_h[j]))|((ph['A']==plats_h[j])&(ph['B']==plats_h[i]))]
            is_sig = len(sig) and sig.iloc[0].get('p-corr',1)<0.05
            tc = C["txt"] if is_sig else C["dim"]
            fw = '700' if is_sig else '400'
            ax.text(j,i, f'{v:.2f}' if i!=j else '—', ha='center', va='center', fontsize=6, color=tc, fontweight=fw)
    ax.set_title(f"Effect Size — {var_ph}", color=C["txt"], fontsize=8, fontweight='600', pad=8)
    cb = fig.colorbar(im, ax=ax, shrink=0.78, pad=0.02); cb.ax.tick_params(colors=C["dim"], labelsize=5.8)
    cb.set_label("Hedges d", color=C["dim"], fontsize=6.5)
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
    st.markdown(f'<div style="font-size:.62rem;color:{C["dim"]};margin-top:.2rem;">Nilai tebal = signifikan (p &lt; 0.05 Bonferroni). Biru = negatif, Merah = positif.</div>', unsafe_allow_html=True)


def page_akurasi():
    res = st.session_state['results']
    df_acc = res['accuracy']
    sel   = st.session_state.get('sel_vars', VARIABLES)
    sel_p = st.session_state.get('sel_plats', PLAT_ORDER[1:])

    df_f = df_acc[df_acc['Platform'].isin(sel_p) & df_acc['Variabel'].isin(sel if sel else VARIABLES)]

    st.markdown(f'<div style="margin-bottom:1rem;"><div style="font-size:1.3rem;font-weight:700;color:{C["txt"]};">Akurasi AI vs HR Baseline</div><div style="font-size:0.72rem;color:{C["dim"]};margin-top:0.15rem;">MAE · RMSE · R² · Pearson r · Spearman ρ</div></div>', unsafe_allow_html=True)

    # Aggregated top cards
    agg = df_f.groupby('Platform').agg({"MAE":"mean","RMSE":"mean","R2":"mean","Pearson_r":"mean"}).sort_values("MAE")
    rank_c = [C["gold"],C["teal"],C["blue"],C["purple"],C["coral"]]
    cols = st.columns(min(len(agg),5))
    for i,(plat,row) in enumerate(agg.iterrows()):
        if i<len(cols):
            cols[i].markdown(f'<div class="card" style="border-top:3px solid {rank_c[i]};"><div class="lbl" style="color:{rank_c[i]};">#{i+1} — {plat}</div><div class="val" style="font-size:1.2rem;">MAE {row["MAE"]:.3f}</div><div class="sub">r = {row["Pearson_r"]:.3f} · R² = {row["R2"]:.3f}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>")

    # MAE bar per variable
    st.markdown(sh("Mean Absolute Error","Per Variabel"), unsafe_allow_html=True)
    n = len(sel) if sel else 1
    fig, axes = plt.subplots(1, n, figsize=(5*n, 3.2)); fig.patch.set_facecolor(C["card"])
    if n==1: axes=[axes]
    for idx, var in enumerate(sel if sel else VARIABLES[:1]):
        ax = axes[idx]; ax.set_facecolor(C["card"])
        sub = df_f[df_f['Variabel']==var].sort_values('MAE')
        plats_v = sub['Platform'].tolist(); maes = sub['MAE'].tolist()
        clrs = [PLAT_CLR.get(p,"#aaa") for p in plats_v]
        bars = ax.barh(plats_v, maes, color=clrs, height=0.4, edgecolor=C["card"])
        for b,v in zip(bars,maes): ax.text(b.get_width()+0.04, b.get_y()+b.get_height()/2, f'{v:.3f}', va='center', fontsize=6.2, color=C["dim"], fontweight='500')
        ax.set_title(var, color=C["txt"], fontsize=7.8, fontweight='600')
        ax.set_xlabel("MAE (↓ lebih baik)", fontsize=6.8, color=C["dim"])
        ax.set_xlim(0, max(maes)*1.22)
        fig_dark(fig, ax)
    plt.tight_layout(pad=1); st.pyplot(fig, use_container_width=True); plt.close()

    # Pearson grouped bar
    st.markdown("<br>"); st.markdown(sh("Korelasi Pearson vs HR","Per Variabel"), unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(9, 3.2)); fig.patch.set_facecolor(C["card"]); ax.set_facecolor(C["card"])
    plats_x = [p for p in PLAT_ORDER[1:] if p in sel_p]
    x = np.arange(len(plats_x)); nv = len(sel) if sel else 1; w=0.2
    vc = [C["gold"],C["teal"],C["blue"]]
    for i, var in enumerate(sel if sel else VARIABLES[:1]):
        vals = [df_f[(df_f['Platform']==p)&(df_f['Variabel']==var)]['Pearson_r'].values[0] if len(df_f[(df_f['Platform']==p)&(df_f['Variabel']==var)]) else 0 for p in plats_x]
        ax.bar(x+(i-nv/2+0.5)*w, vals, w, label=var, color=vc[i], edgecolor=C["card"], linewidth=0.7)
    ax.axhline(0.349, color=C["coral"], linewidth=0.7, linestyle='--', alpha=0.55)
    ax.text(len(plats_x)-0.3, 0.37, 'sig. threshold', fontsize=5.2, color=C["coral"], alpha=0.7)
    short = [p.replace('Chat ','').replace(' Sonnet 4.5','*').replace(' V3.2','*').replace(' 4.7','*').replace(' 3 Pro','*') for p in plats_x]
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=6.8, color=C["dim"])
    ax.set_ylabel("Pearson r", fontsize=7, color=C["dim"])
    ax.set_title("Korelasi Pearson: Skor AI vs HR Baseline", color=C["txt"], fontsize=8, fontweight='600')
    ax.legend(fontsize=5.8, frameon=True); fig_dark(fig, ax)
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    # Full table
    st.markdown("<br>"); st.markdown(sh("Tabel Akurasi Lengkap","Detail"), unsafe_allow_html=True)
    disp = df_f[["Platform","Variabel","MAE","RMSE","R2","Pearson_r","Spearman_rho","Mean_Diff"]].copy()
    disp.columns = ["Platform","Variabel","MAE","RMSE","R²","Pearson r","Spearman ρ","Mean Diff"]
    st.dataframe(disp.sort_values(["Variabel","MAE"]).reset_index(drop=True), hide_index=True, use_container_width=True)


def page_ranking():
    res = st.session_state['results']
    ov  = res['ranking']

    st.markdown(f'<div style="margin-bottom:1rem;"><div style="font-size:1.3rem;font-weight:700;color:{C["txt"]};">Ranking & Komparatif Platform</div><div style="font-size:0.72rem;color:{C["dim"]};margin-top:0.15rem;">Composite Score · Agregasi metrik akurasi</div></div>', unsafe_allow_html=True)

    # Ranking rows
    st.markdown(sh("Overall Ranking","Composite Score"), unsafe_allow_html=True)
    medals = ["🥇","🥈","🥉","4️⃣","5️⃣"]
    max_cs = ov['Composite Score'].max()
    for i,(_, r) in enumerate(ov.iterrows()):
        p = r['Platform']; cs = r['Composite Score']
        bw = cs/max_cs*100
        st.markdown(f'<div class="rank-row" style="border-left:3px solid {PLAT_CLR.get(p,"#aaa")};"><span class="rn">{medals[i]}</span><span class="rname" style="color:{C["txt"]}">{p}</span><div style="flex:1;max-width:160px;"><div style="background:{C["bg"]};border-radius:3px;height:5px;overflow:hidden;"><div style="width:{bw}%;height:100%;background:{PLAT_CLR.get(p,"#aaa")};border-radius:3px;"></div></div></div><span class="rscore">{cs:.4f}</span></div>', unsafe_allow_html=True)

    st.markdown("<br>")

    # Radar
    st.markdown(sh("Profil Komparatif","Radar"), unsafe_allow_html=True)
    cats = ['MAE\n(inv)','RMSE\n(inv)','R²','Pearson r','Spearman ρ']
    N = len(cats)
    angles = [n/N*2*np.pi for n in range(N)]; angles += angles[:1]

    def n01(arr, inv=False):
        mn,mx = arr.min(), arr.max()
        if mx==mn: return np.full_like(arr,0.5,dtype=float)
        n = (arr-mn)/(mx-mn)
        return 1-n if inv else n

    fig, ax = plt.subplots(figsize=(5.5,5.5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(C["card"]); ax.set_facecolor(C["card"])
    mae_n = n01(ov['MAE'].values, True); rmse_n = n01(ov['RMSE'].values, True)
    r2_n = n01(ov['R²'].values); pr_n = n01(ov['Pearson r'].values); sr_n = n01(ov['Spearman ρ'].values)

    for i,(_, r) in enumerate(ov.iterrows()):
        p = r['Platform']
        v = [mae_n[i],rmse_n[i],r2_n[i],pr_n[i],sr_n[i]]; v += v[:1]
        clr = PLAT_CLR.get(p,"#aaa")
        ax.plot(angles, v, 'o-', linewidth=1.6, label=p, color=clr, markersize=3.5)
        ax.fill(angles, v, alpha=0.07, color=clr)

    ax.set_xticks(angles[:-1]); ax.set_xticklabels(cats, fontsize=5.8, color=C["dim"])
    ax.set_ylim(0,1); ax.set_yticks([.25,.5,.75,1]); ax.set_yticklabels(['','','',''], fontsize=0)
    ax.grid(color=C["border"], linewidth=0.5, alpha=0.4); ax.spines['polar'].set_color(C["border"])
    ax.legend(loc='upper right', bbox_to_anchor=(1.32,1.08), fontsize=5.5, frameon=True)
    lg=ax.get_legend(); lg.get_frame().set_facecolor(C["bg"]); lg.get_frame().set_edgecolor(C["border"])
    for t in lg.get_texts(): t.set_color(C["dim"])
    ax.set_title("Profil Akurasi", color=C["txt"], fontsize=7.8, fontweight='600', pad=22)
    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("<br>"); st.markdown(sh("Tabel Ranking Detail","Semua Metrik"), unsafe_allow_html=True)
    st.dataframe(ov, hide_index=True, use_container_width=True)


def page_validitas():
    res = st.session_state['results']
    df_corr = res['correlation']

    st.markdown(f'<div style="margin-bottom:1rem;"><div style="font-size:1.3rem;font-weight:700;color:{C["txt"]};">Validitas Eksternal</div><div style="font-size:0.72rem;color:{C["dim"]};margin-top:0.15rem;">Korelasi Skor Kinerja vs Tingkat Kelulusan Peserta</div></div>', unsafe_allow_html=True)

    sig = df_corr[df_corr['P_Spearman']<0.05]
    ns  = df_corr[df_corr['P_Spearman']>=0.05]
    best = df_corr.loc[df_corr['Spearman_rho'].idxmax()]

    cols = st.columns(3)
    cols[0].markdown(card("Signifikan (Spearman)", str(len(sig)), "Korelasi bermakna", "Sig.", "b-sig"), unsafe_allow_html=True)
    cols[1].markdown(card("Tidak Signifikan", str(len(ns)), "Korelasi tidak bermakna", "n.s.", "b-ns"), unsafe_allow_html=True)
    cols[2].markdown(card("Tertinggi (ρ)", f"{best['Spearman_rho']:.4f}", best['Platform'], "Terbaik", "b-sig"), unsafe_allow_html=True)

    st.markdown("<br>"); st.markdown(sh("Korelasi Spearman","Validitas Eksternal"), unsafe_allow_html=True)

    df_s = df_corr.sort_values('Spearman_rho', ascending=True)
    fig, ax = plt.subplots(figsize=(7.5, 3.6)); fig.patch.set_facecolor(C["card"]); ax.set_facecolor(C["card"])
    plats = df_s['Platform'].tolist(); rhos = df_s['Spearman_rho'].tolist(); pvals = df_s['P_Spearman'].tolist()
    clrs = [PLAT_CLR.get(p,"#aaa") for p in plats]
    bars = ax.barh(plats, rhos, color=clrs, height=0.4, edgecolor=C["card"])
    for b, p, rho in zip(bars, pvals, rhos):
        sig_t = "*" if p<0.05 else "ns"
        xp = rho+(0.018 if rho>=0 else -0.018)
        ha = 'left' if rho>=0 else 'right'
        ax.text(xp, b.get_y()+b.get_height()/2, f'{rho:.3f} ({sig_t})', va='center', ha=ha, fontsize=6, color=C["dim"], fontweight='500')
    ax.axvline(0, color=C["border"], linewidth=0.7)
    ax.set_xlabel("Spearman ρ", fontsize=7, color=C["dim"])
    ax.set_title("Validitas Eksternal: Kinerja → Kelulusan Peserta", color=C["txt"], fontsize=8, fontweight='600')
    ax.set_xlim(-0.65, 0.85)
    fig_dark(fig, ax); plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown("<br>"); st.markdown(sh("Tabel Korelasi Lengkap","Pearson & Spearman"), unsafe_allow_html=True)
    disp = df_corr.copy()
    disp['Pearson Sig.']  = disp['P_Pearson'].apply(lambda p: "✓ Sig." if p<0.05 else "✗ n.s.")
    disp['Spearman Sig.'] = disp['P_Spearman'].apply(lambda p: "✓ Sig." if p<0.05 else "✗ n.s.")
    disp = disp[["Platform","Pearson_r","P_Pearson","Pearson Sig.","Spearman_rho","P_Spearman","Spearman Sig."]].round(4)
    disp.columns = ["Platform","Pearson r","p (Pearson)","Sig.","Spearman ρ","p (Spearman)","Sig. "]
    st.dataframe(disp, hide_index=True, use_container_width=True)

    st.markdown(f'<div style="background:{C["card"]};border:1px solid {C["border"]};border-left:3px solid {C["gold"]};border-radius:9px;padding:0.75rem 0.95rem;margin-top:0.8rem;font-size:0.72rem;color:{C["dim"]};line-height:1.8;"><strong style="color:{C["gold"]};">Interpretasi:</strong><br>• <strong style="color:{C["txt"]};">Claude Sonnet 4.5</strong> menunjukkan korelasi Spearman tertinggi (ρ = 0.6975, p &lt; .001).<br>• <strong style="color:{C["txt"]};">GLM 4.7</strong> juga kuat (ρ = 0.6748, p &lt; .001).<br>• <strong style="color:{C["txt"]};">HR (Baseline)</strong> menunjukkan korelasi <em>negatif</em> signifikan (ρ = −0.4856).<br>• <strong style="color:{C["txt"]};">Chat GPT 5.2</strong> tidak signifikan dengan tingkat kelulusan.</div>', unsafe_allow_html=True)


# ============================================================
# MAIN ROUTER
# ============================================================
if not st.session_state.get('analysed', False):
    upload_screen()
else:
    # Sidebar nav
    with st.sidebar:
        st.markdown(f'<div style="padding:1rem 0.3rem 0.4rem;"><div style="font-size:1rem;font-weight:700;color:{C["txt"]};line-height:1.3;">AI Eval<br>Dashboard</div><div style="font-size:0.56rem;color:{C["dim"]};text-transform:uppercase;letter-spacing:1.1px;margin-top:0.2rem;">Comparative Multi-AI</div></div>', unsafe_allow_html=True)
        st.markdown("<h2>Navigasi</h2>", unsafe_allow_html=True)
        pages = ["📋  Overview","📊  Distribusi & Deskriptif","🔬  Uji Asumsi & ANOVA",
                 "🎯  Akurasi AI vs HR","🏆  Ranking & Komparatif","🔗  Validitas Eksternal"]
        selected = st.radio("", pages, index=0)

        st.markdown("<h2 style='margin-top:1.4rem'>Filter</h2>", unsafe_allow_html=True)
        sel_vars  = st.multiselect("Variabel", VARIABLES, default=VARIABLES)
        sel_plats = st.multiselect("Platform AI", PLAT_ORDER[1:], default=PLAT_ORDER[1:])
        st.session_state['sel_vars']  = sel_vars
        st.session_state['sel_plats'] = sel_plats

        st.markdown(f"<hr style='border-color:{C['border']};margin:1.2rem 0 0.8rem;'>")
        df_c = st.session_state['df_complete']
        st.markdown(f'<div style="font-size:0.56rem;color:{C["dim"]};line-height:1.8;padding:0 .15rem;">Data dimuat:<br><span style="color:{C["gold"]};">{df_c["ID"].nunique()} anggota tim</span> · {df_c["Platform"].nunique()} platform<br>{len(st.session_state["df_kelulusan"])} peserta</div>', unsafe_allow_html=True)

        st.markdown("<br>")
        if st.button("↩ Upload Ulang", use_container_width=True):
            for k in ['df_complete','df_kelulusan','df_member','results','analysed']:
                st.session_state.pop(k, None)
            st.rerun()

    # Route
    if "Overview"    in selected: page_overview()
    elif "Distribusi" in selected: page_distribusi()
    elif "Asumsi"     in selected: page_anova()
    elif "Akurasi"    in selected: page_akurasi()
    elif "Ranking"    in selected: page_ranking()
    elif "Validitas"  in selected: page_validitas()
