import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Sparks Edu: Executive Strategy Deck",
    page_icon="📊",
    layout="wide"
)

# --- 2. CUSTOM CSS FOR METRICS ---
st.markdown("""
<style>
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        color: #0f172a;
    }
    .big-font {
        font-size:18px !important;
        color: #334155;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. SIDEBAR LEGEND ---
with st.sidebar:
    st.header("Executive Summary")
    st.info("📊 **Analysis Context**\n\nComparing **efficiency** across spend levels to determine optimal scaling strategy.")
    st.markdown("---")
    st.markdown("**Definitions:**")
    st.success("🟢 **Efficient Zone**\nLow Spend Weeks")
    st.error("🔴 **Inefficient Zone**\nHigh Spend Weeks")

# --- 4. LOAD DATA DYNAMICALLY ---
@st.cache_data
def load_data():
    try:
        # NOTE: Ensure 'ads_data.csv' is in the same directory
        df = pd.read_csv('ads_data.csv')
        
        # CLEAN COLUMN NAMES
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # FIX: HANDLE DATES & NUMBERS
        YEAR_SUFFIX = " 2019" 
        df['week_start_date'] = df['week_start_date'].astype(str) + YEAR_SUFFIX
        df['week_start_date'] = pd.to_datetime(df['week_start_date'], format='%d %b %Y', errors='coerce')
        
        # Drop rows that failed date conversion
        df = df.dropna(subset=['week_start_date']).sort_values('week_start_date')

        numeric_cols = ['ads_spend', 'ads_impression', 'ads_click', 'ads_cta', 'ads_unique_cta', 'cpa', 'cpc', 'cpm']
        for col in numeric_cols:
            if col in df.columns:
                # Remove commas and convert to numeric
                df[col] = df[col].astype(str).str.replace(',', '', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # --- CALCULATED METRICS ---
        # 1. Unique CPA
        df['unique_cpa'] = df.apply(lambda x: x['ads_spend'] / x['ads_unique_cta'] if x['ads_unique_cta'] > 0 else 0, axis=1)
        
        # 2. CTR
        df['ctr'] = df.apply(lambda x: (x['ads_click'] / x['ads_impression']) * 100 if x['ads_impression'] > 0 else 0, axis=1)
        
        # 3. Booking Rate
        df['booking_rate'] = df.apply(lambda x: (x['ads_cta'] / x['ads_click']) * 100 if x['ads_click'] > 0 else 0, axis=1)
        
        # 4. Calculated CPC
        df['calculated_cpc'] = df.apply(lambda x: x['ads_spend'] / x['ads_click'] if x['ads_click'] > 0 else 0, axis=1)
        
        # 5. Status Definition
        cpa_25 = df['unique_cpa'].quantile(0.25)
        cpa_75 = df['unique_cpa'].quantile(0.75)
        
        def get_status(cpa):
            if cpa <= cpa_25: return 'Good (Efficient)'
            elif cpa >= cpa_75: return 'Problem (Expensive)'
            else: return 'Normal'
            
        df['status'] = df['unique_cpa'].apply(get_status)
        
        return df

    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return pd.DataFrame()

df = load_data()

# --- 5. MAIN DASHBOARD LAYOUT ---

if not df.empty:
    st.title("🚀 Sparks Edu: Strategic Performance Review")
    st.markdown("### FY2019 Ad Performance & Optimization Plan")
    st.divider()

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["1. Performance Snapshot", "2. Efficiency Audit", "3. Deep Dive & Strategy (The Fix)"])

    # ==========================================
    # TAB 1: SNAPSHOT
    # ==========================================
    with tab1:
        st.subheader("Executive Overview: Spend vs. Acquisition")
        
        # High Level Metrics
        total_spend = df['ads_spend'].sum()
        total_unique_leads = df['ads_unique_cta'].sum()
        avg_unique_cpa = total_spend / total_unique_leads if total_unique_leads > 0 else 0
        
        k1, k2, k3 = st.columns(3)
        k1.metric("Total Investment", f"Rp {total_spend/1e9:.2f} B")
        k2.metric("Total Unique Leads", f"{total_unique_leads:,.0f}")
        k3.metric("Avg Cost Per Lead", f"Rp {avg_unique_cpa:,.0f}")
        
        st.markdown("---")
        
        # Dual Axis Chart
        fig_dual = go.Figure()
        fig_dual.add_trace(go.Bar(x=df['week_start_date'], y=df['ads_spend'], name='Ad Spend (Rp)', marker_color='#cbd5e1', opacity=0.6))
        fig_dual.add_trace(go.Scatter(x=df['week_start_date'], y=df['ads_unique_cta'], name='Unique Leads', yaxis='y2', line=dict(color='#0f172a', width=3)))
        
        fig_dual.update_layout(
            title="Weekly Volume: Spend vs Unique Leads",
            yaxis=dict(title='Spend (IDR)', showgrid=False),
            yaxis2=dict(title='Unique Leads', overlaying='y', side='right', showgrid=False),
            legend=dict(orientation="h", y=1.1),
            hovermode='x unified',
            template="plotly_white",
            height=450
        )
        st.plotly_chart(fig_dual, use_container_width=True)

    # ==========================================
    # TAB 2: EFFICIENCY AUDIT
    # ==========================================
    with tab2:
        st.subheader("Efficiency Audit: Identifying Value Leaks")
        
        # Color Map
        color_map = {'Good (Efficient)': '#22c55e', 'Problem (Expensive)': '#ef4444', 'Normal': '#94a3b8'}
        
        fig_status = px.scatter(
            df, 
            x='week_start_date', 
            y='unique_cpa', 
            color='status', 
            size='ads_spend', 
            color_discrete_map=color_map,
            title="Cost Efficiency Timeline (Bubble Size = Spend Amount)",
            labels={'unique_cpa': 'CPA (Rp)', 'week_start_date': 'Week'}
        )
        fig_status.add_hline(y=df['unique_cpa'].mean(), line_dash="dot", annotation_text="Avg CPA", annotation_position="bottom right")
        
        st.plotly_chart(fig_status, use_container_width=True)
        
        col_good, col_bad = st.columns(2)
        top_5_good = df.nsmallest(5, 'unique_cpa')
        top_5_bad = df.nlargest(5, 'unique_cpa')
        
        def format_currency(x): return f"Rp {x:,.0f}"

        with col_good:
            st.success("✅ Top Performers (Benchmark)")
            st.dataframe(
                top_5_good[['week_start_date', 'ads_spend', 'ads_unique_cta', 'unique_cpa']]
                .style.format({'ads_spend': format_currency, 'unique_cpa': format_currency, 'ads_unique_cta': '{:,.0f}'}),
                hide_index=True, use_container_width=True
            )

        with col_bad:
            st.error("⚠️ Value Leaks (Inefficient)")
            st.dataframe(
                top_5_bad[['week_start_date', 'ads_spend', 'ads_unique_cta', 'unique_cpa']]
                .style.format({'ads_spend': format_currency, 'unique_cpa': format_currency, 'ads_unique_cta': '{:,.0f}'}),
                hide_index=True, use_container_width=True
            )

    # ==========================================
    # TAB 3: DEEP DIVE & STRATEGY (MERGED)
    # ==========================================
    with tab3:
        st.subheader("🛑 Deep Dive: The Diminishing Returns 'Wall'")
        
        # --- CONTROL: THRESHOLD SLIDER ---
        st.markdown("Use the slider to define the 'High Spend' threshold. See how efficiency breaks as we scale.")
        threshold = st.slider("Define High Spend Threshold (IDR)", min_value=1_000_000_000, max_value=3_000_000_000, value=2_000_000_000, step=100_000_000)
        
        # --- DATA SEGMENTATION ---
        low_spend_weeks = df[df['ads_spend'] < threshold]
        high_spend_weeks = df[df['ads_spend'] >= threshold]
        
        # Metrics Calculation
        cpa_low = low_spend_weeks['unique_cpa'].mean()
        cpa_high = high_spend_weeks['unique_cpa'].mean()
        
        cpc_low = low_spend_weeks['calculated_cpc'].mean()
        cpc_high = high_spend_weeks['calculated_cpc'].mean()
        
        br_low = low_spend_weeks['booking_rate'].mean()
        br_high = high_spend_weeks['booking_rate'].mean()
        
        # Growth Rates
        cpa_growth = ((cpa_high - cpa_low) / cpa_low) * 100
        cpc_growth = ((cpc_high - cpc_low) / cpc_low) * 100
        br_growth = ((br_high - br_low) / br_low) * 100

        # --- SECTION A: VISUAL EVIDENCE (THE ELBOW) ---
        col_viz1, col_viz2 = st.columns([2, 1])
        
        with col_viz1:
            st.markdown("#### 1. The 'Elbow' Curve")
            # FIX: Removed 'trendline="lowess"' to prevent ModuleNotFoundError
            fig_elbow = px.scatter(
                df, x="ads_spend", y="unique_cpa", 
                # trendline="lowess", # Removed to ensure compatibility
                title="Diminishing Returns: As Spend (X) Increases, CPA (Y) Rises",
                labels={"ads_spend": "Weekly Ad Spend (IDR)", "unique_cpa": "Cost Per Acquisition (IDR)"},
                color="status",
                color_discrete_map=color_map
            )
            # Add vertical line for threshold
            fig_elbow.add_vline(x=threshold, line_dash="dash", line_color="black", annotation_text="Threshold Limit")
            st.plotly_chart(fig_elbow, use_container_width=True)
            
        with col_viz2:
            st.markdown("#### 2. The Verdict")
            st.info(f"**Threshold Set: Rp {threshold/1e9:.1f} B**")
            
            delta_color = "inverse" if cpa_growth > 0 else "normal"
            st.metric(label="CPA Impact (High vs Low Spend)", value=f"Rp {cpa_high:,.0f}", delta=f"{cpa_growth:.1f}% Increase", delta_color=delta_color)
            
            st.write("---")
            if cpa_growth > 5:
                st.error(f"⚠️ **Inefficient Scaling**\n\nWhen spending >{threshold/1e9:.1f}B, we pay **{cpa_growth:.0f}% more** for every customer.")
            else:
                st.success(f"✅ **Efficient Scaling**\n\nSpending >{threshold/1e9:.1f}B is currently sustainable.")

        st.divider()

        # --- SECTION B: ROOT CAUSE ANALYSIS (DRIVER TREE) ---
        st.markdown("#### 3. Root Cause Analysis: Why is High Spend performing worse?")
        st.caption("CPA is driven by two factors: How expensive the traffic is (CPC) and how well it converts (Booking Rate).")
        
        rc1, rc2, rc3 = st.columns(3)
        
        with rc1:
            st.markdown("**Factor A: Media Cost (CPC)**")
            st.metric("Avg CPC (High Spend)", f"Rp {cpc_high:,.0f}", delta=f"{cpc_growth:.1f}% vs Low Spend", delta_color="inverse")
            st.progress(min(100, max(0, int(50 + cpc_growth))))
            st.caption("Higher is worse. This means competition is fiercer or audience is saturated.")
            
        with rc2:
            st.markdown("**Factor B: Quality (Booking Rate)**")
            st.metric("Avg Booking Rate", f"{br_high:.2f}%", delta=f"{br_growth:.1f}% vs Low Spend")
            st.progress(min(100, max(0, int(br_high * 10)))) # Arbitrary scale for visual
            st.caption("Higher is better. We are actually converting slightly better at scale.")

        with rc3:
            st.markdown("**➡️ The Conclusion**")
            st.warning(f"""
            The math is clear:
            
            We are paying **{cpc_growth:.0f}% more** for clicks, but the conversion rate only improved by **{br_growth:.0f}%**.
            
            **The Media Cost inflation is outpacing the Quality improvement.**
            """)

        st.divider()
        
        # --- SECTION C: RECOMMENDATIONS ---
        st.subheader("💡 Strategic Recommendations")
        
        rec1, rec2 = st.columns(2)
        with rec1:
            st.markdown("##### 🛡️ Defensive Action: Cap Spend")
            st.markdown(f"""
            * **Strict Limit:** Implement a soft cap at **Rp {threshold/1e9:.1f}B per week**.
            * **Rationale:** Every Rupiah spent above this generates diminishing returns.
            * **Action:** Reallocate excess budget to new channels (TikTok, Google UAC) to reset the CPC curve.
            """)
            
        with rec2:
            st.markdown("##### ⚔️ Offensive Action: Fix CPC")
            st.markdown(f"""
            * **The Problem:** CPC rose by {cpc_growth:.0f}%. Our creatives are fatiguing.
            * **Refresh Rate:** We need fresh creatives to drop CPC back to ~Rp {cpc_low:,.0f}.
            * **Target:** Launch **4 new creative angles** next week to lower auction costs.
            """)

else:
    st.info("👋 Waiting for data. Please upload 'ads_data.csv' to your repository.")
