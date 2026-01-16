import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Sparks Edu: 2019 Performance Review",
    page_icon="🚀",
    layout="wide"
)

# --- 2. LOAD DATA DYNAMICALLY ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('ads_data.csv')
        
        # Clean column names
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # --- FIX: HANDLE DATES & NUMBERS ---
        # 1. Dates: Add Year 2019
        YEAR_SUFFIX = " 2019" 
        df['week_start_date'] = df['week_start_date'].astype(str) + YEAR_SUFFIX
        df['week_start_date'] = pd.to_datetime(df['week_start_date'], format='%d %b %Y', errors='coerce')
        
        df = df.dropna(subset=['week_start_date']).sort_values('week_start_date')

        # 2. Numbers: Remove commas and convert to float
        numeric_cols = ['ads_spend', 'ads_impression', 'ads_click', 'ads_cta', 'ads_unique_cta', 'cpa', 'cpc', 'cpm']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 3. Calculate Key Metrics (USING UNIQUE CTA)
        # CPA (Cost Per Unique Acquisition)
        df['unique_cpa'] = df.apply(lambda x: x['ads_spend'] / x['ads_unique_cta'] if x['ads_unique_cta'] > 0 else 0, axis=1)
        
        # CTR (Click Through Rate) 
        df['ctr'] = df.apply(lambda x: (x['ads_click'] / x['ads_impression']) * 100 if x['ads_impression'] > 0 else 0, axis=1)
        
        # CVR (Unique Conversion Rate)
        df['unique_cvr'] = df.apply(lambda x: (x['ads_unique_cta'] / x['ads_click']) * 100 if x['ads_click'] > 0 else 0, axis=1)
        
        # 4. Define Efficiency Status for Visualization
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

# --- 3. DASHBOARD LAYOUT ---

if not df.empty:
    st.title("🚀 Sparks Edu: 2019 Ads Performance Case Study")
    st.markdown("""
    **Objective:** Analyze 2019 Weekly Ads Data to identify efficiency gaps.
    <br>
    **Key Metric:** `Unique CPA` (Cost per *New* Student Lead).
    """, unsafe_allow_html=True)
    
    # Global Metrics
    total_spend = df['ads_spend'].sum()
    total_unique_leads = df['ads_unique_cta'].sum()
    avg_unique_cpa = total_spend / total_unique_leads if total_unique_leads > 0 else 0
    
    # Top KPI Row
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total Investment (2019)", f"Rp {total_spend/1e9:.2f} B")
    kpi2.metric("Total Unique Leads", f"{total_unique_leads:,.0f}")
    kpi3.metric("Avg Cost Per Unique Lead", f"Rp {avg_unique_cpa:,.0f}")
    
    st.divider()

    # --- TABS FOR "SLIDE DECK" FLOW ---
    tab1, tab2, tab3, tab4 = st.tabs(["1. The Big Picture", "2. Success vs Failure", "3. Deep Dive Insights", "4. Strategic Plan"])

    # === TAB 1: SNAPSHOT VIEW ===
    with tab1:
        st.header("1. The Big Picture")
        st.markdown("### Spending vs Results Trend")
        
        fig_dual = go.Figure()
        fig_dual.add_trace(go.Bar(
            x=df['week_start_date'], y=df['ads_spend'], name='Ad Spend (Rp)',
            marker_color='#cbd5e1', opacity=0.7
        ))
        fig_dual.add_trace(go.Scatter(
            x=df['week_start_date'], y=df['ads_unique_cta'], name='Unique Leads',
            yaxis='y2', line=dict(color='#2563eb', width=3), mode='lines+markers'
        ))
        fig_dual.update_layout(
            title="Weekly Spend vs. Unique Leads Generated",
            yaxis=dict(title='Spend (IDR)', showgrid=False),
            yaxis2=dict(title='Unique Leads', overlaying='y', side='right', showgrid=False),
            legend=dict(orientation="h", y=1.1),
            hovermode='x unified', template="plotly_white",
            height=500
        )
        st.plotly_chart(fig_dual, use_container_width=True)

    # === TAB 2: SUCCESS VS PROBLEM WEEKS ===
    with tab2:
        st.header("2. Identifying Performance Extremes")
        st.markdown("""
        We analyze **Efficiency**: How much did we pay for each new student?
        """)
        
        # 1. Visual Overview
        fig_status = px.scatter(
            df, 
            x='week_start_date', 
            y='unique_cpa', 
            color='status',
            size='ads_spend',
            color_discrete_map={'Good (Efficient)': '#22c55e', 'Problem (Expensive)': '#ef4444', 'Normal': '#94a3b8'},
            title="Timeline: Cost Per Unique Lead (CPA)",
            labels={'unique_cpa': 'Cost Per Unique Lead', 'week_start_date': 'Week'}
        )
        st.plotly_chart(fig_status, use_container_width=True)

        st.divider()

        # 2. Detailed Tables
        col_good, col_bad = st.columns(2)
        
        top_5_good = df.nsmallest(5, 'unique_cpa')
        top_5_bad = df.nlargest(5, 'unique_cpa')
        
        def format_currency(x): return f"Rp {x:,.0f}"

        with col_good:
            st.success("✅ 'Success' Weeks (Efficient)")
            st.dataframe(
                top_5_good[['week_start_date', 'ads_spend', 'ads_unique_cta', 'unique_cpa']].style.format({
                    'ads_spend': format_currency, 'unique_cpa': format_currency, 'ads_unique_cta': '{:,.0f}'
                }),
                hide_index=True, use_container_width=True
            )

        with col_bad:
            st.error("⚠️ 'Problem' Weeks (Expensive)")
            st.dataframe(
                top_5_bad[['week_start_date', 'ads_spend', 'ads_unique_cta', 'unique_cpa']].style.format({
                    'ads_spend': format_currency, 'unique_cpa': format_currency, 'ads_unique_cta': '{:,.0f}'
                }),
                hide_index=True, use_container_width=True
            )

    # === TAB 3: INSIGHTS ===
    with tab3:
        st.header("3. Deep Dive Insights")
        
        col_insight_1, col_insight_2 = st.columns(2)
        
        # --- INSIGHT A: SCALING TRAP ---
        with col_insight_1:
            st.subheader("A. The 'Scaling Trap'")
            st.markdown("""
            **Observation:** Efficiency collapses when we spend too much too fast.
            """)
            
            fig_scatter = px.scatter(
                df, x='ads_spend', y='unique_cpa', size='ads_unique_cta', color='status',
                color_discrete_map={'Good (Efficient)': '#22c55e', 'Problem (Expensive)': '#ef4444', 'Normal': '#94a3b8'},
                title="Spend vs. Cost Efficiency",
                labels={'ads_spend': 'Total Weekly Spend', 'unique_cpa': 'Cost Per Unique Lead'}
            )
            fig_scatter.add_vline(x=2000000000, line_dash="dash", line_color="black", annotation_text="Efficiency Limit")
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            st.info("When Spend > Rp 2B (Right side), we almost always hit the 'Red Zone'. We are paying more to reach the same people repeatedly.")

        # --- INSIGHT B: QUALITY vs QUANTITY ---
        with col_insight_2:
            st.subheader("B. The 'Empty Click' Problem")
            st.markdown("""
            **Question:** "Clicks are always higher than Buys, so why is this a problem?"
            \n**Answer:** The *rate* of conversion dropped dangerously low.
            """)
            
            # Create a comparison of CVR for Good vs Bad weeks
            avg_cvr_good = top_5_good['unique_cvr'].mean()
            avg_cvr_bad = top_5_bad['unique_cvr'].mean()
            
            fig_bar_cvr = go.Figure(data=[
                go.Bar(name='Success Weeks', x=['Success Weeks'], y=[avg_cvr_good], marker_color='#22c55e', text=[f"{avg_cvr_good:.1f}%"], textposition='auto'),
                go.Bar(name='Problem Weeks', x=['Problem Weeks'], y=[avg_cvr_bad], marker_color='#ef4444', text=[f"{avg_cvr_bad:.1f}%"], textposition='auto')
            ])
            fig_bar_cvr.update_layout(title="Conversion Rate (Clicks to Leads)", yaxis_title="Conversion Rate (%)")
            st.plotly_chart(fig_bar_cvr, use_container_width=True)
            
            st.warning(f"""
            During Problem Weeks, the conversion rate crashed to **{avg_cvr_bad:.1f}%**. 
            This means we bought tons of "low quality" clicks (people who click but have no intent to buy). 
            During Good Weeks, we were much better at targeting "high intent" people (**{avg_cvr_good:.1f}%**).
            """)

    # === TAB 4: RECOMMENDATIONS ===
    with tab4:
        st.header("4. Strategic Recommendations")
        
        # Calculate dynamic numbers for the text
        avg_spend_bad = top_5_bad['ads_spend'].mean()
        avg_cpa_bad = top_5_bad['unique_cpa'].mean()
        
        st.markdown(f"""
        ### 1. 🛑 The "Soft Cap" Rule (Rp 2 Billion)
        * **The Insight:** Spending above Rp 2B yields diminishing returns.
        * **The Action:** Cap weekly spend at Rp 2B. If you have extra budget, save it for a different week or a different campaign. Do not force it into the same audience.

        ### 2. 🎯 Change the "Objective" (Fixing the Click Problem)
        * **The Insight:** Our high-spend weeks had huge clicks but low conversions. This usually happens when the Ad Algorithm is told to find "Clickers" instead of "Buyers".
        * **The Action:** Ensure your campaign objective is set to **"Leads"** or **"Conversions"**, NOT "Traffic" or "Link Clicks". We need the algorithm to filter out the "Window Shoppers."

        ### 3. 📉 Focus on "Unique" Metrics
        * **The Insight:** We found a big gap between `Total Clicks` and `Unique Leads`.
        * **The Action:** Stop reporting on "Total Clicks". It's a vanity metric. If a user clicks 10 times but buys once, we only care about the 1 buy. Base all future ROI reports on **Cost Per Unique Lead**.
        """)
        
        st.divider()
        st.caption("Generated for Sparks Edu Case Study | Data Source: 2019 Ads Export")

else:
    st.info("👋 Waiting for data. Please upload 'ads_data.csv' to your repository.")
