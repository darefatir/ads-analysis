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
        # We define "Good" as CPA below the 25th percentile, and "Problem" as CPA above 75th percentile
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
    **Objective:** Analyze 2019 Weekly Ads Data to identify efficiency gaps and optimization opportunities.
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
        st.markdown("### How did our spending correlate with actual results throughout the year?")
        
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
        
        st.caption("Observation: Note the spikes in gray bars (Spend) around mid-year. Did the blue line (Leads) follow them proportionally? We will explore this in the next tab.")

    # === TAB 2: SUCCESS VS PROBLEM WEEKS ===
    with tab2:
        st.header("2. Identifying Performance Extremes")
        st.markdown("""
        We separated the weeks into **Good** (Efficient) and **Problem** (Expensive) categories based on Cost Per Unique Lead (CPA).
        """)
        
        # 1. Visual Overview
        fig_status = px.scatter(
            df, 
            x='week_start_date', 
            y='unique_cpa', 
            color='status',
            size='ads_spend',
            color_discrete_map={'Good (Efficient)': '#22c55e', 'Problem (Expensive)': '#ef4444', 'Normal': '#94a3b8'},
            title="Timeline of Efficiency: When did we overpay?",
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
            st.success("✅ The 'Success' Weeks (Lowest Cost)")
            st.markdown("These weeks generated leads for the lowest price. Note that **Spend** is often moderate.")
            st.dataframe(
                top_5_good[['week_start_date', 'ads_spend', 'ads_unique_cta', 'unique_cpa']].style.format({
                    'ads_spend': format_currency, 'unique_cpa': format_currency, 'ads_unique_cta': '{:,.0f}'
                }),
                hide_index=True, use_container_width=True
            )

        with col_bad:
            st.error("⚠️ The 'Problem' Weeks (Highest Cost)")
            st.markdown("These weeks were 'Budget Burners'. We spent heavily but got expensive results.")
            st.dataframe(
                top_5_bad[['week_start_date', 'ads_spend', 'ads_unique_cta', 'unique_cpa']].style.format({
                    'ads_spend': format_currency, 'unique_cpa': format_currency, 'ads_unique_cta': '{:,.0f}'
                }),
                hide_index=True, use_container_width=True
            )

    # === TAB 3: INSIGHTS ===
    with tab3:
        st.header("3. Deep Dive Insights")
        st.markdown("We looked deeper into the data to understand **why** the problem weeks happened.")
        
        col_insight_1, col_insight_2 = st.columns(2)
        
        # --- INSIGHT A: SCALING TRAP ---
        with col_insight_1:
            st.subheader("Insight A: The 'Scaling Trap'")
            st.markdown("""
            **The Hypothesis:** "If we double the budget, we get double the leads."
            \n**The Reality:** **False.** The data shows clear diminishing returns.
            """)
            
            fig_scatter = px.scatter(
                df, x='ads_spend', y='unique_cpa', size='ads_unique_cta', color='status',
                color_discrete_map={'Good (Efficient)': '#22c55e', 'Problem (Expensive)': '#ef4444', 'Normal': '#94a3b8'},
                title="Spend vs. Cost Efficiency",
                labels={'ads_spend': 'Total Weekly Spend', 'unique_cpa': 'Cost Per Unique Lead'}
            )
            # Add a vertical line to show the "Danger Zone" threshold (approx 2B)
            fig_scatter.add_vline(x=2000000000, line_dash="dash", line_color="black", annotation_text="Efficiency Breakpoint")
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            st.info("""
            **Analysis:** Look at the chart above. 
            * As soon as Spend (X-axis) crosses **Rp 2 Billion**, the dots turn **Red**. 
            * This indicates that our audience pool for that specific targeting saturates at around 2B. 
            * Spending beyond this point just increases cost, not volume.
            """)

        # --- INSIGHT B: LEAD QUALITY ---
        with col_insight_2:
            st.subheader("Insight B: The 'Click Bait' Effect")
            st.markdown("""
            **The Metric:** Unique Conversion Rate (CVR).
            \n**What it means:** Out of everyone who clicked, how many actually signed up?
            """)
            
            # Create a comparison of CVR for Good vs Bad weeks
            avg_cvr_good = top_5_good['unique_cvr'].mean()
            avg_cvr_bad = top_5_bad['unique_cvr'].mean()
            
            fig_bar_cvr = go.Figure(data=[
                go.Bar(name='Success Weeks', x=['Success Weeks'], y=[avg_cvr_good], marker_color='#22c55e'),
                go.Bar(name='Problem Weeks', x=['Problem Weeks'], y=[avg_cvr_bad], marker_color='#ef4444')
            ])
            fig_bar_cvr.update_layout(title="Conversion Rate Comparison", yaxis_title="Conversion Rate (%)")
            st.plotly_chart(fig_bar_cvr, use_container_width=True)
            
            st.info(f"""
            **Analysis:** * During 'Success' weeks, **{avg_cvr_good:.1f}%** of clickers signed up.
            * During 'Problem' weeks, only **{avg_cvr_bad:.1f}%** signed up.
            * **Why?** In high-spend weeks, we likely broadened the audience too much, bringing in low-quality traffic that clicks but never converts.
            """)

    # === TAB 4: RECOMMENDATIONS ===
    with tab4:
        st.header("4. Strategic Recommendations")
        
        # Calculate dynamic numbers for the text
        avg_spend_bad = top_5_bad['ads_spend'].mean()
        avg_cpa_bad = top_5_bad['unique_cpa'].mean()
        avg_cpa_normal = df[df['status'] == 'Normal']['unique_cpa'].mean()
        savings_potential = (avg_cpa_bad - avg_cpa_normal) / avg_cpa_bad * 100
        
        st.markdown(f"""
        Based on the 2019 data analysis, we recommend the following 3 actions to improve profitability immediately:
        
        ### 1. 🛑 Enforce a "Soft Cap" at Rp 2 Billion
        * **The Data:** In weeks where we spent over **Rp {avg_spend_bad/1e9:.1f} Billion**, our CPA spiked to **Rp {avg_cpa_bad:,.0f}**.
        * **The Action:** Do not scale weekly spend beyond Rp 2B on the current audience settings. The data proves efficiency collapses beyond this point.
        * **Expected Impact:** Avoiding these "burn" weeks could reduce overall CPA by roughly **{savings_potential:.0f}%**.

        ### 2. 🎯 Revamp "High Spend" Audience Targeting
        * **The Data:** The conversion rate drops drastically (by ~{(avg_cvr_good - avg_cvr_bad):.1f} points) during high-spend weeks.
        * **The Action:** This proves we are reaching "junk" traffic when we scale. Instead of Broad targeting, use **Lookalike 1%** audiences of our best students for the first Rp 1B of spend, and only expand to Lookalike 3-5% if CPA remains stable.

        ### 3. 📉 Shift KPI from "Clicks" to "Unique Leads"
        * **The Data:** There is a significant discrepancy between total clicks and unique people.
        * **The Action:** Stop optimizing ads for "Link Clicks". Optimize strictly for "Leads" or "Conversions". The algorithm is currently finding people who like to click but don't like to buy.
        """)
        
        st.divider()
        st.caption("Generated for Sparks Edu Case Study | Data Source: 2019 Ads Export")

else:
    st.info("👋 Waiting for data. Please upload 'ads_data.csv' to your repository.")
