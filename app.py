import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

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
        st.markdown("We analyze **Efficiency**: How much did we pay for each new student?")
        
        fig_status = px.scatter(
            df, 
            x='week_start_date', 
            y='unique_cpa', 
            color='status',
            size='ads_spend',
            color_discrete_map={'Good (Efficient)': '#22c55e', 'Problem (Expensive)': '#ef4444', 'Normal': '#94a3b8'},
            title="Timeline: Cost Per Unique Lead (CPA)",
            labels={'unique_cpa': 'CPA - Cost Per Acquisition (Rp)', 'week_start_date': 'Week'}
        )
        st.plotly_chart(fig_status, use_container_width=True)

        st.divider()

        col_good, col_bad = st.columns(2)
        top_5_good = df.nsmallest(5, 'unique_cpa')
        top_5_bad = df.nlargest(5, 'unique_cpa')
        def format_currency(x): return f"Rp {x:,.0f}"

        with col_good:
            st.success("✅ 'Success' Weeks (Efficient)")
            st.dataframe(top_5_good[['week_start_date', 'ads_spend', 'ads_unique_cta', 'unique_cpa']].style.format({'ads_spend': format_currency, 'unique_cpa': format_currency, 'ads_unique_cta': '{:,.0f}'}), hide_index=True, use_container_width=True)

        with col_bad:
            st.error("⚠️ 'Problem' Weeks (Expensive)")
            st.dataframe(top_5_bad[['week_start_date', 'ads_spend', 'ads_unique_cta', 'unique_cpa']].style.format({'ads_spend': format_currency, 'unique_cpa': format_currency, 'ads_unique_cta': '{:,.0f}'}), hide_index=True, use_container_width=True)

    # === TAB 3: INSIGHTS (NUANCED) ===
    with tab3:
        st.header("3. Deep Dive Insights")
        
        col_insight_1, col_insight_2 = st.columns(2)
        
        # --- INSIGHT A: SCALING TRAP (IMPROVED) ---
        with col_insight_1:
            st.subheader("A. The 'Scaling Trap'")
            st.markdown("""
            **Observation:** Efficiency degrades when we scale aggressively.
            
            **What is happening? (Audience Saturation)**
            When we spend > Rp 2B, we aren't just hitting the "same people." We are hitting **"harder to convince" people**.
            We have exhausted the easy wins (Low CPA), so the algorithm has to pay more to find the next student.
            """)
            
            # Create Labels (Only label the worst outliers for clarity)
            df['chart_label'] = df.apply(lambda x: x['week_start_date'].strftime('%d %b') if x['status'] == 'Problem (Expensive)' else '', axis=1)

            fig_scatter = px.scatter(
                df, x='ads_spend', y='unique_cpa', size='ads_unique_cta', color='status',
                color_discrete_map={'Good (Efficient)': '#22c55e', 'Problem (Expensive)': '#ef4444', 'Normal': '#94a3b8'},
                title="Spend vs. CPA (The 'Elbow' Curve)",
                labels={
                    'ads_spend': 'Total Weekly Spend (Rp)', 
                    'unique_cpa': 'CPA - Cost Per Acquisition (Rp)',
                    'ads_unique_cta': 'Leads Volume',
                    'status': 'Status'
                },
                text='chart_label'
            )
            
            # Visual Threshold
            fig_scatter.add_vline(x=2000000000, line_dash="dash", line_color="black", annotation_text="Risk Threshold (2B)", annotation_position="top left")
            fig_scatter.update_traces(textposition='top center')
            fig_scatter.update_layout(showlegend=True)
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Dynamic Calc
            high_spend_df = df[df['ads_spend'] > 2e9]
            low_spend_df = df[df['ads_spend'] <= 2e9]
            
            if not high_spend_df.empty:
                avg_cpa_high = high_spend_df['unique_cpa'].mean()
                avg_cpa_low = low_spend_df['unique_cpa'].mean()
                diff_pct = ((avg_cpa_high - avg_cpa_low) / avg_cpa_low) * 100
                st.info(f"👉 **Impact:** Weeks with >2B Spend cost **{diff_pct:.0f}% more** per student on average.")

        # --- INSIGHT B: CONVERSION RATE DROP ---
        with col_insight_2:
            st.subheader("B. The 'Intent' Drop")
            st.markdown("""
            **Why does scaling fail?**
            Because the **Conversion Rate** drops.
            
            This proves we are bringing in "lower quality" traffic. These users click the ad but are far less likely to sign up than our core audience.
            """)
            
            avg_cvr_good = top_5_good['unique_cvr'].mean()
            avg_cvr_bad = top_5_bad['unique_cvr'].mean()
            
            fig_bar_cvr = go.Figure(data=[
                go.Bar(name='Success Weeks', x=['Success Weeks'], y=[avg_cvr_good], marker_color='#22c55e', text=[f"{avg_cvr_good:.1f}%"], textposition='auto'),
                go.Bar(name='Problem Weeks', x=['Problem Weeks'], y=[avg_cvr_bad], marker_color='#ef4444', text=[f"{avg_cvr_bad:.1f}%"], textposition='auto')
            ])
            fig_bar_cvr.update_layout(title="Conversion Rate: Success vs Problem Weeks", yaxis_title="Conversion Rate (%)")
            st.plotly_chart(fig_bar_cvr, use_container_width=True)

    # === TAB 4: RECOMMENDATIONS (RISK ANALYSIS ADDED) ===
    with tab4:
        st.header("4. Strategic Recommendations")
        st.markdown("Strategy adjustments based on the saturation point analysis.")
        
        st.divider()

        # --- RECOMMENDATION 1: RISK ANALYSIS ---
        c1_text, c1_chart = st.columns([1, 1])
        
        # Calculate Risk Ratio
        df['spend_bucket'] = df['ads_spend'].apply(lambda x: 'Safe Zone (<2B)' if x < 2000000000 else 'High Risk (>2B)')
        
        # Calculate % of "Problem" weeks in each bucket
        risk_data = df.groupby('spend_bucket')['status'].value_counts(normalize=True).unstack().fillna(0)
        risk_data = risk_data * 100 # Convert to percentage
        
        # Ensure 'Problem (Expensive)' column exists
        prob_bad_low = risk_data.loc['Safe Zone (<2B)', 'Problem (Expensive)'] if 'Problem (Expensive)' in risk_data.columns else 0
        prob_bad_high = risk_data.loc['High Risk (>2B)', 'Problem (Expensive)'] if 'High Risk (>2B)' in risk_data.index and 'Problem (Expensive)' in risk_data.columns else 0
        
        with c1_text:
            st.subheader("1. 🛑 The 2B 'Soft Cap'")
            st.markdown(f"""
            **The Nuance:** "Can we spend more than 2B?"
            **The Answer:** Yes, but it is risky.
            
            **The Evidence:**
            * In the Safe Zone (<2B), "Problem Weeks" are rare (**{prob_bad_low:.1f}%** chance).
            * In the High Risk Zone (>2B), the chance of a "Problem Week" jumps to **{prob_bad_high:.1f}%**.
            
            **Recommendation:** Do not treat 2B as a wall, but as a **Checkpoint**. If you scale past 2B, you must monitor CPA daily.
            """)
        
        with c1_chart:
            # Stacked Bar for Risk
            fig_risk = px.bar(
                risk_data.reset_index(), 
                x='spend_bucket', 
                y=['Good (Efficient)', 'Normal', 'Problem (Expensive)'],
                title="Risk Profile: Probability of Failure by Spend Level",
                color_discrete_map={'Good (Efficient)': '#22c55e', 'Normal': '#cbd5e1', 'Problem (Expensive)': '#ef4444'},
                labels={'value': 'Probability (%)', 'variable': 'Outcome'}
            )
            st.plotly_chart(fig_risk, use_container_width=True)

        st.divider()

        # --- RECOMMENDATION 2 & 3 ---
        c2, c3 = st.columns(2)
        
        with c2:
            st.subheader("2. 🎯 Fix the 'Click Objective'")
            st.markdown("**Action:** Stop optimizing for 'Traffic'. Optimize for 'Leads'. The drop in conversion rate at high spend proves we are attracting low-intent users who click but don't buy.")
            
            # Manual Trendline for CVR
            fig_corr = px.scatter(
                df, x='ads_spend', y='unique_cvr',
                title="Correlation: Spend vs Conversion Rate",
                labels={'ads_spend': 'Ad Spend (Rp)', 'unique_cvr': 'Conversion Rate (%)'}
            )
            x_data = df['ads_spend']
            y_data = df['unique_cvr']
            if len(x_data) > 1:
                m, b = np.polyfit(x_data, y_data, 1)
                fig_corr.add_trace(go.Scatter(x=x_data, y=m * x_data + b, mode='lines', name='Trend'))
            st.plotly_chart(fig_corr, use_container_width=True)

        with c3:
            st.subheader("3. 📉 Reality Check: Total vs Unique")
            total_leads_raw = df['ads_cta'].sum()
            total_leads_unique = df['ads_unique_cta'].sum()
            
            st.markdown(f"""
            **Action:** Stop reporting 'Total Leads'. It is a vanity metric.
            
            **The Gap:** {total_leads_raw:,.0f} (Reported) vs {total_leads_unique:,.0f} (Actual People).
            We are over-reporting success by counting duplicates.
            """)
            
            fig_gap = go.Figure(data=[
                go.Bar(name='Reported', x=['Metrics'], y=[total_leads_raw], marker_color='#94a3b8'),
                go.Bar(name='Actual', x=['Metrics'], y=[total_leads_unique], marker_color='#2563eb')
            ])
            fig_gap.update_layout(title="The Inflation Gap", barmode='group')
            st.plotly_chart(fig_gap, use_container_width=True)

        st.caption("Generated for Sparks Edu Case Study | Data Source: 2019 Ads Export")

else:
    st.info("👋 Waiting for data. Please upload 'ads_data.csv' to your repository.")
