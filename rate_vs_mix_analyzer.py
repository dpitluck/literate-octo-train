import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional
import io

# Set page config
st.set_page_config(
    page_title="Mix vs Rate Analysis Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class GeneralizedMixRateAnalyzer:
    """
    Generalized Mix vs Rate analyzer that works with any metrics and target variable.
    """
    
    def __init__(self, df: pd.DataFrame, period_col: str, segment_col: str, 
                 volume_col: str, target_col: str):
        """
        Initialize the analyzer.
        
        Args:
            df: Input DataFrame
            period_col: Column representing time periods (e.g., 'Year', 'Quarter')
            segment_col: Column representing segments (e.g., 'Product', 'Region')
            volume_col: Column representing volume/size metric (e.g., 'Revenue', 'Units')
            target_col: Column representing the target metric to analyze (e.g., 'Margin%', 'Price')
        """
        self.df = df.copy()
        self.period_col = period_col
        self.segment_col = segment_col
        self.volume_col = volume_col
        self.target_col = target_col
        
        # Determine if target is a percentage metric that should use Revenue for weighting
        self.is_percentage_metric = (
            'percent' in target_col.lower() or 
            'pct' in target_col.lower() or 
            '%' in target_col.lower() or
            'margin' in target_col.lower() or
            'rate' in target_col.lower()
        )
        
        # For percentage metrics, try to find a revenue-like column for consistent weighting
        self.weighting_col = volume_col  # Default
        if self.is_percentage_metric:
            revenue_cols = [col for col in df.columns if 'revenue' in col.lower() or 'sales' in col.lower()]
            if revenue_cols:
                self.weighting_col = revenue_cols[0]  # Use first revenue-like column
                
        self._validate_and_clean_data()
        
        self.periods = sorted(self.df[period_col].unique())
        self.segments = sorted([s for s in self.df[segment_col].unique() if str(s).lower() != 'total'])
        
    def _validate_and_clean_data(self):
        """Validate and clean the input data."""
        # Check required columns exist
        required_cols = [self.period_col, self.segment_col, self.volume_col, self.target_col]
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert to numeric where needed
        numeric_cols = [self.volume_col, self.target_col]
        for col in numeric_cols:
            # Remove formatting if string
            if self.df[col].dtype == 'object':
                self.df[col] = self.df[col].astype(str).str.replace(r'[$,%\s]', '', regex=True)
                self.df[col] = self.df[col].str.replace('', '0')
            
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Remove rows with missing critical data
        initial_rows = len(self.df)
        self.df = self.df.dropna(subset=numeric_cols)
        removed_rows = initial_rows - len(self.df)
        
        if removed_rows > 0:
            st.warning(f"Removed {removed_rows} rows with missing data")
        
        if len(self.df[self.period_col].unique()) < 2:
            raise ValueError("Need at least 2 periods for comparison")
    
    def calculate_volume_weights(self, period) -> Dict[str, float]:
        """Calculate volume weights for a given period."""
        period_data = self.df[self.df[self.period_col] == period]
        segments_data = period_data[period_data[self.segment_col].str.lower() != 'total']
        
        total_volume = segments_data[self.volume_col].sum()
        weights = {}
        
        for segment in self.segments:
            seg_data = segments_data[segments_data[self.segment_col] == segment]
            if not seg_data.empty and total_volume > 0:
                weights[segment] = (seg_data[self.volume_col].iloc[0] / total_volume) * 100
            else:
                weights[segment] = 0
        
        return weights
    
    def calculate_total_target_metric(self, period) -> float:
        """Calculate volume-weighted total target metric for a given period."""
        period_data = self.df[self.df[self.period_col] == period]
        segments_data = period_data[period_data[self.segment_col].str.lower() != 'total']
        
        if segments_data.empty:
            return 0
        
        # Use consistent weighting column for percentage metrics
        weighting_col = self.weighting_col if hasattr(self, 'weighting_col') else self.volume_col
        
        total_volume = segments_data[weighting_col].sum()
        if total_volume == 0:
            return 0
        
        weighted_sum = (segments_data[weighting_col] * segments_data[self.target_col]).sum()
        return weighted_sum / total_volume
    
    def calculate_mix_rate_effects(self) -> Dict:
        """Perform mix vs rate effect decomposition."""
        if len(self.periods) < 2:
            raise ValueError("Need at least 2 periods for comparison")
        
        period_1, period_2 = self.periods[0], self.periods[1]
        
        # Get weights and target metrics for both periods
        weights_1 = self.calculate_volume_weights(period_1)
        weights_2 = self.calculate_volume_weights(period_2)
        
        target_1 = self.calculate_total_target_metric(period_1)
        target_2 = self.calculate_total_target_metric(period_2)
        
        # Calculate period 2 target with period 1 mix
        target_2_with_1_mix = 0
        for segment in self.segments:
            try:
                seg_data_2 = self.df[(self.df[self.period_col] == period_2) & 
                                   (self.df[self.segment_col] == segment)]
                if not seg_data_2.empty:
                    weight_1 = weights_1.get(segment, 0) / 100
                    target_2_with_1_mix += weight_1 * seg_data_2[self.target_col].iloc[0]
            except:
                continue
        
        # Calculate effects
        rate_effect = target_1 - target_2_with_1_mix
        mix_effect = target_2_with_1_mix - target_2
        total_change = target_2 - target_1
        
        return {
            'period_1': period_1,
            'period_2': period_2,
            'target_period_1': target_1,
            'target_period_2': target_2,
            'target_period_2_with_period_1_mix': target_2_with_1_mix,
            'rate_effect': rate_effect,
            'mix_effect': mix_effect,
            'total_change': total_change,
            'weights_period_1': weights_1,
            'weights_period_2': weights_2
        }
    
    def calculate_segment_impacts(self) -> pd.DataFrame:
        """Calculate individual segment contributions."""
        period_1, period_2 = self.periods[0], self.periods[1]
        mix_rate = self.calculate_mix_rate_effects()
        
        impacts = []
        
        for segment in self.segments:
            try:
                seg_1 = self.df[(self.df[self.period_col] == period_1) & 
                               (self.df[self.segment_col] == segment)]
                seg_2 = self.df[(self.df[self.period_col] == period_2) & 
                               (self.df[self.segment_col] == segment)]
                
                if seg_1.empty or seg_2.empty:
                    continue
                
                weight_1 = mix_rate['weights_period_1'][segment] / 100
                weight_2 = mix_rate['weights_period_2'][segment] / 100
                target_1 = seg_1[self.target_col].iloc[0]
                target_2 = seg_2[self.target_col].iloc[0]
                
                # Mix effect: (new_weight - old_weight) * old_target
                mix_effect = (weight_2 - weight_1) * target_1
                
                # Rate effect: new_weight * (new_target - old_target)
                rate_effect = weight_2 * (target_2 - target_1)
                
                total_impact = mix_effect + rate_effect
                volume_growth = ((seg_2[self.volume_col].iloc[0] / seg_1[self.volume_col].iloc[0]) - 1) * 100
                
                impacts.append({
                    'Segment': segment,
                    f'Weight_{period_1}': weight_1 * 100,
                    f'Weight_{period_2}': weight_2 * 100,
                    f'{self.target_col}_{period_1}': target_1,
                    f'{self.target_col}_{period_2}': target_2,
                    f'{self.target_col}_Change': target_2 - target_1,
                    f'{self.volume_col}_Growth': volume_growth,
                    'Mix_Effect': mix_effect,
                    'Rate_Effect': rate_effect,
                    'Total_Impact': total_impact
                })
                
            except Exception as e:
                st.warning(f"Could not process segment {segment}: {e}")
                continue
        
        return pd.DataFrame(impacts)


def create_plotly_charts(analyzer: GeneralizedMixRateAnalyzer):
    """Create interactive Plotly charts."""
    mix_rate = analyzer.calculate_mix_rate_effects()
    segment_impacts = analyzer.calculate_segment_impacts()
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Volume Mix Comparison', 
            f'{analyzer.target_col} by Segment',
            'Waterfall Analysis',
            'Segment Impact Breakdown'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 1. Volume Mix Comparison (Pie charts side by side)
    period_1, period_2 = analyzer.periods[0], analyzer.periods[1]
    
    # Get mix data for both periods
    segments = list(mix_rate['weights_period_1'].keys())
    weights_1 = [mix_rate['weights_period_1'][s] for s in segments]
    weights_2 = [mix_rate['weights_period_2'][s] for s in segments]
    
    # Create bar chart for mix comparison
    fig.add_trace(
        go.Bar(name=str(period_1), x=segments, y=weights_1, marker_color='lightblue'),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(name=str(period_2), x=segments, y=weights_2, marker_color='orange'),
        row=1, col=1
    )
    
    # 2. Target metric by segment
    if not segment_impacts.empty:
        target_1_col = f'{analyzer.target_col}_{period_1}'
        target_2_col = f'{analyzer.target_col}_{period_2}'
        
        fig.add_trace(
            go.Bar(name=f'{analyzer.target_col} {period_1}', 
                   x=segment_impacts['Segment'], 
                   y=segment_impacts[target_1_col], 
                   marker_color='green'),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(name=f'{analyzer.target_col} {period_2}', 
                   x=segment_impacts['Segment'], 
                   y=segment_impacts[target_2_col], 
                   marker_color='red'),
            row=1, col=2
        )
    
    # 3. Waterfall Chart
    categories = [f'{period_1} Actual', 'Rate Effect', 'Mix Effect', f'{period_2} Actual']
    values = [
        mix_rate['target_period_1'],
        -mix_rate['rate_effect'],
        -mix_rate['mix_effect'],
        mix_rate['target_period_2']
    ]
    colors = ['green', 'red', 'orange', 'blue']
    
    fig.add_trace(
        go.Bar(x=categories, y=values, marker_color=colors, showlegend=False),
        row=2, col=1
    )
    
    # 4. Segment Impact Breakdown
    if not segment_impacts.empty:
        fig.add_trace(
            go.Bar(name='Mix Effect', 
                   x=segment_impacts['Segment'], 
                   y=segment_impacts['Mix_Effect'], 
                   marker_color='orange'),
            row=2, col=2
        )
        fig.add_trace(
            go.Bar(name='Rate Effect', 
                   x=segment_impacts['Segment'], 
                   y=segment_impacts['Rate_Effect'], 
                   marker_color='red'),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(height=800, showlegend=True, title_text="Mix vs Rate Analysis Dashboard")
    fig.update_yaxes(title_text="Mix %", row=1, col=1)
    fig.update_yaxes(title_text=analyzer.target_col, row=1, col=2)
    fig.update_yaxes(title_text=analyzer.target_col, row=2, col=1)
    fig.update_yaxes(title_text="Impact", row=2, col=2)
    
    return fig


def main():
    """Main Streamlit app."""
    st.title("📊 Mix vs Rate Analysis Tool")
    st.markdown("Upload your data and analyze the drivers of change in any metric!")
    
    # Sidebar for configuration
    st.sidebar.header("📋 Configuration")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file", 
        type=['csv'],
        help="Upload a CSV file with your data"
    )
    
    # Sample data option
    if st.sidebar.button("🔄 Use Sample Data"):
        sample_data = pd.DataFrame([
            {'Year': 2024, 'Segment': 'Sale', 'Revenue': 2460526, 'COGS': 869007, 'GM_Percent': 64.7},
            {'Year': 2025, 'Segment': 'Sale', 'Revenue': 2692547, 'COGS': 1060079, 'GM_Percent': 60.6},
            {'Year': 2024, 'Segment': 'NonSale', 'Revenue': 3298748, 'COGS': 974425, 'GM_Percent': 70.5},
            {'Year': 2025, 'Segment': 'NonSale', 'Revenue': 1929408, 'COGS': 589430, 'GM_Percent': 69.5},
            {'Year': 2024, 'Segment': 'NCI', 'Revenue': 537871, 'COGS': 168793, 'GM_Percent': 68.6},
            {'Year': 2025, 'Segment': 'NCI', 'Revenue': 700922, 'COGS': 235510, 'GM_Percent': 66.4}
        ])
        st.session_state['df'] = sample_data
        st.sidebar.success("Sample data loaded!")
    
    # Load data
    df = None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state['df'] = df
            st.sidebar.success(f"File uploaded! {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")
    
    # Check if we have data
    if 'df' not in st.session_state:
        st.info("👆 Please upload a CSV file or use sample data to get started")
        
        # Show expected format
        st.subheader("📄 Expected Data Format")
        st.markdown("""
        Your CSV should have these types of columns:
        - **Period Column**: Time periods (e.g., Year, Quarter, Month)
        - **Segment Column**: Categories to analyze (e.g., Product, Region, Customer Type)
        - **Volume Column**: Size/volume metric (e.g., Revenue, Units Sold, Customers)
        - **Target Column**: Metric to analyze changes (e.g., Margin%, Price, Conversion Rate)
        """)
        
        example_df = pd.DataFrame([
            {'Year': 2023, 'Product': 'A', 'Revenue': 1000000, 'Margin_Pct': 25.5},
            {'Year': 2024, 'Product': 'A', 'Revenue': 1200000, 'Margin_Pct': 23.8},
            {'Year': 2023, 'Product': 'B', 'Revenue': 800000, 'Margin_Pct': 30.2},
            {'Year': 2024, 'Product': 'B', 'Revenue': 900000, 'Margin_Pct': 28.1}
        ])
        st.dataframe(example_df)
        return
    
    df = st.session_state['df']
    
    # Show data preview
    st.subheader("📊 Data Preview")
    st.dataframe(df.head())
    
    # Column selection
    st.sidebar.subheader("🔧 Column Mapping")
    
    columns = df.columns.tolist()
    
    period_col = st.sidebar.selectbox(
        "Period Column (Time)", 
        columns,
        help="Column representing time periods (e.g., Year, Quarter)"
    )
    
    segment_col = st.sidebar.selectbox(
        "Segment Column (Categories)", 
        columns,
        index=1 if len(columns) > 1 else 0,
        help="Column representing segments to analyze"
    )
    
    volume_col = st.sidebar.selectbox(
        "Volume Column (Size metric)", 
        columns,
        index=2 if len(columns) > 2 else 0,
        help="Column representing volume/size (e.g., Revenue, Units)"
    )
    
    target_col = st.sidebar.selectbox(
        "Target Column (Metric to analyze)", 
        columns,
        index=3 if len(columns) > 3 else 0,
        help="Column representing the metric to analyze changes"
    )
    
    # Validate selections
    selected_cols = [period_col, segment_col, volume_col, target_col]
    if len(set(selected_cols)) != 4:
        st.sidebar.error("Please select 4 different columns")
        return
    
    # Analysis button
    if st.sidebar.button("🚀 Run Analysis", type="primary"):
        try:
            with st.spinner("Running analysis..."):
                # Create analyzer
                analyzer = GeneralizedMixRateAnalyzer(
                    df, period_col, segment_col, volume_col, target_col
                )
                
                # Show weighting info for percentage metrics
                if hasattr(analyzer, 'is_percentage_metric') and analyzer.is_percentage_metric:
                    if analyzer.weighting_col != volume_col:
                        st.info(f"📊 **Note**: For percentage metric '{target_col}', using '{analyzer.weighting_col}' for consistent weighting instead of '{volume_col}'. This ensures the target metric calculation remains stable regardless of volume column selection.")
                
                # Store in session state
                st.session_state['analyzer'] = analyzer
                st.sidebar.success("Analysis complete!")
                
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            return
    
    # Show results if analysis is done
    if 'analyzer' in st.session_state:
        analyzer = st.session_state['analyzer']
        
        # Results
        st.header("📈 Analysis Results")
        
        # Key metrics
        mix_rate = analyzer.calculate_mix_rate_effects()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                f"{target_col} Change", 
                f"{mix_rate['total_change']:.2f}",
                delta=f"{mix_rate['total_change']:.2f}"
            )
        
        with col2:
            st.metric(
                "Rate Effect", 
                f"{mix_rate['rate_effect']:.2f}",
                delta=f"{mix_rate['rate_effect']:.2f}"
            )
        
        with col3:
            st.metric(
                "Mix Effect", 
                f"{mix_rate['mix_effect']:.2f}",
                delta=f"{mix_rate['mix_effect']:.2f}"
            )
        
        with col4:
            primary_driver = "Rate" if abs(mix_rate['rate_effect']) > abs(mix_rate['mix_effect']) else "Mix"
            st.metric("Primary Driver", primary_driver)
        
        # Interactive charts
        st.subheader("📊 Interactive Dashboard")
        fig = create_plotly_charts(analyzer)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed results
        st.subheader("🎯 Segment Impact Analysis")
        segment_impacts = analyzer.calculate_segment_impacts()
        
        if not segment_impacts.empty:
            st.dataframe(segment_impacts.round(2))
            
            # Download button
            csv_buffer = io.StringIO()
            segment_impacts.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv_buffer.getvalue(),
                file_name="mix_rate_analysis_results.csv",
                mime="text/csv"
            )
        
        # Insights
        st.subheader("💡 Key Insights")
        
        if not segment_impacts.empty:
            positive_segments = segment_impacts[segment_impacts['Total_Impact'] > 0]
            negative_segments = segment_impacts[segment_impacts['Total_Impact'] < 0]
            
            if not positive_segments.empty:
                st.success("**Segments that helped:**")
                for _, row in positive_segments.iterrows():
                    impact = row['Total_Impact']
                    contribution_pct = (impact / mix_rate['total_change']) * 100 if mix_rate['total_change'] != 0 else 0
                    st.write(f"• {row['Segment']}: {impact:+.2f} ({contribution_pct:.1f}% offset)")
            
            if not negative_segments.empty:
                st.error("**Segments that hurt:**")
                for _, row in negative_segments.iterrows():
                    impact = row['Total_Impact']
                    contribution_pct = (abs(impact) / abs(mix_rate['total_change'])) * 100 if mix_rate['total_change'] != 0 else 0
                    st.write(f"• {row['Segment']}: {impact:+.2f} ({contribution_pct:.1f}% of decline)")


if __name__ == "__main__":
    main()
