import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

class SoilQualityAnalyzer:
    """Analyze soil quality for different crop types."""
    
    def __init__(self):
        # Define optimal soil conditions for different crops
        self.crop_requirements = {
            'corn': {
                'ph_range': (5.8, 7.0),
                'organic_matter_range': (2.5, 5.0),
                'optimal_yield': 180,
                'description': 'Corn requires well-drained soil with good organic matter content.'
            },
            'soybeans': {
                'ph_range': (6.0, 7.5),
                'organic_matter_range': (2.0, 4.5),
                'optimal_yield': 50,  # bushels per acre
                'description': 'Soybeans prefer slightly acidic to neutral soil with moderate organic matter.'
            },
            'wheat': {
                'ph_range': (6.0, 7.5),
                'organic_matter_range': (1.5, 4.0),
                'optimal_yield': 60,  # bushels per acre
                'description': 'Wheat grows well in a wide range of soil conditions.'
            },
            'cotton': {
                'ph_range': (5.5, 8.5),
                'organic_matter_range': (1.0, 3.0),
                'optimal_yield': 800,  # pounds per acre
                'description': 'Cotton is tolerant of various soil conditions but prefers well-drained soil.'
            }
        }
    
    def analyze_soil_quality(self, soil_ph, organic_matter, crop_type='corn'):
        """Analyze soil quality for a specific crop."""
        if crop_type not in self.crop_requirements:
            return None
        
        requirements = self.crop_requirements[crop_type]
        
        # Calculate pH score (0-100)
        ph_min, ph_max = requirements['ph_range']
        if ph_min <= soil_ph <= ph_max:
            ph_score = 100
        else:
            # Calculate distance from optimal range
            if soil_ph < ph_min:
                distance = ph_min - soil_ph
            else:
                distance = soil_ph - ph_max
            ph_score = max(0, 100 - (distance * 20))
        
        # Calculate organic matter score (0-100)
        om_min, om_max = requirements['organic_matter_range']
        if om_min <= organic_matter <= om_max:
            om_score = 100
        else:
            if organic_matter < om_min:
                distance = om_min - organic_matter
            else:
                distance = organic_matter - om_max
            om_score = max(0, 100 - (distance * 25))
        
        # Overall soil quality score (average of pH and OM scores)
        overall_score = (ph_score + om_score) / 2
        
        # Determine quality category
        if overall_score >= 80:
            quality_category = "Excellent"
            recommendation = "Soil is ideal for this crop. Maintain current practices."
        elif overall_score >= 60:
            quality_category = "Good"
            recommendation = "Soil is suitable with minor improvements recommended."
        elif overall_score >= 40:
            quality_category = "Fair"
            recommendation = "Soil needs improvement for optimal crop performance."
        else:
            quality_category = "Poor"
            recommendation = "Significant soil amendments required before planting."
        
        return {
            'crop_type': crop_type,
            'ph_score': ph_score,
            'om_score': om_score,
            'overall_score': overall_score,
            'quality_category': quality_category,
            'recommendation': recommendation,
            'requirements': requirements
        }
    
    def analyze_all_crops(self, soil_ph, organic_matter):
        """Analyze soil quality for all available crops."""
        results = {}
        
        for crop in self.crop_requirements.keys():
            result = self.analyze_soil_quality(soil_ph, organic_matter, crop)
            if result:
                results[crop] = result
        
        return results
    
    def get_soil_recommendations(self, soil_ph, organic_matter):
        """Get specific soil improvement recommendations."""
        recommendations = []
        
        # pH recommendations
        if soil_ph < 5.5:
            recommendations.append({
                'issue': 'Low pH (Acidic Soil)',
                'solution': 'Apply agricultural lime to raise pH to 6.0-7.0',
                'priority': 'High'
            })
        elif soil_ph > 7.5:
            recommendations.append({
                'issue': 'High pH (Alkaline Soil)',
                'solution': 'Consider sulfur applications or acidifying fertilizers',
                'priority': 'Medium'
            })
        
        # Organic matter recommendations
        if organic_matter < 2.0:
            recommendations.append({
                'issue': 'Low Organic Matter',
                'solution': 'Add compost, manure, or cover crops to increase organic matter',
                'priority': 'High'
            })
        elif organic_matter > 5.0:
            recommendations.append({
                'issue': 'Very High Organic Matter',
                'solution': 'Monitor for potential nitrogen tie-up during decomposition',
                'priority': 'Low'
            })
        
        return recommendations
    
    def plot_soil_analysis(self, data, crop_type='corn'):
        """Create visualization for soil analysis."""
        # Analyze soil quality for each field
        soil_scores = []
        for _, row in data.iterrows():
            analysis = self.analyze_soil_quality(
                row['soil_ph'], 
                row['organic_matter_percent'], 
                crop_type
            )
            if analysis:
                soil_scores.append(analysis['overall_score'])
            else:
                soil_scores.append(0)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Soil Quality Analysis for {crop_type.title()}', fontsize=16)
        
        # 1. Soil Quality Distribution
        axes[0, 0].hist(soil_scores, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 0].set_xlabel('Soil Quality Score')
        axes[0, 0].set_ylabel('Number of Fields')
        axes[0, 0].set_title('Distribution of Soil Quality Scores')
        axes[0, 0].axvline(np.mean(soil_scores), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(soil_scores):.1f}')
        axes[0, 0].legend()
        
        # 2. pH vs Organic Matter scatter
        scatter = axes[0, 1].scatter(data['soil_ph'], data['organic_matter_percent'], 
                                   c=soil_scores, cmap='RdYlGn', s=50, alpha=0.7)
        axes[0, 1].set_xlabel('Soil pH')
        axes[0, 1].set_ylabel('Organic Matter (%)')
        axes[0, 1].set_title('Soil pH vs Organic Matter (colored by quality)')
        plt.colorbar(scatter, ax=axes[0, 1], label='Soil Quality Score')
        
        # 3. Quality categories
        quality_categories = []
        for score in soil_scores:
            if score >= 80:
                quality_categories.append('Excellent')
            elif score >= 60:
                quality_categories.append('Good')
            elif score >= 40:
                quality_categories.append('Fair')
            else:
                quality_categories.append('Poor')
        
        quality_counts = pd.Series(quality_categories).value_counts()
        axes[1, 0].pie(quality_counts.values, labels=quality_counts.index, autopct='%1.1f%%')
        axes[1, 0].set_title('Soil Quality Categories')
        
        # 4. Yield vs Soil Quality
        axes[1, 1].scatter(soil_scores, data['yield_bushels_per_acre'], alpha=0.6, color='blue')
        axes[1, 1].set_xlabel('Soil Quality Score')
        axes[1, 1].set_ylabel('Yield (bushels/acre)')
        axes[1, 1].set_title('Yield vs Soil Quality')
        
        # Add trend line
        z = np.polyfit(soil_scores, data['yield_bushels_per_acre'], 1)
        p = np.poly1d(z)
        axes[1, 1].plot(soil_scores, p(soil_scores), "r--", alpha=0.8)
        
        plt.tight_layout()
        return fig
    
    def create_soil_dashboard(self, data):
        """Create a comprehensive soil analysis dashboard."""
        st.subheader("🌱 Soil Quality Analysis Dashboard")
        
        # Crop selection
        crop_type = st.selectbox(
            "Select Crop for Analysis",
            list(self.crop_requirements.keys()),
            index=0
        )
        
        # Display crop requirements
        requirements = self.crop_requirements[crop_type]
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Optimal pH Range", f"{requirements['ph_range'][0]} - {requirements['ph_range'][1]}")
        with col2:
            st.metric("Optimal OM Range", f"{requirements['organic_matter_range'][0]} - {requirements['organic_matter_range'][1]}%")
        with col3:
            st.metric("Target Yield", f"{requirements['optimal_yield']} units/acre")
        
        st.info(requirements['description'])
        
        # Analyze all fields
        soil_scores = []
        quality_categories = []
        
        for _, row in data.iterrows():
            analysis = self.analyze_soil_quality(
                row['soil_ph'], 
                row['organic_matter_percent'], 
                crop_type
            )
            if analysis:
                soil_scores.append(analysis['overall_score'])
                quality_categories.append(analysis['quality_category'])
            else:
                soil_scores.append(0)
                quality_categories.append('Unknown')
        
        # Add scores to data
        analysis_data = data.copy()
        analysis_data['soil_quality_score'] = soil_scores
        analysis_data['quality_category'] = quality_categories
        
        # Display summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Average Soil Score", f"{np.mean(soil_scores):.1f}")
        with col2:
            st.metric("Best Score", f"{np.max(soil_scores):.1f}")
        with col3:
            st.metric("Worst Score", f"{np.min(soil_scores):.1f}")
        with col4:
            excellent_count = quality_categories.count('Excellent')
            st.metric("Excellent Fields", f"{excellent_count}")
        
        # Create visualization
        fig = self.plot_soil_analysis(data, crop_type)
        st.pyplot(fig)
        
        # Show detailed analysis for a sample field
        st.subheader("📋 Sample Field Analysis")
        sample_field = data.iloc[0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Field Information:**")
            st.write(f"- Field ID: {sample_field['field_id']}")
            st.write(f"- Soil pH: {sample_field['soil_ph']:.2f}")
            st.write(f"- Organic Matter: {sample_field['organic_matter_percent']:.2f}%")
            st.write(f"- Actual Yield: {sample_field['yield_bushels_per_acre']:.1f} bushels/acre")
        
        with col2:
            analysis = self.analyze_soil_quality(
                sample_field['soil_ph'],
                sample_field['organic_matter_percent'],
                crop_type
            )
            
            if analysis:
                st.write("**Soil Quality Assessment:**")
                st.write(f"- Overall Score: {analysis['overall_score']:.1f}/100")
                st.write(f"- Quality Category: {analysis['quality_category']}")
                st.write(f"- pH Score: {analysis['ph_score']:.1f}/100")
                st.write(f"- OM Score: {analysis['om_score']:.1f}/100")
        
        # Recommendations
        recommendations = self.get_soil_recommendations(
            sample_field['soil_ph'],
            sample_field['organic_matter_percent']
        )
        
        if recommendations:
            st.subheader("🔧 Soil Improvement Recommendations")
            for rec in recommendations:
                if rec['priority'] == 'High':
                    st.error(f"**{rec['issue']}** - {rec['solution']}")
                elif rec['priority'] == 'Medium':
                    st.warning(f"**{rec['issue']}** - {rec['solution']}")
                else:
                    st.info(f"**{rec['issue']}** - {rec['solution']}")
        
        return analysis_data

if __name__ == "__main__":
    # Test the soil analyzer
    analyzer = SoilQualityAnalyzer()
    
    # Test with sample data
    test_ph = 6.2
    test_om = 3.5
    
    print("Soil Quality Analysis Test")
    print("=" * 40)
    
    for crop in ['corn', 'soybeans', 'wheat', 'cotton']:
        result = analyzer.analyze_soil_quality(test_ph, test_om, crop)
        if result:
            print(f"\n{crop.upper()}:")
            print(f"  Overall Score: {result['overall_score']:.1f}/100")
            print(f"  Quality: {result['quality_category']}")
            print(f"  Recommendation: {result['recommendation']}")
    
    print("\nTest completed successfully!") 