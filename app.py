import streamlit as st
from src.model_handler import StrategyPredictor
from src.ui_components import render_sidebar, display_strategy_card, display_metrics_comparison, display_model_validation

def main():
    st.set_page_config(page_title="Analytica HR Advisor", layout="wide")
    st.markdown(
        """
        <div style="
            background-color: #3498db; /* biru cerah */
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        ">
            <h1 style="color: white; margin: 0;">Analytica HR Advisor</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<h2 style='text-align: center; color: white;'>🎯 Recruitment Strategic Advisor</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: white;'>Optimasi strategi rekrutmen berbasis AI untuk mencapai target efisiensi perusahaan.</p>", unsafe_allow_html=True)
    #st.write("Optimasi strategi rekrutmen berbasis AI untuk mencapai target efisiensi perusahaan.")
    

    # Inisialisasi Predictor
    predictor = StrategyPredictor(
        kmeans_path='models/kmeans_model.pkl',
        scaler_path='models/scaler.pkl',
        data_path='data/datasets_final_cluster.csv',
        proxy_path='models/proxy_model.pkl'
    )

    # Sidebar Input
    mode, inputs, baseline, selected_job = render_sidebar(predictor)

    tab1, tab2 = st.tabs(["🚀 Strategy Recommender", "🧪 Model Validation (Proxy Check)"])

    with tab1:
        if st.button("Generate Strategic Analysis"):
            strategy, cluster_idx = predictor.predict_cluster(*inputs)

            # Ambil nama jabatan jika modenya Role-Based
            role_name = selected_job if mode == "Role-Based Optimization" else None

            # Ambil penjelasan dampak secara terpisah
            impact_info = predictor.get_impact_explanations()
            
            # Tampilkan kartu strategi dengan narasi penjelasan lengkap
            display_strategy_card(strategy, selected_role=role_name)
            
            # Tampilkan Perbandingan Metrik
            st.divider()
            # Tampilkan Metrik & Penjelasan Dampak
            display_metrics_comparison(strategy['metrics'], baseline, impact_info)
            
            # ROI Insight
            st.divider()
            saving = baseline[1] - strategy['metrics']['cost']
            if saving > 0:
                st.success(f"💰 **Financial Impact:** Strategi ini berpotensi menghemat anggaran sebesar **${saving:.0f} per hire**.")
            else:
                st.warning(f"⚠️ **Investment Note:** Strategi ini memerlukan alokasi budget tambahan sebesar **${abs(saving):.0f}** untuk prioritas kecepatan/kualitas.")

    with tab2:
        st.subheader("🧪 Model Validation (Proxy Check)")
        st.info("Validasi ini memastikan apakah K-Means (Unsupervised) konsisten dengan pola Klasifikasi (Supervised).")
        
        if st.button("Jalankan Validasi Model"):
            # 1. Logic Processing
            strategy_main, cluster_idx = predictor.predict_cluster(*inputs)
            label_kmeans = predictor.get_label_name(cluster_idx)
            
            proxy_idx = int(predictor.proxy_model.predict([list(inputs)])[0])
            label_proxy = predictor.get_label_name(proxy_idx)
            strategy_proxy = predictor.get_strategy_details(proxy_idx)
            
            # 2. UI Rendering (Memanggil fungsi dari ui_components)
            display_model_validation(strategy_main, label_kmeans, strategy_proxy, label_proxy)

if __name__ == "__main__":
    main()