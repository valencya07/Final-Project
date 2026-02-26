import streamlit as st

def render_sidebar(predictor):
    st.sidebar.header("🕹️ Strategic Planner")
    mode = st.sidebar.radio("Pilih Mode:", ["General Simulation", "Role-Based Optimization"])
    
    baseline = (47.0, 5214.0, 0.65)
    selected_job = None # Inisialisasi awal

    if mode == "General Simulation":
        st.sidebar.subheader("Input Target Baru")
        t = st.sidebar.number_input("Target Deadline (Hari)", 10, 100, 47)
        c = st.sidebar.number_input("Target Budget ($)", 1000, 10000, 5214)
        o = st.sidebar.number_input("Target OAR", 0.0, 1.0, 0.65)
        return mode, (t, c, o), baseline, selected_job
    
    else:
        st.sidebar.subheader("Optimasi Per Jabatan")
        selected_job = st.sidebar.selectbox("Pilih Posisi:", predictor.get_job_titles())
        h_t, h_c, h_o = predictor.get_historical_stats(selected_job)
        
        st.sidebar.divider()
        t = st.sidebar.slider("Adjust Deadline", 10, 100, int(h_t))
        c = st.sidebar.slider("Adjust Budget", 1000, 10000, int(h_c))
        o = st.sidebar.slider("Adjust OAR", 0.0, 1.0, float(h_o))
        return mode, (t, c, o), (h_t, h_c, h_o), selected_job

def display_strategy_card(strategy, selected_role=None):
    header_text = f"Strategi Rekomendasi: {selected_role}" if selected_role else "Rekomendasi Strategi Utama"
    
    st.markdown(f"""
        <div style="border-radius:15px; padding:20px; background-color:{strategy['color']}15; border: 2px solid {strategy['color']}; margin-bottom: 10px">
            <h4 style="color:{strategy['color']}; margin-top:0">{header_text}</h4>
            <h2 style="color:{strategy['color']}; margin-top:0">{strategy['icon']} {strategy['label']}</h2>
            <p><b>Kanal Prioritas:</b> {strategy['source']}</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Fitur Tombol Lihat Penjelasan
    with st.expander("🔍 Lihat Penjelasan Strategi"):
        st.info(f"**Alasan Prioritas:**\n\n{strategy['reasoning']}")
        st.success(f"**Langkah Taktis:** {strategy['action']}")

def display_metrics_comparison(current_metrics, baseline_data, impact_info):
    st.subheader("📊 Analisis Dampak Metrik")
    col1, col2, col3 = st.columns(3)
    
    # Unpack data baseline (Time, Cost, OAR)
    b_time, b_cost, b_oar = baseline_data

    # Target Kelompok (Business Success Metrics)
    target_cost = 4900  # Target Reduce Cost 6%
    target_time = 40    # Target Reduce Time 15%
    target_oar = 0.72   # Target Increase OAR 7%

    # Perhitungan Delta
    c_diff = current_metrics['cost'] - baseline_data[1]
    t_diff = current_metrics['time'] - baseline_data[0]
    o_diff = (current_metrics['oar'] - baseline_data[2]) * 100

    col1.metric("Estimated Cost", f"${current_metrics['cost']:,.0f}", f"{c_diff:.0f}", delta_color="inverse")
    col2.metric("Estimated Time", f"{current_metrics['time']:,.0f} Days", f"{t_diff:.1f} Days", delta_color="inverse")
    col3.metric("Estimated OAR", f"{current_metrics['oar']:.2%}", f"{o_diff:.1f}%")

    # Expander Detail
    with st.expander("💡 Bagaimana cara membaca angka ini?"):
        st.markdown(f"""
        ### 🔍 Asal Angka Estimasi
        Angka **Estimated Cost/Time/OAR** yang muncul di atas diambil dari **Profil Klaster (Cluster Centroids)** hasil riset *Stage 3*. 
        Ini melambangkan performa rata-rata historis dari kelompok strategi tersebut. Jadi, jika Anda mengikuti strategi ini, angka inilah yang secara statistik paling mungkin Anda capai.
        
        ---
        ### 🟢 Logika Indikator Warna:
        Indikator naik/turun dihitung berdasarkan perbandingan dengan **Baseline Perusahaan** (Kondisi saat ini):
        * **Warna Hijau:** Menunjukkan **perbaikan** (Biaya lebih hemat, waktu lebih cepat, atau OAR lebih tinggi).
        * **Warna Merah:** Menunjukkan **peningkatan/penurunan** yang perlu diperhatikan (Misal: biaya naik demi mengejar kecepatan).
        
        ---
        ### 🎯 Status Target Perusahaan (Success Metrics)
        Berikut adalah perbandingan hasil prediksi terhadap target KPI kelompok:
        * **Target Cost:** `${target_cost:,.0f}` (Gap: `${current_metrics['cost'] - target_cost:,.0f}`)
        * **Target Time:** `{target_time} Hari` (Gap: `{current_metrics['time'] - target_time:.1f} Hari`)
        * **Target OAR:** `{target_oar:.1%}` (Gap: `{(target_oar - current_metrics['oar'])*100:.1f}%`)
        
        ---
        ### 🏢 Baseline Perusahaan (Kondisi Saat Ini)
        Indikator warna dihitung berdasarkan perbandingan dengan kondisi rata-rata perusahaan saat ini:
        * **Baseline Cost:** `${b_cost:,.0f}`
        * **Baseline Time:** `{b_time} Hari`
        * **Baseline OAR:** `{b_oar:.1%}`
        
        ---
        ### 📖 Penjelasan Detail:
        * **Cost Impact:** {impact_info['cost']}
        * **Time Impact:** {impact_info['time']}
        * **OAR Impact:** {impact_info['oar']}
        """)

def display_model_validation(strategy_main, label_main, strategy_proxy, label_proxy):
    # 1. Tampilan Kolom Perbandingan
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🤖 K-Means Clustering")
        st.success(f"**Cluster:** {label_main.title()}")
        st.caption(f"Persona: {strategy_main['persona']}")
        
    with col2:
        st.markdown("### 🛡️ Proxy Classification")
        if label_proxy == label_main:
            st.success(f"**Cluster:** {label_proxy.title()}")
        else:
            st.warning(f"**Cluster:** {label_proxy.title()}")
        st.caption(f"Persona: {strategy_proxy['persona']}")

    st.divider()

    # 2. Detail Penjelasan Cluster
    st.markdown(f"### 📝 Analisis Detail: {label_main.title()}")
    
    m1, m2, m3 = st.columns(3)
    m1.write(f"**Est. Cost:** ${strategy_main['metrics']['cost']:.0f}")
    m2.write(f"**Est. Time:** {strategy_main['metrics']['time']:.1f} Hari")
    m3.write(f"**Est. OAR:** {strategy_main['metrics']['oar']:.1%}")

    with st.expander(f"🔍 Lihat Detail Strategi {label_main.title()}"):
        st.info(f"**Alasan Prioritas:**\n\n{strategy_main['reasoning']}")
        st.write(f"**Langkah Taktis:** {strategy_main['action']}")
        st.write(f"**Kanal Utama:** {strategy_main['source']}")

    # 3. Kesimpulan Validasi
    st.write("---")
    if label_proxy == label_main:
        st.balloons()
        st.success("✅ **Hasil Konsisten:** Kedua model menyetujui klasifikasi ini. Model memiliki tingkat kepercayaan yang tinggi.")
    else:
        st.warning("⚠️ **Hasil Berbeda:** Input berada di area perbatasan antar cluster. Disarankan untuk meninjau kembali variabel input.")