import joblib
import pandas as pd
import numpy as np

class StrategyPredictor:
    def __init__(self, kmeans_path, scaler_path, data_path, proxy_path=None):
        # Memuat Locked Pipeline (K-Means + Scaler)
        self.kmeans = joblib.load(kmeans_path)
        self.scaler = joblib.load(scaler_path)
        self.raw_data = pd.read_csv(data_path)
        self.proxy_model = joblib.load(proxy_path) if proxy_path else None


        # Hitung statistik berdasarkan label teks yang ada di CSV
        # Hasil stats_raw kuncinya adalah: 'high acceptance chanel', dsb.
        stats_raw = self.raw_data.groupby('cluster').agg({
            'avg_cost_per_hire': 'mean',
            'avg_time_to_hire': 'mean',
            'avg_OAR': 'mean'
        }).to_dict('index')

        # BUAT JEMBATAN PEMETAAN (Mapping)
        # Kita hubungkan angka 0, 1, 2 ke label teks di CSV kamu
        label_map = {
            0: 'high acceptance chanel',
            1: 'fast/urgent hiring chanel',
            2: 'cost efficient chanel'
        }

        # Buat dictionary stats baru yang kuncinya adalah ANGKA (0, 1, 2)
        # agar bisa dibaca oleh strategy_map di bawah
        stats = {}
        for num, text_label in label_map.items():
            if text_label in stats_raw:
                stats[num] = stats_raw[text_label]
            else:
                # Fallback jika label tidak ditemukan agar tidak error
                stats[num] = {'avg_cost_per_hire': 0, 'avg_time_to_hire': 0, 'avg_OAR': 0}

        # 2. Narasi tetap "dititipkan" di sini (Versi Lama Kamu)
        self.strategy_map = {
            "impact_reasoning": {
                "cost": "Angka ini merupakan proyeksi pengeluaran per kandidat. Jika angka ini negatif (hijau), berarti strategi ini mendukung target efisiensi biaya 6% perusahaan.",
                "time": "Melambangkan durasi dari pembukaan lowongan hingga kontrak ditandatangani. Penurunan angka ini krusial untuk mencapai target percepatan 15%.",
                "oar": "Menunjukkan persentase kandidat yang menerima tawaran. Angka yang lebih tinggi dari baseline berarti strategi ini meningkatkan kualitas rekrutmen sebesar 7%."
            },
            0: {
                "cluster_id": 0,
                "icon": "🎯",
                "label": "High Acceptance Channel",
                "persona": "Kualitas & Kepastian Kandidat",
                "source": "Recruiter & Job Portal (Premium)",
                "reasoning": (
                    "Kanal ini menjadi prioritas utama ketika fokus Anda adalah mendapatkan kandidat dengan kualifikasi spesifik. "
                    "Berdasarkan data historis, penggunaan Recruiter (Headhunter) dan Premium Portal memberikan pendekatan yang lebih personal. "
                    f"Hal ini secara signifikan meningkatkan tingkat penerimaan tawaran (OAR) hingga {stats[0]['avg_OAR']:.1%}, yang berarti risiko kandidat menolak "
                    "tawaran (turnover di tahap akhir) sangat minim. Strategi ini memastikan investasi biaya rekrutmen Anda berbanding lurus "
                    "dengan kepastian mendapatkan talent terbaik."
                ),
                "action": "Prioritaskan Headhunting untuk posisi krusial. Fokus pada pengalaman kandidat untuk menjaga OAR tetap tinggi.",
                "color": "#28a745",
                # Ambil angka langsung dari stats CSV
                "metrics": {
                    "cost": stats[0]['avg_cost_per_hire'], 
                    "time": stats[0]['avg_time_to_hire'], 
                    "oar": stats[0]['avg_OAR']
                }
            },
            1: {
                "cluster_id": 1,
                "icon": "⚡",
                "label": "Fast/Urgent Hiring Channel",
                "persona": "Kecepatan Pemenuhan Posisi",
                "source": "LinkedIn Promoted & Job Portal",
                "reasoning": (
                    "Untuk mengejar target percepatan waktu (Reduce Time 15%), kombinasi LinkedIn Promoted dan Job Portal adalah kanal paling gesit. "
                    "Fitur iklan berbayar dan otomatisasi screening pada platform ini memungkinkan perusahaan menjangkau volume pelamar masif "
                    f"dalam waktu singkat. Secara data, strategi ini mampu memangkas waktu rekrutmen hingga rata-rata {stats[1]['avg_time_to_hire']:.1f} hari. "
                    "Meskipun ada tambahan budget iklan, efisiensi waktu yang didapat sangat krusial untuk mencegah kekosongan posisi "
                    "yang dapat menghambat operasional bisnis."
                ),
                "action": "Gunakan fitur 'Promoted' atau 'Urgent' pada portal. Percepat proses seleksi awal untuk mengejar deadline.",
                "color": "#ffc107",
                "metrics": {
                    "cost": stats[1]['avg_cost_per_hire'], 
                    "time": stats[1]['avg_time_to_hire'], 
                    "oar": stats[1]['avg_OAR']
                }
            },
            2: {
                "cluster_id": 2,
                "icon": "💰",
                "label": "Cost Efficient Channel",
                "persona": "Optimasi Anggaran Rekrutmen",
                "source": "Referral Internal & LinkedIn Organic",
                "reasoning": (
                    "Strategi ini adalah kunci utama untuk mencapai target penghematan biaya (Reduce Cost 6%). "
                    "Kanal Referral Internal dan LinkedIn Organik sangat direkomendasikan karena meminimalisir ketergantungan pada vendor "
                    "eksternal atau iklan berbayar yang mahal. Dengan mengandalkan jaringan internal, perusahaan dapat menekan biaya hingga "
                    f"ke angka ${stats[2]['avg_cost_per_hire']:,.0f} per hire. Ini adalah jalur paling optimal untuk rekrutmen volume besar dengan tetap menjaga stabilitas "
                    "kualitas kandidat melalui rekomendasi karyawan yang sudah terpercaya."
                ),
                "action": "Aktifkan program insentif referral karyawan. Gunakan konten branding organik di media sosial untuk menarik pelamar.",
                "color": "#17a2b8",
                "metrics": {
                    "cost": stats[2]['avg_cost_per_hire'], 
                    "time": stats[2]['avg_time_to_hire'], 
                    "oar": stats[2]['avg_OAR']
                }
            }
        }

    def predict_cluster(self, t, c, o):
        # WAJIB: Scaling input sebelum masuk ke K-Means
        scaled_input = self.scaler.transform([[t, c, o]])
        cluster_idx = int(self.kmeans.predict(scaled_input)[0])
        return self.strategy_map[cluster_idx], cluster_idx

    def get_job_titles(self):
        return sorted(self.raw_data['job_title'].unique())

    def get_historical_stats(self, job_title):
        # Karena data mentah punya banyak baris untuk satu jabatan, 
        # kita harus memfilter jabatan tersebut lalu ambil rata-ratanya.
        job_data = self.raw_data[self.raw_data['job_title'] == job_title]
        
        avg_t = job_data['avg_time_to_hire'].mean()
        avg_c = job_data['avg_cost_per_hire'].mean()
        avg_o = job_data['avg_OAR'].mean()
        
        return avg_t, avg_c, avg_o

    def get_impact_explanations(self):
        return self.strategy_map["impact_reasoning"]

    def get_label_name(self, cluster_idx):
        # Jembatan pemetaan balik dari angka ke teks dataset
        label_map = {
            0: 'high acceptance chanel',
            1: 'fast/urgent hiring chanel',
            2: 'cost efficient chanel'
        }
        return label_map.get(cluster_idx, "Unknown")

    def get_strategy_details(self, cluster_idx):
        # Mengambil detail dari strategy_map yang sudah kita buat
        return self.strategy_map.get(cluster_idx)