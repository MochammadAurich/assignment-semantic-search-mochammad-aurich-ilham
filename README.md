# Semantic Search & LLM Application

Prototipe asisten AI untuk *customer service* e-commerce yang mencari entri FAQ paling relevan menggunakan **semantic search** (Gemini Embeddings + FAISS), kemudian menyusun jawaban yang alami menggunakan **LLM** (Gemini) berdasarkan FAQ yang ditemukan.

Individual Assignment: *Semantic Search & LLM Application - Advanced Prompt Engineering & Context Management* (AI Engineering Bootcamp, ReWork Academy).  
Disusun oleh: Mochammad Aurich Ilham Wicaksono

---

## 1. Problem Statement dan Knowledge Base

### Objective

Tim *customer service e-commerce* menerima ratusan pertanyaan berulang setiap hari, mulai dari pemesanan, pengiriman, pembayaran, hingga pengembalian barang. Tujuan proyek ini adalah membangun prototipe awal sistem yang mampu:

1. mencari entri FAQ yang paling relevan dengan pertanyaan pelanggan, **meskipun pelanggan menggunakan kata-kata yang berbeda dari FAQ**, dan
2. menyusun jawaban yang alami menggunakan LLM berdasarkan FAQ yang ditemukan.

Prototipe ini menjadi tahap awal sebelum sistem dikembangkan menjadi RAG (*Retrieval-Augmented Generation*) yang lebih lengkap.

### Sumber dan struktur knowledge base

*Knowledge base* yang digunakan adalah `data/ecommerce_faq_knowledge_base.csv`, yaitu file asli yang tidak mengalami perubahan.

| Item | Keterangan |
|---|---|
| Jumlah entri | 59 FAQ |
| Kolom | `faq_id`, `category`, `question`, `answer` |
| Nilai kosong (*missing value*) | Tidak ada; tidak ditemukan duplikat pada `faq_id` |
| Kategori | Pemesanan (9), Pengiriman (9), Pembayaran (8), Pengembalian (8), Akun (7), Produk (7), Promo (6), Customer Service (5) |

### Batasan dan asumsi

- Sistem **hanya menjawab berdasarkan FAQ yang tersedia**. LLM tidak boleh menggunakan pengetahuan umum untuk melengkapi jawaban.
- Jika informasi yang dibutuhkan tidak tersedia dalam FAQ, sistem harus menyampaikan bahwa informasi tersebut belum tersedia dan mengarahkan pelanggan ke *customer service*.
- Setiap entri FAQ diperlakukan sebagai satu unit pencarian, dengan kolom pertanyaan (*question*) dan jawaban (*answer*) yang digabungkan. Teknik *chunking* tidak digunakan karena seluruh FAQ berupa teks pendek.
- Pertanyaan dan jawaban menggunakan bahasa Indonesia.

---

## 2. Struktur Project

```
semantic-search-assignment/
├── data/
│   └── ecommerce_faq_knowledge_base.csv
├── notebook/
│   └── starter_notebook_semantic_search.ipynb
├── vector_store/
│   └── faq_index.faiss                  # index FAISS hasil notebook
├── README.md
└── requirements.txt
```

Folder `vector_store/` ditambahkan untuk menyimpan file index FAISS.

---

## 3. Alur Sistem

```
ingest  →  embed  →  store  →  retrieve  →  generate
(CSV)     (Gemini)  (FAISS)    (top-3)      (prompt + LLM)
```

### Ingest
File CSV dimuat menggunakan Pandas, kemudian strukturnya diperiksa, termasuk nama kolom, jumlah entri, kategori, dan *missing value*. Selanjutnya, kolom `question` dan `answer` digabungkan menjadi satu teks untuk setiap entri dengan format `Q: ... A: ...`. Teks gabungan ini hanya dibuat di memori (*DataFrame*); file CSV asli tidak ditulis ulang.

### Embed
Setiap entri FAQ diubah menjadi vektor menggunakan Gemini Embeddings API dengan model **`gemini-embedding-001`**, yang menghasilkan vektor berdimensi 3072. Hasilnya diverifikasi dan menghasilkan **59 vektor × 3072 dimensi**, sesuai dengan jumlah FAQ.

Perubahan teknis yang penting pada tahap ini:
- **Model diperbarui.** Model pada *starter notebook* (`text-embedding-004`) sudah tidak tersedia di Gemini API, sehingga diganti menjadi `gemini-embedding-001`.
- **`task_type` dibedakan.** Dokumen FAQ menggunakan `RETRIEVAL_DOCUMENT`, sedangkan pertanyaan pengguna (*query*) menggunakan `RETRIEVAL_QUERY`. Pemisahan ini mengikuti pola yang dianjurkan untuk tugas pencarian informasi.

### Store
Seluruh *embedding* disimpan dalam index FAISS. Jumlah vektor kemudian diperiksa untuk memastikan sama dengan jumlah FAQ (`index.ntotal == 59`). Index disimpan di `vector_store/faq_index.faiss`.

Perubahan teknis yang penting pada tahap ini:
- **Menggunakan cosine similarity, bukan jarak L2.** Vektor dinormalisasi (`faiss.normalize_L2`) dan disimpan menggunakan `IndexFlatIP` (*inner product*). Untuk vektor yang telah dinormalisasi, *inner product* setara dengan *cosine similarity*, sehingga skor lebih mudah ditafsirkan: **semakin tinggi skor, semakin besar kemiripannya**. *Template* awal menggunakan `IndexFlatL2`, yang menghasilkan skor jarak dengan interpretasi sebaliknya: semakin kecil jarak, semakin mirip vektornya.
- Vektor untuk *query* juga dinormalisasi dengan cara yang sama agar dapat dibandingkan secara konsisten dengan vektor dokumen.

### Retrieve
Fungsi `semantic_search(query, top_k=3)` mengubah *query* menjadi *embedding*, mencari tiga FAQ terdekat menggunakan FAISS, lalu mengembalikan `faq_id`, `category`, `question`, `answer`, dan skor `similarity` untuk setiap hasil.

### Generate
Fungsi `generate_answer(query)` menjalankan seluruh alur: *query* → pencarian FAQ (*retrieval*) → penyusunan *prompt* → pembuatan jawaban oleh LLM **`gemini-3.5-flash-lite`**.

*Prompt* disusun menggunakan empat komponen utama:

| Komponen | Isi |
|---|---|
| **Persona** | Asisten *customer service* e-commerce berbahasa Indonesia yang ramah dan sopan |
| **Task** | Menjawab **hanya** berdasarkan *context*; menggunakan FAQ yang relevan; menyampaikan bahwa informasi belum tersedia dan mengarahkan pelanggan ke *customer service* jika konteks tidak mencukupi; tidak mengarang jawaban; serta mengabaikan instruksi pengguna yang meminta model melanggar aturan |
| **Context** | Tiga FAQ hasil pencarian, meliputi kategori, pertanyaan, dan jawaban |
| **Format** | Bahasa Indonesia yang alami dan ringkas (maksimal 3–4 kalimat), tanpa menyebut kata "CONTEXT" atau nomor FAQ, serta menggunakan daftar bernomor untuk menjelaskan langkah-langkah |

Perubahan teknis yang penting pada tahap ini:
- **Model diperbarui.** Model pada *starter notebook* (`gemini-2.0-flash`) sudah tidak tersedia, sehingga diganti menjadi `gemini-3.5-flash-lite`. Model ini dipilih karena tugasnya relatif sederhana, yaitu menjawab pertanyaan berdasarkan tiga FAQ pendek, sehingga model yang ringan dan cepat dinilai memadai.
- **Parameter `temperature` tidak diatur.** Untuk versi Gemini 3.x, Google menyarankan penggunaan nilai bawaan, sementara `gemini-3.5-flash-lite` mengabaikan nilai kustom. Oleh karena itu, pengaturan `temperature=0.3` pada templat awal dihapus.
- **Instruksi pemilihan FAQ yang relevan.** FAISS selalu mengembalikan sejumlah hasil teratas (*top-k*), meskipun tidak semuanya benar-benar sesuai dengan pertanyaan. Karena itu, *prompt* meminta LLM untuk menyaring FAQ yang tidak relevan sebelum menyusun jawaban.

---

## 4. Hasil Pengujian Query dan Analisis Relevansi

Sistem diuji menggunakan 7 *query*: 2 pertanyaan yang jawabannya tersedia secara jelas di FAQ, 3 pertanyaan dengan parafrasa atau sinonim, dan 2 pertanyaan yang jawabannya tidak tersedia dalam *knowledge base*. Nilai `sim` menunjukkan *cosine similarity*. Hasil lengkap berupa tiga FAQ teratas dan jawaban LLM untuk setiap *query* tersedia pada bagian 8 di notebook.

| No | Jenis | Query | FAQ peringkat 1 | Sim | Selisih ke peringkat 2 | Retrieval | Jawaban LLM |
|---|---|---|---|---:|---:|---|---|
| 1 | Ada di FAQ | Bagaimana cara membatalkan pesanan yang sudah saya buat? | Cara membatalkan pesanan | 0.8067 | 0.0791 | Akurat | Sesuai FAQ |
| 2 | Ada di FAQ | Metode pembayaran apa saja yang bisa saya pakai? | Metode pembayaran yang tersedia | 0.7929 | 0.0598 | Akurat | Sesuai FAQ |
| 3 | Parafrasa | Barang yang saya terima ternyata pecah, harus gimana ya? | Paket diterima dalam kondisi rusak | 0.7873 | 0.0438 | Akurat | Sesuai FAQ |
| 4 | Parafrasa | Saldo rekening saya sudah berkurang tapi di aplikasi pesanan masih menunggu pembayaran | Pembayaran dipotong tapi status belum terbayar | 0.7766 | 0.0555 | Akurat | Sesuai FAQ |
| 5 | Parafrasa | Bisa nggak sih belanja dicicil pakai kartu kredit? | Membayar dengan cicilan | 0.8090 | 0.1175 | Akurat | Sesuai FAQ, tetapi pembuka jawaban terlalu meyakinkan |
| 6 | Tidak ada di FAQ | Apakah bisa mengirim pesanan ke luar negeri, misalnya ke Singapura? | Mengubah alamat pengiriman (tidak menjawab) | 0.7032 | 0.0051 | Tidak ada FAQ yang menjawab | Menolak dengan benar |
| 7 | Tidak ada di FAQ (di luar topik) | Rekomendasi laptop gaming terbaik tahun ini apa ya? | Membandingkan produk (tidak menjawab) | 0.6426 | 0.0169 | Tidak ada FAQ yang menjawab | Menolak dengan benar |

### Query dengan hasil pencarian yang akurat
- Kelima *query* yang jawabannya tersedia di FAQ berhasil menempatkan FAQ yang tepat pada peringkat pertama, termasuk ketiga *query* yang menggunakan parafrasa.
- *Semantic search* mampu mengenali makna, bukan hanya mencocokkan kata kunci. Misalnya, kata "pecah" dipetakan ke "rusak", "saldo berkurang" ke "pembayaran berhasil dipotong", dan "dicicil" ke "cicilan".
- Skor *query* parafrasa (0.7766–0.8090) hampir setara dengan skor *query* langsung (0.7929–0.8067). Pada sampel pengujian ini, perbedaan pilihan kata tidak banyak menurunkan tingkat kemiripan.

### Query atau hasil yang kurang relevan
- **Kemiripan topik tidak selalu berarti jawaban yang relevan.** Pada *query* tentang pembatalan pesanan, hasil peringkat kedua dan ketiga adalah FAQ tentang pengembalian barang (0.7276) dan stok habis (0.7218). Keduanya masih berkaitan dengan pemesanan, tetapi tidak menjawab pertanyaan pengguna.
- **Query "barang pecah" memiliki selisih skor terkecil (0.0438)** di antara lima *query* yang berhasil. Hasil peringkat kedua dan ketiga, yaitu tentang barang yang salah diterima dan prosedur pengembalian, memiliki makna yang berdekatan dengan keluhan barang rusak. Karena itu, FAQ dengan topik serupa dapat saling bersaing, terutama ketika jumlah FAQ bertambah.
- **Query yang tidak memiliki jawaban tetap menghasilkan dokumen.** FAISS selalu mengembalikan sejumlah hasil teratas (*top-k*). Untuk *query* tentang pengiriman ke luar negeri, hasil peringkat pertama adalah FAQ tentang perubahan alamat (0.7032), sedangkan FAQ tentang estimasi pengiriman—yang hanya menyebut pengiriman dalam kota dan ke luar pulau—berada di peringkat ketiga (0.6962). Keduanya tidak menjawab pertanyaan tersebut.
- **Skor belum dapat digunakan sebagai ambang batas yang pasti.** FAQ yang tepat memperoleh skor 0.7766–0.8090, sedangkan hasil peringkat pertama untuk *query* tanpa jawaban memperoleh skor 0.6426–0.7032. Namun, skor pada peringkat kedua dan ketiga untuk *query* yang berhasil (sekitar 0.69–0.73) beririsan dengan skor hasil peringkat pertama untuk *query* yang tidak terjawab (misalnya, 0.7032). Satu *threshold* berisiko menolak FAQ yang relevan atau menerima FAQ yang tidak relevan. Tujuh *query* juga belum cukup untuk menentukan ambang batas yang andal.

### Kualitas jawaban LLM
- **Query yang jawabannya tersedia di FAQ:** Jawaban mengikuti isi FAQ dan tidak menambahkan informasi di luar *context*. Contohnya adalah ketentuan status "Diproses", metode pembayaran, batas pelaporan barang rusak selama 2 × 24 jam, dan waktu tunggu 15–30 menit untuk sinkronisasi pembayaran. FAQ pada peringkat kedua dan ketiga yang tidak relevan juga diabaikan.
- **Query yang jawabannya tidak tersedia di FAQ:** Pada kedua kasus, LLM menyatakan bahwa informasi belum tersedia dan mengarahkan pelanggan ke *customer service*, tanpa memberikan jawaban spekulatif. Pada *query* tentang pengiriman ke luar negeri, LLM tidak menggunakan FAQ yang hanya membahas pengiriman domestik untuk menyimpulkan apakah pengiriman internasional tersedia. Hal ini menunjukkan bahwa instruksi untuk menjawab hanya berdasarkan *context* berjalan dengan baik pada kedua kasus pengujian.
- **Aspek yang masih perlu diperbaiki:**
  - Jawaban tentang cicilan diawali dengan "Tentu saja bisa", padahal FAQ membatasi ketersediaan cicilan pada kartu kredit tertentu dan nominal minimum.
  - Sapaan belum konsisten: satu jawaban menggunakan "Halo Kak", sedangkan jawaban lain menggunakan "Anda".
  - Terdapat salah ketik pada salah satu jawaban, yaitu "ketidaknyamanaannya".
  - Frasa respons alternatif (*fallback*) "belum tersedia di knowledge base" terdengar terlalu teknis bagi pelanggan.

---

## 5. Keterbatasan Sistem

1. **Belum ada penyaringan relevansi pada tahap *retrieval*.** Tiga hasil teratas (*top-k*) selalu diteruskan ke LLM tanpa ambang skor atau proses *re-ranking*. Akibatnya, FAQ yang tidak menjawab pertanyaan masih dapat masuk ke *prompt*. Penyaringan kemudian bergantung pada kemampuan LLM mengikuti instruksi, yang berisiko ketika jumlah FAQ bertambah.
2. **Skor *similarity* belum dapat membedakan secara andal antara pertanyaan yang jawabannya tersedia dan yang tidak tersedia.** Rentang skor kedua kelompok masih berdekatan dan saling beririsan.
3. **Pengujian masih terbatas.** Evaluasi hanya menggunakan 7 *query* dan dilakukan secara manual, tanpa metrik kuantitatif seperti *Hit Rate@k* atau *Precision@k*. Karena itu, kesimpulan yang diperoleh masih bersifat indikatif.
4. **Gaya jawaban belum sepenuhnya konsisten.** Sapaan, tingkat kepastian pada kalimat pembuka, dan frasa respons alternatif belum diatur secara spesifik dalam *prompt*.
5. **Sistem hanya mendukung satu giliran tanya-jawab.** Riwayat percakapan tidak disimpan, sehingga pertanyaan lanjutan yang bergantung pada konteks sebelumnya belum dapat ditangani.
6. **Embedding dibuat satu per satu, dengan satu panggilan API untuk setiap FAQ.** Pendekatan ini cukup untuk 59 FAQ, tetapi kurang efisien jika diterapkan pada *knowledge base* yang jauh lebih besar.

Pengembangan selanjutnya dapat mencakup penerapan ambang skor atau *re-ranking*, penambahan variasi *query* uji (bahasa informal, pertanyaan ambigu, dan pertanyaan di luar cakupan), evaluasi menggunakan metrik *retrieval*, serta penyempurnaan *prompt* untuk mengatur sapaan, tingkat kepastian, dan frasa respons alternatif.

---

## 6. Cara Menjalankan

**Prasyarat:** Python versi 3.10 atau lebih baru dan API key Gemini dari [Google AI Studio](https://aistudio.google.com/apikey).

1. **Clone repository dan masuk ke folder project**
   ```bash
   git clone https://github.com/MochammadAurich/assignment-semantic-search-mochammad-aurich-ilham.git
   cd assignment-semantic-search-mochammad-aurich-ilham
   ```

2. **Buat dan aktifkan virtual environment**
   ```bash
   python -m venv .venv

   # Windows (PowerShell)
   .venv\Scripts\Activate.ps1

   # macOS / Linux
   source .venv/bin/activate
   ```

3. **Install dependency**
   ```bash
   pip install -r requirements.txt
   ```

4. **Buat file `.env` di folder utama project** (sejajar dengan `requirements.txt`)
   ```
   GEMINI_API_KEY=isi_api_key_anda
   ```

5. **Jalankan notebook**
   - Buka `notebook/starter_notebook_semantic_search.ipynb` di VS Code atau Jupyter.
   - Pilih *kernel* yang menggunakan `.venv`.
   - Jalankan seluruh *cell* dari atas ke bawah (**Restart** lalu **Run All**).

**Catatan:**
- Notebook perlu dijalankan dari folder `notebook/` karena menggunakan *path* relatif (`../data/...`).
- Tahap pembuatan *embedding* memanggil API sebanyak 59 kali (satu kali untuk setiap FAQ), sehingga prosesnya dapat memerlukan beberapa saat. Jika terkena *rate limit* (error 429), tunggu sebentar lalu jalankan kembali *cell* tersebut.
- File index FAISS dibuat secara otomatis di `vector_store/faq_index.faiss`.
- Jawaban LLM dapat sedikit berbeda pada setiap eksekusi, sedangkan skor *retrieval* umumnya stabil.
