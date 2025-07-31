# AAS - Computer - Vision
## 4222201039 Abdul Fathah Fathullah
# OCR Plat Nomor Kendaraan dengan Visual Language Model (VLM)

Proyek ini dibuat untuk menyelesaikan tugas akhir mata kuliah **Computer Vision (RE604)** pada semester genap 2024/2025, Program Studi Teknik Robotika.

## 🎯 Tujuan

Melakukan Optical Character Recognition (OCR) pada plat nomor kendaraan **menggunakan Visual Language Model (VLM)** berbasis gambar dari dataset plat nomor Indonesia.

---

## 🧠 Teknologi yang Digunakan

- Python
- [LM Studio](https://lmstudio.ai) (API lokal)
- Model: `llava-phi-3-mini-gguf`
- Dataset: [Indonesian License Plate Recognition](https://www.kaggle.com/datasets/juanthomaswijaya/indonesian-license-plate-dataset)

---

## ⚙️ Cara Kerja Program

1. **Ambil Data**: Program membaca file `ground_truth.csv` dan gambar dari folder `test`.
2. **Encode Gambar**: Gambar dikonversi menjadi base64.
3. **Inferensi ke Model**: Gambar dan prompt dikirim ke model melalui LM Studio (API lokal).
4. **Evaluasi CER**: Hasil prediksi dibandingkan dengan ground truth menggunakan **Character Error Rate (CER)**.
5. **Simpan Hasil**: Semua hasil disimpan dalam file `ocr_results.csv`.

---
