import os
import pandas as pd
import requests
import base64

# Konfigurasi
DATASET_PATH = r'C:\Users\ABDUL\OneDrive\Documents\Politeknik Negeri Batam\Teknik Robotika\Semester 6\Computer Vision (RE606)\License Plate v1.4\test'
GROUND_TRUTH_CSV = r'C:\Users\ABDUL\OneDrive\Documents\Politeknik Negeri Batam\Teknik Robotika\Semester 6\Computer Vision (RE606)\License Plate v1.4\ground_truth.csv'
MODEL_URL = 'http://10.170.5.208:1234/v1/chat/completions'

# Fungsi untuk mengirim gambar ke model dan mendapatkan prediksi
def get_license_plate_prediction(image_path):
    with open(image_path, 'rb') as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
        payload = {
            "model": "llava-phi-3-mini-gguf",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": "What is the license plate number shown in this image? Respond only with the plate number."
                        }
                    ]
                }
            ],
            "stream": False
        }
        
        response = requests.post(MODEL_URL, json=payload)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()  # Mengembalikan hasil prediksi
        else:
            print(f"Error: {response.status_code}")
            return None

# Fungsi untuk menghitung Character Error Rate (CER)
def calculate_cer(ground_truth, prediction):
    S = sum(1 for a, b in zip(ground_truth, prediction) if a != b)  # Substitusi
    D = len(ground_truth) - len(prediction) if len(ground_truth) > len(prediction) else 0  # Dihapus
    I = len(prediction) - len(ground_truth) if len(prediction) > len(ground_truth) else 0  # Disisipkan
    N = len(ground_truth)  # Jumlah karakter pada ground truth
    cer = (S + D + I) / N if N > 0 else 0
    return cer

# Memuat ground truth dari file CSV
ground_truth_df = pd.read_csv(GROUND_TRUTH_CSV)

# Daftar untuk menyimpan hasil
results = []

# Proses setiap gambar dalam folder 'test'
for index, row in ground_truth_df.iterrows():
    image_file = row['image']
    ground_truth = row['ground_truth']
    
    image_path = os.path.join(DATASET_PATH, image_file)
    
    if os.path.exists(image_path):  # Pastikan file gambar ada
        # Dapatkan prediksi dari model
        prediction = get_license_plate_prediction(image_path)
        
        if prediction:
            cer_score = calculate_cer(ground_truth, prediction)
            results.append({
                'image': image_file,
                'ground_truth': ground_truth,
                'prediction': prediction,
                'CER_score': cer_score
            })
    else:
        print(f"File {image_path} tidak ditemukan.")

# Simpan hasil ke dalam CSV
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_CSV, index=False)

print(f"Hasil OCR disimpan dalam {OUTPUT_CSV}")
