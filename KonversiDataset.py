import wfdb
import pandas as pd

# Nama file record (tanpa ekstensi .dat atau .hea)
record_name = 'MIMIC3Dataset/30/3000003/3000003_0007'

try:
    # Membaca record menggunakan library wfdb
    record = wfdb.rdrecord(record_name)
    
    # Mengubah sinyal menjadi DataFrame pandas
    df = pd.DataFrame(record.p_signal, columns=record.sig_name)
    
    # Menampilkan daftar kanal yang tersedia (untuk memastikan ada ECG dan ABP)
    print("Kanal yang ditemukan:", record.sig_name)
    
    # Rename kolom agar sesuai dengan kode Mas Ilham (Lampiran B.1)
    # Biasanya kanal ECG bernama 'II' atau 'I', dan ABP bernama 'ABP'
    # Sesuaikan mapping di bawah ini jika nama kanalnya berbeda
    mapping = {
        'II': 'ECG',
        'ABP': 'ABP'
    }
    df.rename(columns=mapping, inplace=True)
    
    # Simpan menjadi df.csv (sesuai Lampiran B.1 baris 4)
    df.to_csv('df.csv', index=False)
    print("Konversi Berhasil! File 'df.csv' siap digunakan.")

except Exception as e:
    print(f"Terjadi kesalahan: {e}")