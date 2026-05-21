import pandas as pd
from tqdm import tqdm
import static_ffmpeg
static_ffmpeg.add_paths()
from pythainlp import word_tokenize
from pythainlp.corpus import thai_words

# โหลดไฟล์ต้นฉบับ
file_path = 'C:/Users/66982/OneDrive/Desktop/BOTNOI_AutoQC/email1-QC/Email1.csv'
df_ori = pd.read_csv(file_path, encoding='utf-8-sig')

# สร้างคอลัมน์ audio_index ขึ้นมาใหม่ 
df_ori['audio_index'] = df_ori.index



# --- ตั้งค่าเริ่มต้น ---
batch_size = 1000
total_rows = len(df_ori) # จะได้ 7071 แถว
output_filename = 'C:/Users/66982/OneDrive/Desktop/BOTNOI_AutoQC/NEW_QC_email1.xlsx'
all_results_list = [] # List สำหรับเก็บ DataFrame ของแต่ละ Batch

print(f"🚀 เริ่มประมวลผลทั้งหมด {total_rows} แถว (แบ่งเป็นรอบละ {batch_size} แถว)...")


# --- รวบรวมคำที่ตัดผิดจากที่เจอ ---
custom_words = {
        "เฮช", "แอท", "ซิสเต็ม", "แบงค็อก", "กอฟ", 
        "อีเมล", "ดอท", "คอนเฟิร์ม", "ออฟฟิศ"
    } 

# เอาคำภาษาไทยมาตรฐาน + คำที่เรานิยามเอง
from pythainlp.util import dict_trie
custom_dict = dict_trie(list(custom_words) + list(thai_words()))


# วนลูปใหญ่ตามจำนวน Batch
for start_idx in range(0, total_rows, batch_size):
    end_idx = min(start_idx + batch_size, total_rows)
    df_batch = df_ori.iloc[start_idx:end_idx].copy()
    
    print(f"\n📦 กำลังรัน Batch: {start_idx} ถึง {end_idx}")

    # --- เตรียม List สำหรับเก็บค่าใน Batch นี้ ---
    batch_word_counts = []
    batch_tokens_list = [] # เก็บ "ข้อความที่ตัดแล้ว" เช่น 'สวัสดี|ครับ|ขอบคุณ|ครับ'


    # วนลูปย่อยในแต่ละ Batch
    for index, row in tqdm(df_batch.iterrows(), total=len(df_batch),
                           desc=f"Progress {start_idx}-{end_idx}"):
        audio_idx = row['audio_index']
        text = str(row['text']) # แปลงเป็น string เผื่อเจอค่าว่าง (NaN)
        # --- ส่วนการตัดคำภาษาไทย ---
      
        tokens = word_tokenize( text,
                                engine="newmm", 
                                custom_dict=custom_dict, # เรียกใช้ตัวแปรที่เพิ่งสร้างสำเร็จ
                                keep_whitespace=False
                                )

        word_count = len(tokens)
        
        
        # เก็บค่าลง List ชั่วคราว
        batch_word_counts.append(word_count)
        batch_tokens_list.append(" ".join(tokens)) # เก็บคำที่ตัดแล้วไว้ดู (ใช้ spacebar คั่น)

    # เพิ่มข้อมูลกลับเข้าไปใน df_batch
    df_batch['word_count'] = batch_word_counts
    df_batch['tokens'] = batch_tokens_list

    # เก็บผลลัพธ์ Batch นี้ลงใน List รวม
    all_results_list.append(df_batch)

# รวมผลลัพธ์ทั้งหมดเป็น DataFrame เดียว
final_df = pd.concat(all_results_list, ignore_index=True)

# บันทึกลง Excel
final_df.to_excel(output_filename, index=False)
print(f"\n✅ ประมวลผลเสร็จสิ้น! บันทึกไฟล์ที่: {output_filename}")
print(final_df['tokens'].head(10))
