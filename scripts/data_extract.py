import pdfplumber
import os
import pandas as pd
from tqdm import tqdm

def extract_text_from_pdf(pdf_path):
    full_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # 테이블 처리가 까다로우므로 텍스트 위주로 추출
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return None
    return full_text

# 1. 분류된 파일 리스트 로드 (이전 단계에서 만든 csv)
df = pd.read_csv("/home/wlaud/projects/shinhan/data/classified_file_list_v2.csv")

# 2. 텍스트 추출 실행
base_paths = {
    "susin": "/home/wlaud/projects/shinhan/data/shinhan_pdfs_susin",
    "yeon": "/home/wlaud/projects/shinhan/data/shinhan_pdfs_yeon"
}

extracted_data = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting PDFs"):
    folder = row['folder']
    filename = row['filename']
    category = row['category']
    
    file_path = os.path.join(base_paths[folder], filename)
    
    raw_text = extract_text_from_pdf(file_path)
    
    if raw_text:
        extracted_data.append({
            "filename": filename,
            "category": category,
            "raw_text": raw_text
        })

# 3. 결과 저장 (중간 백업)
result_df = pd.DataFrame(extracted_data)
result_df.to_pickle("pdf_raw_text_data.pkl") # 텍스트가 크므로 pickle 권장
print(f"총 {len(result_df)}개 파일 텍스트 추출 완료")
