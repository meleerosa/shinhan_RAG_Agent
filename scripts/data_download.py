import json
import requests
import os
import re
import time

# 1. JSON 파일 로드
# 저장하신 경로에 맞춰서 파일명을 수정해주세요.
json_file_path = "/home/wlaud/projects/shinhan/data/yeon.json"

try:
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"✅ JSON 파일 로드 성공! 총 {len(data)}개의 데이터가 있습니다.")
except FileNotFoundError:
    print(f"❌ '{json_file_path}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    exit()

# 2. 저장할 폴더 생성
save_folder = "shinhan_pdfs_yeon"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    print(f"📂 '{save_folder}' 폴더를 생성했습니다.")

# 3. 다운로드 시작
success_count = 0
fail_count = 0

print(f"🚀 다운로드를 시작합니다... (저장 위치: {save_folder})")

for idx, item in enumerate(data):
    # PDF URL과 제목 확인
    url = item.get("PDF_FILE_NM")
    raw_title = item.get("TITLE", "제목없음")
    
    if not url:
        continue # URL 없는 항목 건너뜀

    # 파일명 다듬기 (특수문자 제거)
    safe_title = re.sub(r'[\\/*?:"<>|]', "", raw_title).strip()
    # 중복 파일명 방지를 위해 뒤에 ID나 번호 붙이기 (선택)
    form_id = item.get("FORM_ID", str(idx))
    filename = f"{safe_title}_{form_id}.pdf"
    file_path = os.path.join(save_folder, filename)

    # 이미 다운받은 파일이면 스킵 (중단 후 재시작 시 유용)
    if os.path.exists(file_path):
        print(f"[{idx+1}/{len(data)}] 패스 (이미 존재): {filename}")
        success_count += 1
        continue

    print(f"[{idx+1}/{len(data)}] 다운로드 중: {filename}")

    try:
        # 다운로드 요청 (타임아웃 10초)
        response = requests.get(url, stream=True, timeout=15)
        
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            success_count += 1
        else:
            print(f"   ⚠️ 실패 (HTTP {response.status_code})")
            fail_count += 1
            
    except Exception as e:
        print(f"   ❌ 에러 발생: {e}")
        fail_count += 1

    # 서버 부하 방지용 짧은 대기 (0.1초)
    time.sleep(0.1)

print("\n" + "="*40)
print(f"🎉 다운로드 완료!")
print(f"성공: {success_count}개")
print(f"실패: {fail_count}개")
print(f"저장 폴더: {os.path.abspath(save_folder)}")
print("="*40)
