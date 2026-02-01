(async function() {
    var totalPages = 102; 
    var allData = [];
    var submissionID = "sbm_THO0400"; // 로그에서 확인된 ID
    
    // 로그에 찍힌 옵션 그대로 사용
    var subOptions = {
        "serviceType": "TG",
        "serviceCode": "THO0400",
        "callBack": "shbObj.fncDoTHO0400_Callback" 
    };

    console.log("🚀 신한은행 전용 모듈로 수집 시작 (총 " + totalPages + "페이지)");

    for (var i = 1; i <= totalPages; i++) {
        // 1. 페이지 번호 설정
        var paramMap = WebSquare.util.getComponentById("dm_S_THO0400");
        if (paramMap) {
            paramMap.set("PAGE", i);
        } else {
            console.log("❌ 조회 조건 맵(dm_S_THO0400)을 못 찾았습니다.");
            break;
        }

        // 2. 조회 실행 (shbComm 사용)
        // 이전 데이터의 첫 번째 항목 ID 기억 (데이터 변경 확인용)
        var lastFirstID = (allData.length > 0) ? allData[allData.length - 1].FORM_ID : "START";
        
        try {
            // ★ 핵심: 로그에 찍힌 방식 그대로 호출 ★
            shbComm.executeSubmission(submissionID, subOptions);
        } catch (e) {
            console.log("❌ shbComm 실행 실패: " + e.message);
            // 만약 shbComm이 없으면 WebSquare로 시도
            WebSquare.ModelUtil.executeSubmission(submissionID);
        }

        // 3. 데이터 갱신 대기 (최대 5초)
        var pageData = [];
        var retries = 0;
        
        while (retries < 20) { // 0.25초 * 20 = 5초 대기
            await new Promise(r => setTimeout(r, 250)); // 0.25초 대기
            
            var comp = WebSquare.util.getComponentById("dl_R_THO0400");
            if (comp) {
                var currentData = comp.getAllJSON();
                if (currentData.length > 0) {
                    // 첫 페이지(i=1)이거나, 데이터가 이전 페이지와 다르면 성공
                    if (i === 1 || currentData[0].FORM_ID !== lastFirstID) {
                        pageData = currentData;
                        break; // 대기 종료
                    }
                }
            }
            retries++;
        }

        // 4. 결과 저장
        if (pageData.length > 0) {
            allData = allData.concat(pageData);
            console.log(`✅ [${i}/${totalPages}] 수집 성공 (${pageData.length}건)`);
        } else {
            console.log(`⚠️ [${i}페이지] 데이터 갱신 실패 (시간 초과 또는 마지막 페이지)`);
            // 실패해도 일단 진행 (멈추면 안 되니까)
        }
        
        // 서버 부하 방지
        await new Promise(r => setTimeout(r, 100));
    }

    console.log("🎉 최종 완료! 총 " + allData.length + "건");
    
    // 결과 출력 (복사하기 편하게)
    var resultStr = JSON.stringify(allData, null, 2);
    console.log("⬇️ 아래 데이터를 복사하세요 ⬇️");
    console.log(resultStr);
    
    // (선택) 자동으로 클립보드에 복사 시도
    try { copy(resultStr); console.log("📋 클립보드에 복사되었습니다!"); } catch(e) {}

})();