# ... (앞부분 생략) ...

if st.button("마음 날씨 예보하기"):
    try:
        # [해결 1] 모델이 요구하는 9개 칸 채우기
        input_list = [21.0, 1.0, float(sleep), 5.0, float(social), 3.0, float(stress), 0, 0]
        inst = Orange.data.Instance(model.domain, input_list)
        
        # [해결 2] 결과 추출 방식 수정
        prediction = model(inst)
        
        # 만약 결과가 리스트(배열) 형태라면 첫 번째 값을 가져오고, 아니면 그대로 숫자로 바꿉니다.
        # 이 부분이 image_21c4c9.png 의 scalar 오류를 해결하는 핵심입니다.
        if hasattr(prediction, "__len__"):
            pred_value = int(prediction[0])
        else:
            pred_value = int(prediction)

        # 확률값 추출
        probs = model(inst, ret=Orange.classification.Model.ValueProbs)
        risk_prob = float(probs[1]) * 100

        # [결과 출력]
        st.divider()
        if pred_value == 1:
            st.error(f"⚠️ 현재 마음 날씨는 '흐림'입니다. (우울 위험 확률: {risk_prob:.1f}%)")
        else:
            st.success(f"☀️ 현재 마음 날씨는 '맑음'입니다. (안정 확률: {100-risk_prob:.1f}%)")

    except Exception as e:
        st.error(f"분석 중 오류가 발생했습니다: {e}")
