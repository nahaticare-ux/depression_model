import streamlit as st
import Orange
import pickle
import numpy as np

# 1. 모델 불러오기
@st.cache_resource
def load_model():
    # 저장한 모델 파일 이름이 정확한지 확인하세요!
    with open("depression_model.pkcls", "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("☁️ 마음기상청: 대학생 우울증 예보")
st.write("오늘 당신의 마음 날씨를 확인해 보세요.")

# 2. 사용자 입력 받기
stress = st.slider("오늘 스트레스 정도 (1~10)", 1, 10, 5)
sleep = st.number_input("어제 수면 시간 (시간)", 0.0, 24.0, 7.0)
social = st.number_input("SNS 사용 시간 (시간)", 0.0, 24.0, 2.0)

# 3. 예측하기
if st.button("예보 확인하기"):
    try:
        # [핵심 수정] 모델이 요구하는 8개의 자리를 정확히 채워줍니다.
        # 앞의 3개는 사용자의 입력값, 뒤의 5개는 평균값(더미 데이터)입니다.
        # 만약 에러가 계속나면 아래 리스트에 숫자를 하나 더 추가해 보세요.
        input_values = [stress, sleep, social, 3.0, 5.0, 21.0, 0, 0] 
        
        # 모델의 데이터 구조(Domain)를 가져와서 입력 데이터 생성
        domain = model.domain
        inst = Orange.data.Instance(domain, input_values)
        
        # 예측 수행
        prediction = model(inst) # 결과값 (0 또는 1)
        probs = model(inst, ret=Orange.classification.Model.ValueProbs) # 확률값

        # 4. 결과 출력
        st.divider()
        risk_percent = probs[1] * 100 # 우울증(True) 확률

        if prediction == 1 or risk_percent > 50:
            st.error(f"⚠️ 현재 마음 날씨는 '흐림'입니다. (위험 확률: {risk_percent:.1f}%)")
            st.write("마음의 신호에 귀를 기울여야 할 때입니다. 전문가와 이야기 나누는 것을 추천드려요.")
        else:
            st.success(f"☀️ 현재 마음 날씨는 '맑음'입니다. (안정 확률: {probs[0]*100:.1f}%)")
            st.write("건강한 마음 상태를 유지하고 계시네요! 지금처럼 자신을 잘 돌봐주세요.")

    except Exception as e:
        st.error(f"예측 중 오류가 발생했습니다: {e}")
        st.info("입력 데이터 개수가 모델 설정과 일치하지 않습니다. 코드를 다시 확인해 주세요.")
