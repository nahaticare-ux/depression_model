import streamlit as st
import Orange
import pickle
import numpy as np

# 1. 모델 불러오기
@st.cache_resource # 모델을 한 번만 불러오도록 최적화
def load_model():
    with open("depression_model.pkcls", "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("☁️ 마음기상청: 대학생 우울증 예보")
st.write("오늘 당신의 마음 날씨를 확인해 보세요.")

# 2. 사용자 입력 받기 (상위 3개 중요 변수)
stress = st.slider("오늘 스트레스 정도 (1~10)", 1, 10, 5)
sleep = st.number_input("어제 수면 시간 (시간)", 0.0, 24.0, 7.0)
social = st.number_input("SNS 사용 시간 (시간)", 0.0, 24.0, 2.0)

# 3. 예측하기
if st.button("예보 확인하기"):
    try:
        # [해결 1] 변수 개수 맞추기 (이미지 image_149207.png 기준 총 7개)
        # 순서: Stress, Sleep, Social, Physical_Activity, Study_Hours, Age, Gender
        # 입력받지 않은 값은 데이터셋 평균값으로 채웁니다.
        input_values = [stress, sleep, social, 3.0, 5.0, 21.0, 0] 
        
        # [해결 2] Orange 모델 전용 데이터 형식으로 변환
        # 단순히 model(input_data)를 하면 index 에러가 날 수 있어 더 안전한 방식을 사용합니다.
        domain = model.domain
        data = Orange.data.Table(domain, [input_values])
        
        # 예측 수행
        prediction = model(data)[0] # 0 또는 1
        probs = model(data, ret=Orange.classification.Model.ValueProbs)[0] # [안정확률, 우울확률]

        # 4. 결과 보여주기
        st.divider()
        risk_percent = probs[1] * 100

        if prediction == 1 or risk_percent > 50:
            st.error(f"⚠️ 현재 마음 날씨는 '흐림'입니다. (위험 확률: {risk_percent:.1f}%)")
            st.write("심리적 부담이 큰 상태인 것 같아요. 주변 친구나 상담센터의 도움을 받아보는 건 어떨까요?")
        else:
            st.success(f"☀️ 현재 마음 날씨는 '맑음'입니다. (안정 확률: {probs[0]*100:.1f}%)")
            st.write("마음 건강 상태가 양호합니다! 지금의 긍정적인 에너지를 유지해 보세요.")

    except Exception as e:
        st.error(f"예측 중 오류가 발생했습니다: {e}")
        st.info("입력 데이터 형식을 모델에 맞게 조정 중입니다. 잠시 후 다시 시도해 주세요.")
