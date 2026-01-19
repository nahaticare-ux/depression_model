import streamlit as st
import Orange
import pickle
import numpy as np

# 1. 모델 불러오기
@st.cache_resource
def load_model():
    with open("depression_model.pkcls", "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("☁️ 마음기상청: 대학생 우울증 예보")
st.write("당신의 일상 데이터를 기반으로 마음 날씨를 분석합니다.")

# 2. 사용자 입력 받기
stress_val = st.slider("오늘 스트레스 정도 (1~10)", 1, 10, 5)
sleep_val = st.number_input("어제 수면 시간 (시간)", 0.0, 24.0, 7.0)
social_val = st.number_input("SNS 사용 시간 (시간)", 0.0, 24.0, 2.0)

# 3. 예측하기
if st.button("예보 확인하기"):
    try:
        # [핵심 수정] 8개의 자리를 만듭니다 (7개 변수 + 1개 더미 타겟)
        # 순서: Age, Gender, Sleep, Study, Social, Physical, Stress, (Target Dummy)
        # 타겟 자리에 0을 하나 더 추가하여 총 8개를 맞춥니다.
        raw_data = [21.0, 1.0, sleep_val, 5.0, social_val, 3.0, stress_val, 0]
        
        # 모델이 요구하는 도메인 형식에 맞춰 인스턴스 생성
        domain = model.domain
        inst = Orange.data.Instance(domain, raw_data)
        
        # 예측 수행
        prediction = model(inst) 
        probs = model(inst, ret=Orange.classification.Model.ValueProbs)

        # 4. 결과 출력
        st.divider()
        risk_percent = probs[1] * 100

        if prediction == 1 or risk_percent > 50:
            st.error(f"⚠️ 현재 마음 날씨는 '흐림'입니다. (위험 확률: {risk_percent:.1f}%)")
            st.write("오늘은 평소보다 나 자신을 더 아껴주세요. 가벼운 산책은 어떨까요?")
        else:
            st.success(f"☀️ 현재 마음 날씨는 '맑음'입니다. (안정 확률: {probs[0]*100:.1f}%)")
            st.write("마음 기상이 아주 쾌청합니다! 이 기분 그대로 즐거운 하루 보내세요.")

    except Exception as e:
        # 에러가 나면 정확히 몇 개의 데이터가 필요한지 메시지로 띄워줍니다.
        st.error(f"예측 오류 발생: {e}")
        st.info("모델이 요구하는 데이터 규격과 입력값이 일치하지 않습니다. 데이터 개수를 조정해보세요.")
