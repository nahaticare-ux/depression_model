import streamlit as st
import Orange
import pickle

# 1. 모델 불러오기
@st.cache_resource
def load_model():
    with open("depression_model.pkcls", "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("☁️ 마음기상청: 대학생 우울증 예보")
st.write("Orange3에서 학습된 7개 지표를 기반으로 마음 날씨를 분석합니다.")

# 2. 사용자 입력 받기 (주요 변수 3개)
stress_val = st.slider("오늘 스트레스 정도 (1~10)", 1, 10, 5)
sleep_val = st.number_input("어제 수면 시간 (시간)", 0.0, 24.0, 7.0)
social_val = st.number_input("SNS 사용 시간 (시간)", 0.0, 24.0, 2.0)

# 3. 예측하기
if st.button("예보 확인하기"):
    try:
        # [중요] image_16d820.png의 Features(7) 순서를 그대로 적용합니다.
        # 1. Age (나이: 평균 21세)
        # 2. Gender (성별: 남성 0 / 여성 1)
        # 3. Sleep_Duration (수면 시간: 입력값)
        # 4. Study_Hours (공부 시간: 평균 5시간)
        # 5. Social_Media_Hours (SNS 시간: 입력값)
        # 6. Physical_Activity (활동량: 평균 3시간)
        # 7. Stress_Level (스트레스: 입력값)
        
        input_values = [21, 1, sleep_val, 5.0, social_val, 3, stress_val] 
        
        # 모델 예측 (입력 데이터 개수가 정확히 7개인지 확인)
        domain = model.domain
        inst = Orange.data.Instance(domain, input_values)
        
        prediction = model(inst) # 0(False) 또는 1(True)
        probs = model(inst, ret=Orange.classification.Model.ValueProbs) # [안정, 우울] 확률

        # 4. 결과 출력
        st.divider()
        risk_percent = probs[1] * 100

        if prediction == 1 or risk_percent > 50:
            st.error(f"⚠️ 현재 마음 날씨는 '흐림'입니다. (우울 위험 확률: {risk_percent:.1f}%)")
            st.write("혼자 고민하지 마세요. 따뜻한 차 한 잔과 함께 충분한 휴식을 권장합니다.")
        else:
            st.success(f"☀️ 현재 마음 날씨는 '맑음'입니다. (안정 확률: {probs[0]*100:.1f}%)")
            st.write("건강한 마음 상태를 유지하고 계시네요! 멋집니다.")

    except Exception as e:
        st.error(f"예측 중 오류 발생: {e}")
        st.info("입력 변수가 7개인지, 모델 파일이 최신인지 확인해 주세요.")

