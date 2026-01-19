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
st.write("인공지능이 당신의 일상을 분석하여 마음의 날씨를 알려드립니다.")

# 2. 사용자 입력
stress = st.slider("오늘 스트레스 정도 (1~10)", 1, 10, 5)
sleep = st.number_input("어제 수면 시간 (시간)", 0.0, 24.0, 7.0)
social = st.number_input("SNS 사용 시간 (시간)", 0.0, 24.0, 2.0)

# 3. 예측하기
if st.button("예보 확인하기"):
    try:
        # [데이터 규격 맞추기] Age, Gender, Sleep, Study, Social, Physical, Stress, Target
        input_data = [21.0, 1.0, float(sleep), 5.0, float(social), 3.0, float(stress), 0.0]
        
        domain = model.domain
        # 모델이 요구하는 정확한 개수만큼만 잘라내거나 0을 채웁니다.
        required = len(domain.attributes) + (1 if domain.class_var else 0)
        final_input = (input_data + [0.0]*10)[:required]
        
        inst = Orange.data.Instance(domain, final_input)
        
        # 예측 수행
        prediction = model(inst)
        probs = model(inst, ret=Orange.classification.Model.ValueProbs)

        # [에러 해결 포인트] 확률값에서 숫자 하나만 추출합니다.
        risk_prob = float(probs[1]) # 우울 위험 확률
        safe_prob = float(probs[0]) # 안정 확률
        
        # 4. 결과 출력
        st.divider()
        if prediction == 1 or risk_prob > 0.5:
            st.error(f"⚠️ 현재 마음 날씨는 '흐림'입니다. (위험 확률: {risk_prob*100:.1f}%)")
            st.write("마음이 조금 무거운 상태일 수 있어요. 오늘은 자신을 위해 푹 쉬어주는 게 어떨까요?")
        else:
            st.success(f"☀️ 현재 마음 날씨는 '맑음'입니다. (안정 확률: {safe_prob*100:.1f}%)")
            st.write("쾌청한 마음 상태입니다! 오늘 하루도 즐겁게 보내시길 바랄게요.")

    except Exception as e:
        st.error(f"분석 중 오류가 발생했습니다. 다시 시도해 주세요. (에러내용: {e})")
