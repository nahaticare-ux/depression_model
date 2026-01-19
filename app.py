import streamlit as st
import Orange
import pickle

# 1. 모델 불러오기
@st.cache_resource
def load_model():
    with open("depression_model.pkcls", "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("☁️ 마음기상청: 신경망(AI) 분석기")

# 2. 입력 받기
stress = st.slider("오늘 스트레스 (1~10)", 1, 10, 5)
sleep = st.slider("수면 시간", 0.0, 15.0, 7.0)
social = st.slider("SNS 사용", 0.0, 15.0, 2.0)

if st.button("예보 확인하기"):
    try:
        # [해결 1] 9개의 입구를 순서대로 채움 (7개 실데이터 + 2개 가짜데이터)
        # 이미지(16d820.png) 기반 순서: Age, Gender, Sleep, Study, Social, Physical, Stress, (Target), (Meta)
        raw_input = [21.0, 1.0, float(sleep), 5.0, float(social), 3.0, float(stress), 0, 0]
        
        inst = Orange.data.Instance(model.domain, raw_input)
        
        # [해결 2] 신경망 전용 결과 추출법
        # 신경망은 결과가 리스트 안에 들어있으므로 [0]을 통해 '스칼라' 값으로 변환합니다.
        prediction = model(inst)
        pred_value = int(prediction[0]) # 리스트의 첫 번째 값 추출
        
        probs = model(inst, ret=Orange.classification.Model.ValueProbs)
        risk_percent = float(probs[0][1]) * 100 # 우울(1)일 확률

        # 3. 결과 표시
        st.divider()
        if pred_value == 1:
            st.error(f"⚠️ 현재 마음 날씨는 '흐림'입니다. (우울 위험: {risk_percent:.1f}%)")
        else:
            st.success(f"☀️ 현재 마음 날씨는 '맑음'입니다. (안정 확률: {100-risk_percent:.1f}%)")

    except Exception as e:
        st.error(f"오류: {e}")
