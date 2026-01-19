import streamlit as st
import Orange
import pickle

# 1. 모델 불러오기
@st.cache_resource
def load_model():
    with open("depression_model.pkcls", "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("☁️ 마음기상청: 신경망(AI) 우울증 예보")
st.write("우리 반 인공지능이 당신의 마음 데이터를 분석합니다.")

# 2. 사용자 입력 (수업 시간에 직접 조절할 3개 지표)
stress = st.slider("스트레스 지수 (1~10)", 1, 10, 5)
sleep = st.number_input("평균 수면 시간 (시간)", 0.0, 24.0, 7.0)
social = st.number_input("SNS 사용 시간 (시간)", 0.0, 24.0, 2.0)

# 3. 분석 버튼
if st.button("마음 날씨 예보하기"):
    try:
        # [핵심] 신경망 모델이 요구하는 8개의 통로를 순서대로 채웁니다.
        # 순서: Age, Gender, Sleep, Study, Social, Physical, Stress, (Target Dummy)
        # 이미지(16d820.png)의 순서를 100% 반영했습니다.
        raw_data = [21.0, 1.0, float(sleep), 5.0, float(social), 3.0, float(stress), 0] # 총 8개
        
        # 모델의 규격(Domain)에 맞춰 변환
        inst = Orange.data.Instance(model.domain, raw_data)
        
        # 예측 및 확률 계산
        prediction = model(inst)
        probs = model(inst, ret=Orange.classification.Model.ValueProbs)
        
        # [에러 방지] 확률값을 안전하게 숫자로 변환
        risk_percent = float(probs[1]) * 100

        # 4. 결과 출력
        st.divider()
        if prediction == 1 or risk_percent > 50:
            st.error(f"⚠️ 현재 마음 날씨는 '흐림'입니다. (위험 확률: {risk_percent:.1f}%)")
        else:
            st.success(f"☀️ 현재 마음 날씨는 '맑음'입니다. (안정 확률: {100-risk_percent:.1f}%)")

    except Exception as e:
        st.error(f"분석 중 오류 발생: {e}")
