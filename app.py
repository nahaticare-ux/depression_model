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
st.write("오렌지3 모델의 7개 지표를 사용하여 분석합니다.")

# 2. 사용자 입력 받기 (7개 Feature 모두 나열)
age = st.number_input("나이 (Age)", 15, 40, 20)
gender = st.selectbox("성별 (Gender)", options=[0, 1], format_func=lambda x: "남성" if x==0 else "여성")
sleep = st.slider("수면 시간 (Sleep_Duration)", 0, 15, 7)
study = st.slider("공부 시간 (Study_Hours)", 0, 15, 5)
social = st.slider("SNS 사용 시간 (Social_Media_Hours)", 0, 15, 2)
physical = st.slider("운동 시간 (Physical_Activity)", 0, 10, 1)
stress = st.slider("스트레스 지수 (Stress_Level)", 1, 10, 5)

# 3. 예측하기
if st.button("예보 확인하기"):
    try:
        # [중요] 오렌지3 위젯 순서와 동일하게 리스트를 만듭니다 (7개 Feature + 1개 Target)
        # 이미지(16d820.png)에 정의된 순서를 100% 지킵니다.
        raw_input = [age, gender, sleep, study, social, physical, stress, 0]
        
        # 모델 규격(Domain)에 맞춰 데이터 변환
        inst = Orange.data.Instance(model.domain, raw_input)
        
        # 결과 계산
        prediction = model(inst)
        probs = model(inst, ret=Orange.classification.Model.ValueProbs)
        
        # 확률값 추출 (에러 방지를 위해 확실히 float으로 변환)
        risk_prob = float(probs[1]) * 100

        # 4. 결과 출력
        st.divider()
        if prediction == 1 or risk_prob > 50:
            st.error(f"⚠️ 현재 마음 날씨: '흐림' (우울 위험 확률: {risk_prob:.1f}%)")
        else:
            st.success(f"☀️ 현재 마음 날씨: '맑음' (안정 확률: {100-risk_prob:.1f}%)")

    except Exception as e:
        st.error(f"예측 오류: {e}")
        st.info("데이터 개수나 순서가 모델 설정과 다릅니다. 코드를 확인해 주세요.")
