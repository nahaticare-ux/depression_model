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
st.write("오렌지3 모델 규격(입력 7개 + 타겟 1개)을 맞춘 분석 도구입니다.")

# 2. 사용자 입력 받기 (Features: 7개)
age = st.number_input("나이", 15, 40, 20)
gender = st.selectbox("성별", options=[0, 1], format_func=lambda x: "남성" if x==0 else "여성")
sleep = st.slider("수면 시간", 0.0, 15.0, 7.0)
study = st.slider("공부 시간", 0.0, 15.0, 5.0)
social = st.slider("SNS 사용 시간", 0.0, 15.0, 2.0)
physical = st.slider("운동 시간", 0.0, 10.0, 1.0)
stress = st.slider("스트레스 지수", 1, 10, 5)

# 3. 예측 실행
if st.button("예보 확인하기"):
    try:
        # [핵심 설명] 모델은 8개의 데이터를 원합니다. 
        # (7개의 입력 데이터) + (1개의 타겟 데이터 자리)
        # 타겟 자리에는 아무 숫자(0)나 넣어서 칸을 채워줍니다.
        input_list = [age, gender, sleep, study, social, physical, stress, 0] # 총 8개
        
        # Orange 인스턴스 생성 (모델의 규격인 domain을 그대로 사용)
        inst = Orange.data.Instance(model.domain, input_list)
        
        # 결과 계산
        prediction = model(inst)
        probs = model(inst, ret=Orange.classification.Model.ValueProbs)
        
        risk_score = float(probs[1]) * 100 

        # 4. 결과 출력
        st.divider()
        if prediction == 1:
            st.error(f"⚠️ 마음 날씨 '흐림' (우울 위험 확률: {risk_score:.1f}%)")
        else:
            st.success(f"☀️ 마음 날씨 '맑음' (안정 확률: {100 - risk_score:.1f}%)")

    except Exception as e:
        st.error(f"오류가 발생했습니다: {e}")
        st.info("모델이 요구하는 8번째 칸(타겟 변수 자리)까지 데이터를 채웠는지 확인하세요.")
