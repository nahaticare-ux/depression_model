import streamlit as st
import pickle
import numpy as np

# 1. 저장한 모델 불러오기
def load_model():
    with open('depression_model.pkcls', 'rb') as f:
        data = pickle.load(f)
    return data

model_data = load_model()

# Orange3 모델 파일 구조에 따라 모델 추출
# 만약 model_data가 바로 모델 객체가 아니라면 아래와 같이 처리합니다.
if hasattr(model_data, 'model'):
    model = model_data.model
else:
    model = model_data

# 2. 앱 UI 꾸미기
st.title("🌱 청소년 마음건강 지킴이")

# 3. 사용자 입력 받기 (학습시킨 7개 Feature 순서와 동일해야 합니다)
# Age, Gender, Sleep_Duration, Study_Hours, Social_Media, Physical_Activity, Stress_Level
age = st.number_input("나이", min_value=13, max_value=19, value=17)
gender = st.selectbox("성별", ["Female", "Male"]) # Orange3는 알파벳 순서(0: Female, 1: Male)
sleep = st.number_input("하루 평균 수면 시간 (시간)", 0, 12, 7)
study = st.number_input("하루 평균 학습 시간 (시간)", 0, 15, 5)
media = st.number_input("소셜 미디어 사용 시간 (시간)", 0, 10, 2)
active = st.number_input("신체 활동 시간 (분)", 0, 120, 30)
stress = st.slider("현재 느끼는 스트레스 지수", 1, 5, 3)

# 4. 예측 실행
if st.button("결과 확인하기"):
    gender_val = 1 if gender == "Male" else 0
    
    # 입력 데이터를 리스트로 만들기
    features = [age, gender_val, sleep, study, media, active, stress]
    
    # 예측 수행
    prediction = model.predict([features])
    
    st.divider()
    
    # 결과 출력 (Orange3에서 Depression의 Target 값이 True/False이므로)
    if prediction[0] == "True" or prediction[0] == 1:
        st.warning("⚠️ 마음이 조금 지쳐 있는 것 같아요.")
        st.info("💡 처방전: 오늘 밤은 1시간만 일찍 자고, 좋아하는 음악을 들어보는 건 어떨까요?")
    else:
        st.success("✅ 마음이 아주 건강한 상태입니다!")
        st.info("💡 유지 팁: 지금처럼 규칙적인 생활을 이어가면 아주 좋아요!")
