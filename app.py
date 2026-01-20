import streamlit as st
import pickle
import numpy as np

# 1. 모델 불러오기
def load_model():
    with open('depression_model.pkcls', 'rb') as f:
        data = pickle.load(f)
    return data

model_data = load_model()
model = model_data.model if hasattr(model_data, 'model') else model_data

# 2. 앱 UI
st.title("🌱 청소년 마음건강 지킴이")

# 3. 사용자 입력 (8개 항목을 정확한 순서대로 배치)
age = st.number_input("나이", 13, 19, 17)
gender = st.selectbox("성별", ["Female", "Male"])
# 학과(Department) 추가 - 데이터셋의 범주에 맞춰 숫자로 변환 필요
dept = st.selectbox("전공/계열", ["Arts", "Business", "Engineering", "Medical", "Science"])
sleep = st.number_input("하루 평균 수면 시간 (시간)", 0, 12, 7)
study = st.number_input("하루 평균 학습 시간 (시간)", 0, 15, 5)
media = st.number_input("소셜 미디어 사용 시간 (시간)", 0, 10, 2)
active = st.number_input("신체 활동 시간 (분)", 0, 120, 30)
stress = st.slider("현재 느끼는 스트레스 지수", 1, 5, 3)

# 4. 예측
if st.button("결과 확인하기"):
    # 범주형 데이터 변환 (Orange3 내부 변환 방식에 맞춰야 함)
    gender_val = 1 if gender == "Male" else 0
    dept_dict = {"Arts": 0, "Business": 1, "Engineering": 2, "Medical": 3, "Science": 4}
    dept_val = dept_dict[dept]
    
    # 8개의 특징(Feature)을 리스트로 생성
    features = [age, gender_val, dept_val, sleep, study, media, active, stress]
    
    # 예측 수행
    prediction = model.predict([features])
    
    st.divider()
    
    # 결과 출력
    if str(prediction[0]) == "True" or prediction[0] == 1:
        st.warning("⚠️ 마음이 조금 지쳐 있는 것 같아요.")
        st.info("💡 처방전: 오늘 밤은 1시간만 일찍 자고, 좋아하는 음악을 들어보세요!")
    else:
        st.success("✅ 마음이 아주 건강한 상태입니다!")
