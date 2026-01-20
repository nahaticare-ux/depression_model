import streamlit as st
import pickle
import numpy as np

# 1. 저장한 모델 불러오기
def load_model():
    with open('depression_model.pkcls', 'rb') as f:
        model = pickle.load(f)
    return model

model_data = load_model()
model = model_data.model # Orange3 저장 파일에서 실제 모델 객체 추출

# 2. 앱 UI 꾸미기
st.title("🌱 청소년 마음건강 지킴이")
st.subheader("여러분의 생활 습관을 통해 현재 마음 상태를 확인해보세요.")

# 3. 사용자 입력 받기 (오렌지3에서 Feature로 설정했던 항목들)
age = st.number_input("나이", min_value=13, max_value=19, value=17)
gender = st.selectbox("성별", ["Male", "Female"])
sleep = st.slider("하루 평균 수면 시간 (시간)", 0, 12, 7)
study = st.slider("하루 평균 학습 시간 (시간)", 0, 15, 5)
media = st.slider("소셜 미디어 사용 시간 (시간)", 0, 10, 2)
active = st.slider("신체 활동 시간 (분)", 0, 120, 30)
stress = st.select_slider("현재 느끼는 스트레스 지수", options=[1, 2, 3, 4, 5])

# 4. 예측 및 피드백
if st.button("결과 확인하기"):
    # 성별을 숫자로 변환 (Orange3 학습 시 설정에 맞춰야 함)
    gender_val = 1 if gender == "Male" else 0
    
    # 입력 데이터를 모델 형식으로 변환
    input_data = np.array([[age, gender_val, sleep, study, media, active, stress]])
    prediction = model.predict(input_data)
    
    st.divider()
    
    if prediction[0] == True:
        st.warning("⚠️ 마음이 조금 지쳐 있는 것 같아요.")
        st.write("### 💡 힐링 처방법")
        st.write("- **잠깐의 휴식:** 오늘 밤은 평소보다 1시간 일찍 자보는 건 어떨까요?")
        st.write("- **가벼운 산책:** 10분만 햇볕을 쬐며 걸어보세요. 기분이 훨씬 좋아질 거예요.")
    else:
        st.success("✅ 마음이 아주 건강한 상태입니다!")
        st.write("### 💡 건강 유지 팁")
        st.write("- 지금처럼 규칙적인 수면과 활동량을 유지해 주세요!")