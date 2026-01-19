import streamlit as st
import Orange
import pickle

# 1. 모델 불러오기 (캐싱 제외하여 직관적으로 구성)
with open("depression_model.pkcls", "rb") as f:
    model = pickle.load(f)

st.title("☁️ 마음기상청: 대학생 우울증 예보")
st.write("우리 반 인공지능 모델로 마음 날씨를 확인해 봅시다!")

# 2. 사용자 입력 받기 (설명하기 쉽게 7개 모두 받거나 기본값 고정)
# 학생들에게 "인공지능은 우리가 학습시킨 순서를 지켜야 한다"고 설명하기 좋습니다.
age = st.number_input("나이", 15, 30, 20)
gender = st.selectbox("성별", options=[0, 1], format_func=lambda x: "남성" if x==0 else "여성")
sleep = st.slider("평균 수면 시간", 0.0, 12.0, 7.0)
study = st.slider("평균 공부 시간", 0.0, 12.0, 5.0)
social = st.slider("SNS 사용 시간", 0.0, 12.0, 2.0)
physical = st.slider("운동 시간", 0.0, 5.0, 1.0)
stress = st.select_slider("스트레스 정도", options=list(range(1, 11)), value=5)

# 3. 예측하기 버튼
if st.button("결과 확인하기"):
    # [수업 포인트] "학습 데이터셋의 순서와 똑같이 리스트를 만듭니다."
    # 순서: Age, Gender, Sleep, Study, Social, Physical, Stress
    input_list = [age, gender, sleep, study, social, physical, stress]
    
    # Orange 모델 규격에 맞게 변환 (Target 자리에 더미 0 추가하여 총 8개)
    # image_16d820.png에서 Target이 포함된 규격을 요구하므로 8개를 채웁니다.
    data_to_predict = input_list + [0] 
    
    inst = Orange.data.Instance(model.domain, data_to_predict)
    
    # 결과 계산
    prediction = model(inst)
    probs = model(inst, ret=Orange.classification.Model.ValueProbs)
    
    # 4. 결과 보여주기
    st.divider()
    risk_score = float(probs[1]) * 100 # 우울 위험 확률 추출

    if prediction == 1:
        st.error(f"⚠️ 현재 마음 날씨: '흐림' (우울 위험 확률: {risk_score:.1f}%)")
    else:
        st.success(f"☀️ 현재 마음 날씨: '맑음' (안정 확률: {100 - risk_score:.1f}%)")
