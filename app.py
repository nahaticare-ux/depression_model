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
st.write("우리 반 인공지능 모델로 마음 날씨를 분석해 봅시다!")

# 2. 사용자 입력 받기 (7개 지표 모두 나열)
# 학생들에게 "인공지능은 학습한 순서대로 정보를 주어야 이해한다"고 설명해주세요.
age = st.number_input("나이", 15, 30, 20)
gender = st.selectbox("성별", options=[0, 1], format_func=lambda x: "남성" if x==0 else "여성")
sleep = st.slider("평균 수면 시간 (시간)", 0.0, 12.0, 7.0)
study = st.slider("평균 공부 시간 (시간)", 0.0, 12.0, 5.0)
social = st.slider("SNS 사용 시간 (시간)", 0.0, 12.0, 2.0)
physical = st.slider("운동 시간 (시간)", 0.0, 5.0, 1.0)
stress = st.slider("스트레스 정도 (1~10)", 1, 10, 5)

# 3. 예측하기 버튼
if st.button("결과 확인하기"):
    try:
        # [핵심] Orange3 Select Columns 위젯의 순서와 개수를 100% 일치시킵니다.
        # 순서: Age, Gender, Sleep, Study, Social, Physical, Stress
        # 마지막에 Target(Depression) 자리를 위해 0을 하나 추가하여 총 8개를 만듭니다.
        input_data = [age, gender, sleep, study, social, physical, stress, 0]
        
        # 모델 규격에 맞게 데이터 인스턴스 생성
        inst = Orange.data.Instance(model.domain, input_data)
        
        # 예측 및 확률 계산
        prediction = model(inst)
        probs = model(inst, ret=Orange.classification.Model.ValueProbs)
        
        # 확률값 추출 (에러 방지를 위해 float으로 변환)
        risk_score = float(probs[1]) * 100 

        # 4. 결과 보여주기
        st.divider()
        if prediction == 1:
            st.error(f"⚠️ 현재 마음 날씨: '흐림' (우울 위험 확률: {risk_score:.1f}%)")
            st.write("조금 쉬어가도 괜찮아요. 따뜻한 차 한 잔 어떨까요?")
        else:
            st.success(f"☀️ 현재 마음 날씨: '맑음' (안정 확률: {100 - risk_score:.1f}%)")
            st.write("마음 기상이 아주 좋습니다! 멋진 하루 보내세요.")

    except Exception as e:
        st.error(f"데이터 입력 오류: {e}")
        st.info("입력 데이터의 개수나 형식이 모델과 맞지 않습니다.")
