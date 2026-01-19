import streamlit as st
import Orange
import pickle

# 1. 모델 불러오기
@st.cache_resource
def load_model():
    # 파일 경로가 depression_model.pkcls 인지 다시 한번 확인하세요!
    with open("depression_model.pkcls", "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("☁️ 마음기상청: 신경망(AI) 우울증 예보")
st.write("모델이 요구하는 9개의 데이터 규격을 맞추어 정밀 분석합니다.")

# 2. 사용자 입력 받기
stress = st.slider("오늘 스트레스 정도 (1~10)", 1, 10, 5)
sleep = st.number_input("어제 수면 시간 (시간)", 0.0, 24.0, 7.0)
social = st.number_input("SNS 사용 시간 (시간)", 0.0, 24.0, 2.0)

# 3. 분석하기
if st.button("마음 날씨 예보하기"):
    try:
        # [해결 1] 모델이 요구하는 9개의 칸을 정확히 채웁니다.
        # 순서: Age, Gender, Sleep, Study, Social, Physical, Stress, (Target), (Meta)
        # 이미지(16d820.png)의 순서대로 7개 + 가짜 데이터 2개를 넣어 9개를 만듭니다.
        input_data = [21.0, 1.0, float(sleep), 5.0, float(social), 3.0, float(stress), 0, 0]
        
        # Orange 인스턴스 생성
        inst = Orange.data.Instance(model.domain, input_data)
        
        # [해결 2] 신경망 모델의 예측값 처리
        # 신경망은 결과값을 [결과] 형태의 배열로 주므로 [0]을 붙여 알맹이만 꺼냅니다.
        prediction_raw = model(inst)
        prediction = int(prediction_raw[0]) #
        
        # 확률값 처리 (probs[0][1] 형태로 접근해야 안전합니다)
        probs = model(inst, ret=Orange.classification.Model.ValueProbs)
        risk_prob = float(probs[0][1]) * 100

        # 4. 결과 출력
        st.divider()
        if prediction == 1:
            st.error(f"⚠️ 현재 마음 날씨는 '흐림'입니다. (우울 위험 확률: {risk_prob:.1f}%)")
        else:
            st.success(f"☀️ 현재 마음 날씨는 '맑음'입니다. (안정 확률: {100-risk_prob:.1f}%)")

    except Exception as e:
        st.error(f"분석 중 오류 발생: {e}")
        st.info("9개의 데이터 입구는 맞지만, 결과값을 꺼내는 방식(Index)을 조정했습니다.")
