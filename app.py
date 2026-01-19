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
st.write("모델이 요구하는 9개의 데이터 규격을 확인했습니다. 분석을 시작합니다.")

# 2. 사용자 입력 받기
stress = st.slider("오늘 스트레스 정도 (1~10)", 1, 10, 5)
sleep = st.number_input("어제 수면 시간 (시간)", 0.0, 24.0, 7.0)
social = st.number_input("SNS 사용 시간 (시간)", 0.0, 24.0, 2.0)

# 3. 분석하기
if st.button("마음 날씨 예보하기"):
    try:
        # [해결포인트] 모델이 요구하는 9개의 칸을 정확한 순서로 채웁니다.
        # 순서: Age, Gender, Sleep, Study, Social, Physical, Stress, (Target), (Meta/Extra)
        # 앞선 7개 지표 뒤에 0을 두 개 더 붙여 총 9개를 만듭니다.
        input_list = [21.0, 1.0, float(sleep), 5.0, float(social), 3.0, float(stress), 0, 0] # 총 9개
        
        # Orange 인스턴스 생성
        inst = Orange.data.Instance(model.domain, input_list)
        
        # 신경망 예측 수행
        prediction = model(inst)
        probs = model(inst, ret=Orange.classification.Model.ValueProbs)
        
        # [중요] 신경망의 확률값(numpy array)을 단일 숫자로 변환하여 에러 방지
        risk_prob = float(probs[0][1]) * 100

        # 4. 결과 출력
        st.divider()
        if prediction[0] == 1 or risk_prob > 50:
            st.error(f"⚠️ 현재 마음 날씨는 '흐림'입니다. (우울 위험 확률: {risk_prob:.1f}%)")
        else:
            st.success(f"☀️ 현재 마음 날씨는 '맑음'입니다. (안정 확률: {100-risk_prob:.1f}%)")

    except Exception as e:
        st.error(f"분석 중 오류 발생: {e}")
        st.info("데이터 규격을 9개로 맞췄음에도 오류가 난다면, 모델의 입구 순서를 다시 확인해야 합니다.")
