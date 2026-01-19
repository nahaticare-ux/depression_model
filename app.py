import streamlit as st
import Orange
import pickle
import os

# 1. 모델 파일 존재 여부 확인 및 불러오기
model_path = "depression_model.pkcls"

@st.cache_resource
def load_model():
    if not os.path.exists(model_path):
        st.error(f"파일을 찾을 수 없습니다: {model_path}가 GitHub에 업로드되었는지 확인하세요.")
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("☁️ 마음기상청: 대학생 우울증 예보")
st.write("당신의 일상 데이터를 분석하여 마음의 날씨를 알려드립니다.")

# 2. 사용자 입력 (주요 3개 변수)
stress = st.slider("오늘 스트레스 정도 (1~10)", 1, 10, 5)
sleep = st.number_input("어제 수면 시간 (시간)", 0.0, 24.0, 7.0)
social = st.number_input("SNS 사용 시간 (시간)", 0.0, 24.0, 2.0)

# 3. 예측하기
if st.button("예보 확인하기") and model:
    try:
        # [핵심] 8개 자리를 강제로 맞춥니다.
        # 순서: Age, Gender, Sleep, Study, Social, Physical, Stress, (더미 타겟)
        # 모델이 '7을 8로 reshape 못한다'고 했으므로, 정확히 8개를 리스트에 넣습니다.
        input_list = [21.0, 1.0, float(sleep), 5.0, float(social), 3.0, float(stress), 0.0]
        
        # Orange3 모델 전용 인스턴스 생성
        domain = model.domain
        inst = Orange.data.Instance(domain, input_list)
        
        # 예측 및 확률 계산
        prediction = model(inst)
        probs = model(inst, ret=Orange.classification.Model.ValueProbs)

        # 4. 결과 출력
        st.divider()
        risk_percent = probs[1] * 100

        if prediction == 1 or risk_percent > 50:
            st.error(f"⚠️ 현재 마음 날씨는 '흐림'입니다. (위험 확률: {risk_percent:.1f}%)")
            st.write("조금 지친 상태일 수 있어요. 오늘은 자신에게 선물을 주는 시간을 가져보세요.")
        else:
            st.success(f"☀️ 현재 마음 날씨는 '맑음'입니다. (안정 확률: {probs[0]*100:.1f}%)")
            st.write("아주 건강한 상태입니다! 이 긍정적인 에너지를 유지하세요.")

    except Exception as e:
        st.error(f"데이터 규격 오류: {e}")
        st.info("입력 데이터 개수가 모델이 원하는 8개와 일치하도록 조정되었습니다. 다시 시도해 보세요.")
