import streamlit as st
import Orange
import pickle
import numpy as np

# 1. 모델 불러오기
@st.cache_resource
def load_model():
    with open("depression_model.pkcls", "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("☁️ 마음기상청: 신경망(AI) 우울증 예보")
st.write("모델이 요구하는 정확한 데이터 규격을 자동으로 맞춰 분석합니다.")

# 2. 사용자 입력 (수업용 핵심 데이터)
stress = st.slider("스트레스 지수 (1~10)", 1, 10, 5)
sleep = st.slider("평균 수면 시간", 0.0, 15.0, 7.0)
social = st.slider("SNS 사용 시간", 0.0, 15.0, 2.0)

# 3. 예측하기
if st.button("마음 날씨 예보하기"):
    try:
        # [해결 포인트] 모델의 도메인에서 필요한 총 칸수를 읽어옵니다.
        domain = model.domain
        total_required = len(domain.variables) + len(domain.metas)
        
        # 넉넉하게 20칸짜리 기본 데이터를 만들고 0으로 채웁니다.
        # 그 후, 우리가 받은 입력값들을 적절한 위치에 넣습니다.
        test_data = [0] * 20
        test_data[0] = 21 # Age (가정)
        test_data[1] = 1  # Gender (가정)
        test_data[2] = sleep
        test_data[6] = stress
        
        # [핵심] 모델이 원하는 개수만큼만 딱 잘라서 보냅니다.
        final_input = test_data[:total_required]
        
        inst = Orange.data.Instance(domain, final_input)
        
        # 예측 수행
        prediction = model(inst)
        probs = model(inst, ret=Orange.classification.Model.ValueProbs)
        
        # 결과 출력
        risk_percent = float(probs[1]) * 100
        st.divider()
        if prediction == 1:
            st.error(f"⚠️ 현재 마음 날씨: '흐림' (우울 위험 확률: {risk_percent:.1f}%)")
        else:
            st.success(f"☀️ 현재 마음 날씨: '맑음' (안정 확률: {100-risk_percent:.1f}%)")

    except Exception as e:
        st.error(f"분석 실패: {e}")
        st.info(f"모델 요구 칸수: {len(domain.variables) + len(domain.metas)}개")
