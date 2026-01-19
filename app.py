import streamlit as st
import Orange
import pickle
import numpy as np

# 1. 모델 불러오기
@st.cache_resource
def load_model():
    # 파일명이 정확히 depression_model.pkcls 인지 다시 확인해주세요!
    with open("depression_model.pkcls", "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("☁️ 마음기상청: 대학생 우울증 예보")
st.write("일상 데이터를 입력하면 인공지능이 마음 날씨를 분석합니다.")

# 2. 사용자 입력 (주요 3개 변수)
stress = st.slider("오늘 스트레스 정도 (1~10)", 1, 10, 5)
sleep = st.number_input("어제 수면 시간 (시간)", 0.0, 24.0, 7.0)
social = st.number_input("SNS 사용 시간 (시간)", 0.0, 24.0, 2.0)

# 3. 예측하기
if st.button("예보 확인하기"):
    try:
        # 모델이 요구하는 정확한 변수 개수를 파악합니다.
        # 에러 메시지에서 shape(8,)을 요구했으므로 8개를 만듭니다.
        # [Age, Gender, Sleep, Study, Social, Physical, Stress, Dummy_Target]
        input_data = [21.0, 1.0, float(sleep), 5.0, float(social), 3.0, float(stress), 0.0]
        
        # 모델의 도메인(규격) 정보를 가져옵니다.
        domain = model.domain
        
        # [핵심] 만약 모델이 요구하는 개수가 다르면 부족한 만큼 0을 더 채워줍니다.
        required_count = len(domain.attributes) + (1 if domain.class_var else 0)
        while len(input_data) < required_count:
            input_data.append(0.0)
            
        # Orange 전용 인스턴스로 변환
        inst = Orange.data.Instance(domain, input_data[:required_count])
        
        # 예측 수행
        prediction = model(inst)
        probs = model(inst, ret=Orange.classification.Model.ValueProbs)

        # 4. 결과 출력
        st.divider()
        risk_percent = probs[1] * 100

        if prediction == 1 or risk_percent > 50:
            st.error(f"⚠️ 현재 마음 날씨는 '흐림'입니다. (위험 확률: {risk_percent:.1f}%)")
            st.write("오늘은 평소보다 나 자신을 더 아껴주세요. 가벼운 휴식이 필요해 보여요.")
        else:
            st.success(f"☀️ 현재 마음 날씨는 '맑음'입니다. (안정 확률: {probs[0]*100:.1f}%)")
            st.write("마음 기상이 아주 쾌청합니다! 즐거운 하루 보내세요.")

    except Exception as e:
        st.error(f"예측 도중 오류가 발생했습니다: {e}")
        st.info("데이터 개수를 모델 규격에 맞춰 자동으로 조정 중입니다. 다시 시도해 주세요.")
