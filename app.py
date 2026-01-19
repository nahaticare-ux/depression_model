import streamlit as st
import Orange
import pickle
import pandas as pd

# 1. 모델 불러오기
# Orange3에서 저장한 모델 파일을 로드합니다.
with open("depression_model.pkcls", "rb") as f:
    model = pickle.load(f)

st.title("☁️ 마음기상청: 대학생 우울증 예보")
st.write("오늘 당신의 마음 날씨를 확인해 보세요.")

# 2. 사용자 입력 받기 (우리가 학습시킨 중요 변수들 위주)
# Rank 위젯에서 상위권이었던 항목들을 슬라이더로 만듭니다.
stress = st.slider("오늘 스트레스 정도 (1~10)", 1, 10, 5) #
sleep = st.number_input("어제 수면 시간 (시간)", 0.0, 24.0, 7.0) #
social = st.number_input("SNS 사용 시간 (시간)", 0.0, 24.0, 2.0) #

# 3. 예측하기
if st.button("예보 확인하기"):
    # 입력 데이터를 모델 형식에 맞게 변환
    # (주의: 학습할 때 썼던 모든 Feature의 순서를 지켜야 합니다)
    input_data = [[stress, sleep, social]] 
    
    # 모델 예측
    prediction = model(input_data)
    prob = model(input_data, ret=Orange.classification.Model.ValueProbs)

    # 4. 결과 보여주기
    if prediction[0] == 1: # True (우울 위험)
        st.error(f"⚠️ 현재 마음 날씨는 '흐림'입니다. (위험 확률: {prob[0][1]*100:.1f}%)")
        st.write("조금 쉬어가는 건 어떨까요? 전문가와의 상담을 추천드려요.")
    else:
        st.success(f"☀️ 현재 마음 날씨는 '맑음'입니다. (안정 확률: {prob[0][0]*100:.1f}%)")
        st.write("아주 잘하고 있어요! 지금처럼 건강한 습관을 유지하세요.")