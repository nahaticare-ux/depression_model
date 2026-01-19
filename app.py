import streamlit as st
import Orange
import pickle
import pandas as pd

# 1. 모델 불러오기
try:
    with open("depression_model.pkcls", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"모델 파일을 찾을 수 없습니다: {e}")

st.title("☁️ 마음기상청: 대학생 우울증 예보")
st.write("오늘 당신의 마음 날씨를 확인해 보세요.")

# 2. 사용자 입력 받기 (주요 변수 3개)
stress = st.slider("오늘 스트레스 정도 (1~10)", 1, 10, 5)
sleep = st.number_input("어제 수면 시간 (시간)", 0.0, 24.0, 7.0)
social = st.number_input("SNS 사용 시간 (시간)", 0.0, 24.0, 2.0)

# 3. 예측하기
if st.button("예보 확인하기"):
    try:
        # [중요] Orange3 'Select Columns'의 Features 순서와 개수를 그대로 맞춰야 합니다.
        # 사용하지 않는 나머지 변수들은 데이터셋의 평균값(혹은 임의의 값)으로 채웁니다.
        # 순서 예시: Stress, Sleep, Social, Physical_Activity, Study_Hours, Age, Gender
        # (Gender의 경우 Male=0, Female=1 등 Orange3가 숫자로 변환한 기준을 따릅니다)
        
        # 7개의 변수 자리를 모두 채운 리스트 생성
        # 아래 값(3.0, 5.0, 21.0, 0)은 데이터셋의 대략적인 평균/기본값입니다.
        input_list = [stress, sleep, social, 3.0, 5.0, 21.0, 0] 
        
        # Orange 모델이 인식할 수 있는 Table 형식으로 변환하여 예측
        # 10만 건 데이터로 학습된 모델이므로 이 과정이 필수적입니다.
        prediction = model([input_list])
        prob = model([input_list], ret=Orange.classification.Model.ValueProbs)

        # 4. 결과 보여주기
        st.divider() # 시각적인 구분선 추가
        
        # 확률값(prob)을 기준으로 결과 출력
        # prob[0][1]은 우울증(True)일 확률을 의미합니다.
        risk_percent = prob[0][1] * 100

        if prediction[0] == 1 or risk_percent > 50: # 우울 위험군
            st.error(f"⚠️ 현재 마음 날씨는 '흐림'입니다. (위험 확률: {risk_percent:.1f}%)")
            st.write("조금 쉬어가는 건 어떨까요? 따뜻한 차 한 잔과 함께 휴식을 취해보세요.")
        else:
            st.success(f"☀️ 현재 마음 날씨는 '맑음'입니다. (안정 확률: {100 - risk_percent:.1f}%)")
            st.write("아주 잘하고 있어요! 지금처럼 건강한 마음 습관을 유지하세요.")
            
    except ValueError as e:
        st.error(f"데이터 개수 오류: 모델은 더 많은 입력값을 기다리고 있습니다. (현재 입력 개수 확인 필요)")
        st.info("Orange3의 'Select Columns' 위젯에서 Features에 넣었던 항목이 총 몇 개인지 확인해 보세요.")
    except Exception as e:
        st.error(f"예측 중 오류가 발생했습니다: {e}")
