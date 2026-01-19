import streamlit as st
import Orange
import pickle
import warnings
import numpy as np

# 1. 환경 차이에 따른 불필요한 경고 메시지 차단
warnings.filterwarnings("ignore")

# 2. 인공지능 모델 로드 함수 (캐싱 적용)
@st.cache_resource
def load_ai_model():
    # 깃허브에 업로드한 파일명과 대소문자까지 일치해야 합니다.
    file_name = "depression_model.pkcls"
    with open(file_name, "rb") as f:
        return pickle.load(f)

# 모델 불러오기 실행
try:
    model = load_ai_model()
except Exception as e:
    st.error(f"모델 파일을 찾을 수 없습니다: {e}")
    st.stop()

# 3. 웹 페이지 UI 구성
st.set_page_config(page_title="마음기상청", page_icon="☁️")
st.title("☁️ 마음기상청: 대학생 우울증 예보")
st.write("인공지능(랜덤 포레스트)이 당신의 일상을 분석하여 마음의 날씨를 알려드립니다.")

# 4. 입력 섹션
st.divider()
col1, col2 = st.columns(2)

with col1:
    stress = st.slider("🔥 오늘 스트레스 정도 (1~10)", 1, 10, 5)
    sleep = st.number_input("😴 어제 수면 시간 (0~24시간)", 0.0, 24.0, 7.0)

with col2:
    social = st.number_input("📱 SNS 사용 시간 (0~24시간)", 0.0, 24.0, 2.0)
    st.write(" ") 
    st.write("💡 모든 수치를 입력 후 아래 버튼을 눌러주세요.")

# 5. 분석 및 결과 출력
if st.button("마음 날씨 예보하기"):
    try:
        # 모델 규격에 맞는 9개의 데이터 생성
        # 순서: Age(21), Gender(1), Sleep, Study(5), Social, Physical(3), Stress, Target(0), Meta(0)
        input_list = [21.0, 1.0, float(sleep), 5.0, float(social), 3.0, float(stress), 0, 0]
        
        # Orange 데이터 인스턴스로 변환
        instance = Orange.data.Instance(model.domain, input_list)
        
        # 예측 및 확률 계산
        prediction = model(instance)
        probs = model(instance, ret=Orange.classification.Model.ValueProbs)
        
        # [해결] Scalar 변환 오류 방지: 리스트 형태의 결과값을 안전하게 숫자로 변환합니다.
        if hasattr(prediction, "__len__"):
            final_pred = int(prediction[0])
        else:
            final_pred = int(prediction)
            
        # 확률값도 안전하게 리스트에서 추출합니다.
        if hasattr(probs, "__len__"):
            risk_percent = float(probs[1]) * 100
        else:
            risk_percent = float(probs) * 100

        # 결과 리포트 출력
        st.divider()
        if final_pred == 1:
            st.error(f"⚠️ 예보 결과: '흐림' (우울 위험 확률: {risk_percent:.1f}%)")
            st.info("조금 쉬어가도 괜찮아요. 친구나 상담 센터와 이야기를 나누어 보세요.")
        else:
            st.success(f"☀️ 예보 결과: '맑음' (마음 안정 확률: {100-risk_percent:.1f}%)")
            st.balloons() # 성공 축하 풍선 효과

    except Exception as error:
        st.error(f"분석 엔진 작동 중 오류가 발생했습니다: {error}")
