import streamlit as st
import Orange
import pickle
import warnings

# 1. 지저분한 환경 경고 메시지 숨기기
warnings.filterwarnings("ignore")

# 2. 모델 불러오기
@st.cache_resource
def load_model():
    # 파일명이 depression_model.pkcls 인지 확인하세요!
    with open("depression_model.pkcls", "rb") as f:
        return pickle.load(f)

model = load_model()

# 3. 페이지 디자인
st.set_page_config(page_title="마음기상청", page_icon="☁️")
st.title("☁️ 마음기상청: 대학생 우울증 예보")
st.write("인공지능(랜덤 포레스트)이 당신의 일상을 분석하여 마음의 날씨를 알려드립니다.")

# 4. 사용자 입력 받기 (수업용 핵심 변수 3개)
st.divider()
stress = st.slider("🔥 오늘 스트레스 정도 (1~10)", 1, 10, 5)
sleep = st.number_input("😴 어제 수면 시간 (시간)", 0.0, 24.0, 7.0)
social = st.number_input("📱 SNS 사용 시간 (시간)", 0.0, 24.0, 2.0)

# 5. 분석하기 버튼
if st.button("마음 날씨 예보하기"):
    try:
        # [해결] 모델이 요구하는 9개의 칸을 정확한 순서로 채웁니다.
        # 순서: Age(21), Gender(1), Sleep, Study(5), Social, Physical(3), Stress, Target(0), Meta(0)
        input_list = [21.0, 1.0, float(sleep), 5.0, float(social), 3.0, float(stress), 0, 0]
        
        # 데이터 형식 변환
        inst = Orange.data.Instance(model.domain, input_list)
        
        # 예측 수행
        prediction = model(inst)
        probs = model(inst, ret=Orange.classification.Model.ValueProbs)
        
        # [해결] 결과값 추출 (랜덤 포레스트용)
        # prediction[0]은 예측 클래스(0 또는 1), probs[1]은 우울증일 확률입니다.
        pred_value = int(prediction[0])
        risk_prob = float(probs[1]) * 100

        # 6. 결과 출력
        st.divider()
        if pred_value == 1:
            st.error(f"⚠️ 현재 마음 날씨는 '흐림'입니다. (우울 위험 확률: {risk_prob:.1f}%)")
            st.write("조금 쉬어가도 괜찮아요. 친구나 상담 센터와 이야기를 나눠보는 건 어떨까요?")
        else:
            st.success(f"☀️ 현재 마음 날씨는 '맑음'입니다. (안정 확률: {100-risk_prob:.1f}%)")
            st.write("당신의 마음은 아주 건강한 상태입니다! 오늘 하루도 화이팅하세요.")

    except Exception as e:
        st.error(f"분석 중 오류가 발생했습니다: {e}")
        st.info("모델 파일(.pkcls)이 깃허브에 정상적으로 업로드되었는지 확인해 주세요.")
